import os
import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

log = logging.getLogger(__name__)

# Constants
LILITH_HOME = Path(os.getenv("LILITH_HOME", ".lilith"))
MEMORY_DB_PATH = LILITH_HOME / "long_term_memory.sqlite"

def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
                elif "content" in item:
                    nested = _content_to_text(item["content"])
                    if nested:
                        parts.append(nested)
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)

def _memory_content_to_text(content: Any) -> str:
    if hasattr(content, "content"):
        return _content_to_text(content.content)
    return _content_to_text(content)

class MemoryStore:
    def __init__(self, db_path: Path = MEMORY_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    type TEXT DEFAULT 'fact',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    outcome TEXT,
                    created_at TEXT NOT NULL
                )
            """)
        conn.close()

    def get_all_memories(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM memories ORDER BY updated_at DESC")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def save_memories(self, memories: List[Dict[str, Any]]):
        """Replaces the memories table with the provided list (active compression)."""
        conn = sqlite3.connect(str(self.db_path))
        with conn:
            conn.execute("DELETE FROM memories")
            for m in memories:
                now = datetime.now().isoformat()
                conn.execute(
                    "INSERT INTO memories (id, content, type, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (m.get("id", str(hash(m["content"]))), m["content"], m.get("type", "fact"), now, now)
                )
        conn.close()

    def add_episode(self, task: str, summary: str, outcome: str):
        conn = sqlite3.connect(str(self.db_path))
        with conn:
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT INTO episodes (id, task, summary, outcome, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(hash(task + now)), task, summary, outcome, now)
            )
        conn.close()

    def get_recent_episodes(self, limit: int = 3) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

_store = MemoryStore()

def extract_and_compress_facts(messages: List[BaseMessage], model) -> None:
    """
    Extracts new facts and merges them with existing ones using langmem's manager.
    Implements professional reflection and conflict resolution.
    """
    from langmem import create_memory_manager
    log.info("[memory] Running langmem memory management...")
    try:
        # 1. Get existing memories from our local store
        existing_rows = _store.get_all_memories()
        existing_memories = [m["content"] for m in existing_rows]

        # 2. Initialize langmem manager (it's a Runnable)
        # We use the default schema which is basically content strings
        manager = create_memory_manager(model, enable_deletes=True)

        # 3. Invoke manager with history and existing knowledge
        # The manager will return the full updated list of memories
        result = manager.invoke({
            "messages": messages,
            "existing": existing_memories
        })

        # 4. Save results back to our local persistent SQLite
        if isinstance(result, list):
            # The result is a list of ExtractedMemory objects or tuples depending on version
            # Usually it's (id, content) or just objects. We'll be robust here.
            updated_facts = []
            for item in result:
                # result elements can be ExtractedMemory objects or simple dicts
                content = getattr(item, "content", None) or (item[1] if isinstance(item, tuple) else item.get("content"))
                if content:
                    updated_facts.append({"content": _memory_content_to_text(content)})
            
            _store.save_memories(updated_facts)
            log.info(f"[memory] langmem updated store to {len(updated_facts)} facts.")

    except Exception:
        log.exception("[memory] langmem extraction failed")
        # Fallback to summarize episode if manager fails
        summarize_episode(messages, model)

def summarize_episode(messages: List[BaseMessage], model) -> None:
    """Summarizes the trajectory to help avoid future mistakes."""
    log.info("[memory] Summarizing task episode...")
    try:
        initial_question = ""
        outcome = "success"
        for m in messages:
            content = _content_to_text(m.content)
            if isinstance(m, HumanMessage) and not initial_question:
                initial_question = content
            if "ERROR" in content.upper():
                outcome = "failed/struggled"
                
        conv_parts = []
        for m in messages:
            content = _content_to_text(m.content)
            if content:
                conv_parts.append(f"{m.type}: {content[:200]}...")
        conv_str = "\n".join(conv_parts)

        prompt = f"""
Summarize this task trajectory for Lilith's 'Episodic Memory'.
Initial Question: {initial_question}
Outcome: {outcome}

Briefly explain:
1. What was the goal?
2. What tools worked? What failed?
3. What is the 'lesson learned' for next time?

Keep it under 150 words.
"""
        response = model.invoke(prompt)
        _store.add_episode(initial_question, _content_to_text(response.content), outcome)
        log.info("[memory] Episode saved.")
    except Exception:
        log.exception("[memory] Summarization failed")

def retrieve_relevant_context(query: str) -> str:
    """Fetches all facts and recent episodes to inject into the prompt."""
    try:
        facts = _store.get_all_memories()
        episodes = _store.get_recent_episodes(limit=2)
        
        context_parts = []
        if facts:
            fact_lines = "\n".join([f"- {m['content']}" for m in facts])
            context_parts.append(f"<known_facts>\n{fact_lines}\n</known_facts>")
        
        if episodes:
            epi_lines = "\n\n".join([f"Task: {e['task']}\nSummary: {e['summary']}" for e in episodes])
            context_parts.append(f"<past_experiences>\n{epi_lines}\n</past_experiences>")
            
        return "\n\n".join(context_parts)
    except Exception:
        log.exception("[memory] Retrieval failed")
        return ""
