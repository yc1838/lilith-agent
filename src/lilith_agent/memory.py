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
    Extracts new facts and merges them with existing ones using an LLM.
    Implements the 'Engram' / 'HCA' active compression pattern.
    """
    log.info("[memory] Running active memory compression...")
    try:
        existing_memories = _store.get_all_memories()
        existing_str = json.dumps([m["content"] for m in existing_memories], indent=2, ensure_ascii=False)
        
        conv_parts = []
        for m in messages:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            if m.content:
                content = m.content if isinstance(m.content, str) else str(m.content)
                conv_parts.append(f"{role}: {content[:1000]}")
        conv_str = "\n".join(conv_parts)

        prompt = f"""
You are Lilith's Long-Term Memory Manager. Your goal is to maintain a DENSE, ATOMIC, and ACCURATE set of facts about the user and the environment.

### CURRENT MEMORIES:
{existing_str}

### RECENT CONVERSATION:
{conv_str}

### TASK:
1. Identify any NEW persistent facts, preferences, or entities mentioned in the conversation.
2. Update or resolve contradictions with EXISTING memories.
3. REMOVE redundant or trivial memories.
4. Keep the list concise and focused on high-signal information (e.g., user name, preferences, project details, API keys mentioned).

Output the updated list of ALL persistent facts as a JSON array of strings.
Example: ["User name is Alice", "Project uses Python 3.11"]
If no changes or facts, return the existing list.
"""
        
        response = model.invoke(prompt)
        content = response.content
        if isinstance(content, list): # Handle thinking models
            content = content[-1].get("text", "") if isinstance(content[-1], dict) else str(content[-1])
        
        # Sane JSON extraction
        try:
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                facts = json.loads(content[start:end])
                if isinstance(facts, list):
                    updated = [{"content": f} for f in facts]
                    _store.save_memories(updated)
                    log.info(f"[memory] Saved {len(updated)} facts.")
        except Exception as je:
            log.warning(f"[memory] JSON parse failed: {je}")

    except Exception as e:
        log.error(f"[memory] Extraction failed: {e}")

def summarize_episode(messages: List[BaseMessage], model) -> None:
    """Summarizes the trajectory to help avoid future mistakes."""
    log.info("[memory] Summarizing task episode...")
    try:
        initial_question = ""
        outcome = "success"
        for m in messages:
            if isinstance(m, HumanMessage) and not initial_question:
                initial_question = str(m.content)
            if "ERROR" in str(m.content).upper():
                outcome = "failed/struggled"
                
        conv_parts = []
        for m in messages:
            if m.content:
                conv_parts.append(f"{m.type}: {str(m.content)[:200]}...")
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
        _store.add_episode(initial_question, response.content, outcome)
        log.info("[memory] Episode saved.")
    except Exception as e:
        log.error(f"[memory] Summarization failed: {e}")

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
    except Exception as e:
        log.error(f"[memory] Retrieval failed: {e}")
        return ""
