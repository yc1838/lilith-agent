import os
import sqlite3
import logging
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage

log = logging.getLogger(__name__)

# Constants
LILITH_HOME = Path(os.getenv("LILITH_HOME", ".lilith"))
MEMORY_DB_PATH = LILITH_HOME / "long_term_memory.sqlite"
MEMORY_CONTEXT_CHAR_BUDGET = 3000
MIN_MESSAGES_FOR_EXTRACTION = 2
MEMORY_LOG_PREVIEW_CHARS = 300

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


def _preview_for_log(text: str, limit: int = MEMORY_LOG_PREVIEW_CHARS) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _print_memory_log(message: str, *args: Any) -> None:
    text = message % args if args else message
    print(text, flush=True)


class MemoryStore:
    def __init__(self, db_path: Union[Path, str] = MEMORY_DB_PATH):
        self._in_memory = (str(db_path) == ":memory:")
        self.db_path = db_path if self._in_memory else Path(db_path)
        self._mem_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._in_memory:
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._mem_conn
        return sqlite3.connect(str(self.db_path))

    def close(self):
        if self._mem_conn is not None:
            self._mem_conn.close()
            self._mem_conn = None

    def _init_db(self):
        if not self._in_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
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
        if not self._in_memory:
            conn.close()

    def get_all_memories(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM memories ORDER BY updated_at DESC")
        rows = [dict(r) for r in cur.fetchall()]
        if not self._in_memory:
            conn.close()
        return rows

    def save_memories(self, memories: List[Dict[str, Any]], allow_empty: bool = False):
        """Replaces the memories table with the provided list (active compression)."""
        if not memories:
            existing = self.get_all_memories()
            if existing and not allow_empty:
                _print_memory_log("[memory] save_memories called with empty list while %d facts exist — refusing to wipe",
                                  len(existing))
                log.warning("[memory] save_memories called with empty list while %d facts exist — refusing to wipe",
                            len(existing))
                return
        conn = self._connect()
        with conn:
            conn.execute("DELETE FROM memories")
            for m in memories:
                now = datetime.now().isoformat()
                created_at = m.get("created_at", now)
                updated_at = m.get("updated_at", now)
                conn.execute(
                    "INSERT INTO memories (id, content, type, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (m.get("id", str(uuid.uuid4())), m["content"], m.get("type", "fact"), created_at, updated_at)
                )
        if not self._in_memory:
            conn.close()

    def delete_memory_prefix(self, prefix: str) -> int:
        if not prefix:
            return 0
        matching_ids = [
            row["id"]
            for row in self.get_all_memories()
            if row["id"].startswith(prefix)
        ]
        if not matching_ids:
            return 0
        conn = self._connect()
        with conn:
            for memory_id in matching_ids:
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        if not self._in_memory:
            conn.close()
        return len(matching_ids)

    def add_episode(self, task: str, summary: str, outcome: str):
        conn = self._connect()
        with conn:
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT INTO episodes (id, task, summary, outcome, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), task, summary, outcome, now)
            )
        if not self._in_memory:
            conn.close()

    def get_recent_episodes(self, limit: int = 3) -> List[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        if not self._in_memory:
            conn.close()
        return rows

_store = MemoryStore()


def _set_store(store: MemoryStore) -> MemoryStore:
    """Swap the module-level store and return the previous one."""
    global _store
    prev, _store = _store, store
    return prev


@contextmanager
def ephemeral_memory(db_path: Union[str, Path] = ":memory:"):
    """Context manager that replaces the global _store with a fresh, isolated
    MemoryStore for the duration of the block. On exit the store is closed and
    the previous store is restored. Use db_path=':memory:' (default) for tests
    or GAIA benchmarking where cross-contamination must be prevented.

    Example::

        with ephemeral_memory():
            result = graph.invoke(state, config)
            # any memory writes here are discarded on exit
    """
    fresh = MemoryStore(db_path)
    prev = _set_store(fresh)
    try:
        yield fresh
    finally:
        _set_store(prev)
        fresh.close()


def _is_remove_doc(content: Any) -> bool:
    return hasattr(content, "__repr_name__") and content.__repr_name__() == "RemoveDoc"


def extract_and_compress_facts(messages: List[BaseMessage], model) -> None:
    """
    Extracts new facts and merges them with existing ones using langmem's manager.
    Implements professional reflection and conflict resolution.
    """
    from langmem import create_memory_manager
    from langmem.knowledge.extraction import Memory
    _print_memory_log("[memory] Running langmem memory management...")
    log.info("[memory] Running langmem memory management...")
    try:
        # 1. Get existing memories from our local store
        existing_rows = _store.get_all_memories()
        existing_memories = [(m["id"], Memory(content=m["content"])) for m in existing_rows]
        existing_by_content = {m["content"]: m for m in existing_rows}
        existing_by_id = {m["id"]: m for m in existing_rows}

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
            removed_ids = set()
            for item in result:
                item_id = getattr(item, "id", None)
                content = getattr(item, "content", None)
                if isinstance(item, tuple):
                    if len(item) > 0:
                        item_id = item[0]
                    if len(item) > 1 and content is None:
                        content = item[1]
                elif isinstance(item, dict):
                    item_id = item.get("id", item_id)
                    content = item.get("content", content)
                stable_id = str(item_id) if item_id else None
                if _is_remove_doc(content):
                    if stable_id:
                        removed_ids.add(stable_id)
                    continue
                if content:
                    text = _memory_content_to_text(content)
                    existing = existing_by_id.get(stable_id) if stable_id else existing_by_content.get(text)
                    fact = {"content": text}
                    if stable_id:
                        fact["id"] = stable_id
                    elif existing:
                        fact["id"] = existing["id"]
                    if existing:
                        fact["created_at"] = existing["created_at"]
                        if existing["content"] == text:
                            fact["updated_at"] = existing["updated_at"]
                    if fact.get("id") not in removed_ids:
                        updated_facts.append(fact)
            
            if updated_facts or removed_ids:
                _store.save_memories(updated_facts, allow_empty=bool(removed_ids))
                _print_memory_log("[memory] langmem updated store to %d facts.", len(updated_facts))
                log.info("[memory] langmem updated store to %d facts.", len(updated_facts))
            else:
                _print_memory_log("[memory] langmem returned empty result — keeping existing facts")
                log.info("[memory] langmem returned empty result — keeping existing facts")
        summarize_episode(messages, model)

    except Exception as exc:
        _print_memory_log("[memory] langmem extraction failed: %s: %s", type(exc).__name__, exc)
        log.exception("[memory] langmem extraction failed")
        # Fallback to summarize episode if manager fails
        summarize_episode(messages, model)

def summarize_episode(messages: List[BaseMessage], model) -> None:
    """Summarizes the trajectory to help avoid future mistakes."""
    _print_memory_log("[memory] Summarizing task episode...")
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
        summary = _content_to_text(response.content)
        _store.add_episode(initial_question, summary, outcome)
        _print_memory_log(
            "[memory] Episode saved: task=%r outcome=%r summary=%r",
            _preview_for_log(initial_question),
            outcome,
            _preview_for_log(summary),
        )
        log.info(
            "[memory] Episode saved: task=%r outcome=%r summary=%r",
            _preview_for_log(initial_question),
            outcome,
            _preview_for_log(summary),
        )
    except Exception as exc:
        _print_memory_log("[memory] Summarization failed: %s: %s", type(exc).__name__, exc)
        log.exception("[memory] Summarization failed")

def _relevance_score(text: str, query_tokens: set) -> float:
    """Simple word-overlap score between a fact and the query (Jaccard-like)."""
    if not query_tokens:
        return 0.0
    fact_tokens = set(text.lower().split())
    return len(fact_tokens & query_tokens) / (len(fact_tokens | query_tokens) or 1)


def retrieve_relevant_context(query: str, char_budget: int = MEMORY_CONTEXT_CHAR_BUDGET) -> str:
    """Fetches facts and recent episodes ranked by relevance, capped by char_budget.
    If truncated, appends a note instructing the agent to call search_memory for more."""
    try:
        query_tokens = set(query.lower().split())
        facts = _store.get_all_memories()
        episodes = _store.get_recent_episodes(limit=5)

        facts.sort(key=lambda m: _relevance_score(m["content"], query_tokens), reverse=True)

        context_parts = []
        budget_remaining = char_budget

        if facts:
            included, total = [], len(facts)
            for m in facts:
                line = f"- {m['content']}"
                if budget_remaining - len(line) - 1 > 0:
                    included.append(line)
                    budget_remaining -= len(line) + 1
                else:
                    break
            fact_block = "<known_facts>\n" + "\n".join(included) + "\n</known_facts>"
            omitted = total - len(included)
            if omitted:
                fact_block += f"\n<!-- {omitted} fact(s) omitted (budget). Call search_memory(query) to retrieve more. -->"
            context_parts.append(fact_block)

        if episodes and budget_remaining > 0:
            included_ep = []
            for e in episodes:
                line = f"Task: {e['task']}\nSummary: {e['summary']}"
                if budget_remaining - len(line) - 2 > 0:
                    included_ep.append(line)
                    budget_remaining -= len(line) + 2
                else:
                    break
            if included_ep:
                ep_block = "<past_experiences>\n" + "\n\n".join(included_ep) + "\n</past_experiences>"
                omitted_ep = len(episodes) - len(included_ep)
                if omitted_ep:
                    ep_block += f"\n<!-- {omitted_ep} episode(s) omitted. Call search_memory(query) for more. -->"
                context_parts.append(ep_block)

        return "\n\n".join(context_parts)
    except Exception as exc:
        _print_memory_log("[memory] Retrieval failed: %s: %s", type(exc).__name__, exc)
        log.exception("[memory] Retrieval failed")
        return ""


def search_memory_store(query: str, max_results: int = 10) -> str:
    """Keyword search across all facts and episodes. Called by the agent when
    the system-prompt injection was truncated or the query needs deeper lookup."""
    try:
        query_tokens = set(query.lower().split())
        facts = _store.get_all_memories()
        episodes = _store.get_recent_episodes(limit=50)

        scored_facts = sorted(
            [(f, _relevance_score(f["content"], query_tokens)) for f in facts],
            key=lambda x: x[1], reverse=True
        )
        scored_eps = sorted(
            [(e, _relevance_score(e["task"] + " " + e["summary"], query_tokens)) for e in episodes],
            key=lambda x: x[1], reverse=True
        )

        results = []
        for m, score in scored_facts[:max_results]:
            if score > 0:
                results.append((score, f"[fact] {m['content']}"))
        for e, score in scored_eps[:max_results]:
            if score > 0:
                results.append((score, f"[episode] Task: {e['task']}\n  Summary: {e['summary']}"))

        results.sort(key=lambda x: x[0], reverse=True)
        parts = [text for _, text in results[:max_results]]
        if not parts:
            return "No matching memories found."
        return "\n\n".join(parts)
    except Exception as exc:
        _print_memory_log("[memory] search_memory_store failed: %s: %s", type(exc).__name__, exc)
        log.exception("[memory] search_memory_store failed")
        return "Memory search failed."
