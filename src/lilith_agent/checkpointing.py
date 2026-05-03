from __future__ import annotations

import importlib
import logging
import sqlite3
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def build_checkpointer(lilith_home: Path) -> Any:
    """Create the graph checkpointer, preferring SQLite persistence when available."""
    lilith_home.mkdir(parents=True, exist_ok=True)
    db_path = lilith_home / "threads.sqlite"

    try:
        sqlite_module = importlib.import_module("langgraph.checkpoint.sqlite")
        sqlite_saver_cls = sqlite_module.SqliteSaver
    except ModuleNotFoundError as exc:
        memory_module = importlib.import_module("langgraph.checkpoint.memory")
        in_memory_saver_cls = memory_module.InMemorySaver
        log.warning(
            "langgraph-checkpoint-sqlite missing; using InMemorySaver instead. "
            "Install langgraph-checkpoint-sqlite to restore persistent thread checkpoints. "
            "Original import error: %s",
            exc,
        )
        return in_memory_saver_cls()

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return sqlite_saver_cls(conn)
