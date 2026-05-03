from __future__ import annotations
from typing import List, Optional


def todos(
    action: str,
    items: Optional[List[str]] = None,
    index: Optional[int] = None,
) -> str:
    """Branching todo tool. Executor node consumes SET_TODOS / DONE_TODO sentinels."""
    if action == "write":
        if items is None:
            return "ERROR: action='write' requires `items` (list of strings)."
        return f"SET_TODOS: {items}"
    if action == "done":
        if index is None:
            return "ERROR: action='done' requires `index` (0-based position)."
        return f"DONE_TODO: {index}"
    return f"ERROR: unknown action '{action}'. Use 'write' or 'done'."
