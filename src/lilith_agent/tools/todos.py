from __future__ import annotations
from typing import List

def write_todos(todos: List[str]) -> str:
    """
    Initialize or overwrite the current task list (Todo list).
    Used for high-level planning and tracking progress.
    """
    # This tool will be handled specially by the executor node to update AgentState
    return f"SET_TODOS: {todos}"

def mark_todo_done(index: int) -> str:
    """
    Mark a specific todo as complete by its 0-indexed position.
    """
    # This tool will be handled specially by the executor node to update AgentState
    return f"DONE_TODO: {index}"

def read_todos(todo_list: List[str]) -> str:
    """
    Read the current state of the todo list.
    """
    if not todo_list:
        return "Todo list is empty."
    
    lines = []
    for i, todo in enumerate(todo_list):
        lines.append(f"{i}. {todo}")
    return "\n".join(lines)
