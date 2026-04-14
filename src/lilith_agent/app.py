from __future__ import annotations

import json
import logging
from typing import Callable

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState

from lilith_agent.config import Config
from lilith_agent.models import get_extra_strong_model

log = logging.getLogger(__name__)

RECURSION_LIMIT = 50


def _call_key(name: str, args) -> tuple[str, str]:
    try:
        norm = json.dumps(args or {}, sort_keys=True, default=str)
    except Exception:
        norm = repr(args)
    return (name, norm)


def _collect_seen_calls(messages) -> set[tuple[str, str]]:
    """All (tool_name, args) pairs already requested in prior AI messages."""
    seen: set[tuple[str, str]] = set()
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                seen.add(_call_key(tc.get("name"), tc.get("args")))
    return seen


def _route_after_model(state) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


def _build_tool_node(tools: list[BaseTool]) -> Callable:
    """Tool executor with dedup + exception-to-ToolMessage feedback.

    Dedup rule: if the same (tool_name, args) pair appeared in any prior
    AIMessage in history, short-circuit with a synthetic ToolMessage telling
    the model it already tried this, without invoking the tool.
    """
    tools_by_name = {t.name: t for t in tools}

    def tool_node(state):
        messages = state["messages"]
        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        # "seen" = calls that appeared in AIMessages strictly BEFORE the current one.
        seen = _collect_seen_calls(messages[:-1])

        results: list[ToolMessage] = []
        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args") or {}
            tc_id = tc.get("id", "")

            key = _call_key(name, args)
            if key in seen:
                log.warning("[dedup] short-circuited repeat tool call: %s %s", name, args)
                results.append(ToolMessage(
                    tool_call_id=tc_id,
                    name=name or "unknown",
                    content=(
                        f"You already called `{name}` with these exact arguments earlier "
                        "in this conversation and received a result. Do not repeat the same "
                        "call — try different arguments, a different tool, or use the prior "
                        "result to answer the user."
                    ),
                    status="error",
                ))
                continue

            tool = tools_by_name.get(name)
            if tool is None:
                results.append(ToolMessage(
                    tool_call_id=tc_id,
                    name=name or "unknown",
                    content=f"ERROR: unknown tool {name!r}. Available: {sorted(tools_by_name)}",
                    status="error",
                ))
                continue

            try:
                out = tool.invoke(args)
            except Exception as e:
                log.warning("[tool_node] %s raised: %s", name, e)
                out = f"ERROR: {type(e).__name__}: {e}"
                results.append(ToolMessage(tool_call_id=tc_id, name=name, content=str(out), status="error"))
                continue

            results.append(ToolMessage(tool_call_id=tc_id, name=name, content=str(out)))

        return {"messages": results}

    return tool_node


def build_react_agent(cfg: Config):
    """Explicit ReAct graph with tool-call dedup, error feedback, and recursion cap."""
    try:
        from lilith_agent.tools import build_tools
        tools = build_tools(cfg)
    except ImportError:
        log.warning("Tools not found; running with zero tools.")
        tools = []

    model = get_extra_strong_model(cfg).bind_tools(tools)

    def model_node(state):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = _build_tool_node(tools)

    graph = StateGraph(MessagesState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", _route_after_model, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    compiled = graph.compile(checkpointer=MemorySaver())
    return compiled.with_config({"recursion_limit": RECURSION_LIMIT})
