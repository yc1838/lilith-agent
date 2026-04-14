from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as tool_decorator

from lilith_agent.app import _build_tool_node, _route_after_model


@tool_decorator
def echo_tool(text: str) -> str:
    """Echoes back the text."""
    return f"echoed: {text}"


def _ai_with_calls(calls):
    return AIMessage(content="", tool_calls=calls)


def test_router_goes_to_tools_when_tool_calls_present():
    state = {"messages": [_ai_with_calls([{"id": "1", "name": "echo_tool", "args": {"text": "hi"}}])]}
    assert _route_after_model(state) == "tools"


def test_router_ends_when_no_tool_calls():
    state = {"messages": [AIMessage(content="done")]}
    assert _route_after_model(state) == "__end__"


def test_tool_node_invokes_tool_and_returns_tool_message():
    node = _build_tool_node([echo_tool])
    state = {"messages": [
        HumanMessage(content="say hi"),
        _ai_with_calls([{"id": "1", "name": "echo_tool", "args": {"text": "hi"}}]),
    ]}

    out = node(state)
    assert len(out["messages"]) == 1
    msg = out["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "1"
    assert "echoed: hi" in msg.content


def test_tool_node_dedups_repeat_tool_call_without_invoking():
    calls = 0

    @tool_decorator
    def counting_tool(x: str) -> str:
        """Counting tool."""
        nonlocal calls
        calls += 1
        return f"ran {calls}"

    node = _build_tool_node([counting_tool])

    # History: earlier AI message already called counting_tool(x="a")
    prior_call = {"id": "old", "name": "counting_tool", "args": {"x": "a"}}
    prior_ai = _ai_with_calls([prior_call])
    prior_result = ToolMessage(tool_call_id="old", name="counting_tool", content="ran 0")

    # Now a new AI message asks for the same tool with the same args.
    new_call = {"id": "new", "name": "counting_tool", "args": {"x": "a"}}
    state = {"messages": [
        HumanMessage(content="go"),
        prior_ai,
        prior_result,
        _ai_with_calls([new_call]),
    ]}

    out = node(state)

    assert calls == 0, "deduped call must not invoke the tool again"
    msg = out["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "new"
    assert "already called" in msg.content.lower()


def test_tool_node_handles_unknown_tool_name():
    node = _build_tool_node([echo_tool])
    state = {"messages": [_ai_with_calls([{"id": "1", "name": "ghost", "args": {}}])]}

    out = node(state)
    msg = out["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert "unknown tool" in msg.content.lower()


def test_tool_node_catches_tool_exceptions_and_feeds_back():
    @tool_decorator
    def boom_tool(x: str) -> str:
        """Always raises."""
        raise RuntimeError("kaboom")

    node = _build_tool_node([boom_tool])
    state = {"messages": [_ai_with_calls([{"id": "1", "name": "boom_tool", "args": {"x": "y"}}])]}

    out = node(state)
    msg = out["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert "kaboom" in msg.content
