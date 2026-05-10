from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as tool_decorator

from lilith_agent.app import _build_tool_node, _cooldown_limit_for, _route_after_model, build_react_agent
from lilith_agent.config import Config


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
    assert _route_after_model(state) == "extract_memory"


def test_graph_returns_fail_safe_answer_when_hard_cap_hits_near_recursion_limit(monkeypatch, tmp_path, capsys):
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            if "SYSTEM EMERGENCY OVERRIDE" in str(messages[0].content):
                return AIMessage(content="Final Answer: best effort answer")
            self.calls += 1
            return _ai_with_calls([
                {
                    "id": f"call-{self.calls}",
                    "name": "echo_tool",
                    "args": {"text": str(self.calls)},
                }
            ])

    fake_model = FakeModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 4
    cfg.budget_hard_cap = 2
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app.get_extra_strong_model", lambda cfg: fake_model)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: fake_model)
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="answer this")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "hard-cap-test"}},
    )

    captured = capsys.readouterr().out
    assert "[route] recursion threshold reached" in captured
    assert "[fail_safe] emergency override" in captured
    assert result["messages"][-1].content == "Final Answer: best effort answer"


def test_build_react_agent_prints_effective_recursion_limit(monkeypatch, tmp_path, capsys):
    class FakeModel:
        def bind_tools(self, tools):
            return self

    cfg = Config.from_env()
    cfg.recursion_limit = 50
    cfg.budget_hard_cap = 25
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app.get_extra_strong_model", lambda cfg: FakeModel())
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: FakeModel())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])

    build_react_agent(cfg)

    captured = capsys.readouterr().out
    assert "[graph] effective_recursion_limit=79 logical_recursion_limit=50 budget_hard_cap=25 headroom=4" in captured


def test_model_prompt_includes_youtube_fallback_strategy(monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self):
            self.system_prompt = ""

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.system_prompt = str(messages[0].content)
            return AIMessage(content="Final Answer: inspected")

    fake_model = FakeModel()
    cfg = Config.from_env()
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app.get_extra_strong_model", lambda cfg: fake_model)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: fake_model)
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    graph.invoke(
        {"messages": [HumanMessage(content="What happens in https://www.youtube.com/watch?v=abcdefghijk?")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "youtube-fallback-prompt-test"}},
    )

    prompt = fake_model.system_prompt.lower()
    assert "youtube fallback strategy" in prompt
    assert "video id" in prompt
    assert "transcript" in prompt
    assert "do not repeatedly retry" in prompt


def test_fail_safe_uses_unbound_model_to_prevent_more_tool_calls(monkeypatch, tmp_path):
    class FakeBoundModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return _ai_with_calls([
                {
                    "id": f"bound-call-{self.calls}",
                    "name": "echo_tool",
                    "args": {"text": str(self.calls)},
                }
            ])

    class FakeModel:
        def __init__(self):
            self.bound = FakeBoundModel()

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            return AIMessage(content="Final Answer: unbound best effort")

    fake_model = FakeModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 4
    cfg.budget_hard_cap = 1
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app.get_extra_strong_model", lambda cfg: fake_model)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: fake_model)
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="answer this")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "unbound-fail-safe-test"}},
    )

    assert result["messages"][-1].content == "Final Answer: unbound best effort"
    assert not getattr(result["messages"][-1], "tool_calls", None)


def test_fail_safe_prompt_reinforces_original_question_contract(monkeypatch, tmp_path):
    class FakeBoundModel:
        def invoke(self, messages):
            return _ai_with_calls([
                {
                    "id": "bound-call",
                    "name": "echo_tool",
                    "args": {"text": "intermediate"},
                }
            ])

    class FakeModel:
        def __init__(self):
            self.bound = FakeBoundModel()
            self.fail_safe_prompt = ""

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            self.fail_safe_prompt = str(messages[0].content)
            return AIMessage(content="Final Answer: best effort")

    fake_model = FakeModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 4
    cfg.budget_hard_cap = 1
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app.get_extra_strong_model", lambda cfg: fake_model)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: fake_model)
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    graph.invoke(
        {"messages": [HumanMessage(content="What country corresponds to this capital?")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "fail-safe-contract-prompt-test"}},
    )

    prompt = fake_model.fail_safe_prompt.lower()
    assert "original question" in prompt
    assert "not an intermediate" in prompt
    assert "bare final answer" in prompt


def test_supervisor_nudges_agent_to_answer_when_evidence_is_enough(monkeypatch, tmp_path):
    class FakeBoundModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if any("SUPERVISOR" in str(getattr(m, "content", "")) for m in messages):
                return AIMessage(content="Final Answer: backtick")
            return _ai_with_calls([
                {
                    "id": f"call-{self.calls}",
                    "name": "echo_tool",
                    "args": {"text": "Add `: 'For penguins\\n'"},
                }
            ])

    class FakeStrongModel:
        def __init__(self):
            self.bound = FakeBoundModel()

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            return AIMessage(content="Final Answer: fallback")

    class FakeSupervisorModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content='{"status":"nudge","best_answer":"backtick","guidance":"You have verified the answer. Stop and answer backtick."}')

    strong = FakeStrongModel()
    supervisor = FakeSupervisorModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 10
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_extra_strong_model", lambda cfg: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: supervisor)
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Unlambda question")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "supervisor-nudge-test"}},
    )

    assert result["messages"][-1].content == "Final Answer: backtick"
    assert supervisor.calls == 1
    assert strong.bound.calls == 2


def test_supervisor_uses_extra_strong_model_not_cheap_model(monkeypatch, tmp_path):
    class FakeBoundModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if any("SUPERVISOR" in str(getattr(m, "content", "")) for m in messages):
                return AIMessage(content="Final Answer: backtick")
            return _ai_with_calls([
                {
                    "id": f"call-{self.calls}",
                    "name": "echo_tool",
                    "args": {"text": "evidence"},
                }
            ])

    class FakeExtraStrongModel:
        def __init__(self):
            self.bound = FakeBoundModel()
            self.supervisor_calls = 0

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            self.supervisor_calls += 1
            return AIMessage(content='{"status":"nudge","best_answer":"backtick","guidance":"Stop and answer backtick."}')

    strong = FakeExtraStrongModel()

    def cheap_should_not_be_used(cfg):
        raise AssertionError("supervisor should use extra strong model, not cheap model")

    cfg = Config.from_env()
    cfg.recursion_limit = 10
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_extra_strong_model", lambda cfg: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", cheap_should_not_be_used)
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Unlambda question")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "supervisor-extra-strong-test"}},
    )

    assert result["messages"][-1].content == "Final Answer: backtick"
    assert strong.supervisor_calls == 1


def test_supervisor_finalizer_prompt_reinforces_original_question_contract(monkeypatch, tmp_path):
    class FakeBoundModel:
        def invoke(self, messages):
            return _ai_with_calls([
                {
                    "id": "call",
                    "name": "echo_tool",
                    "args": {"text": "evidence"},
                }
            ])

    class FakeStrongModel:
        def __init__(self):
            self.bound = FakeBoundModel()
            self.finalizer_prompt = ""

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            self.finalizer_prompt = str(messages[0].content)
            return AIMessage(content="Final Answer: final")

    class FakeSupervisorModel:
        def invoke(self, messages):
            return AIMessage(content='{"status":"finalize","best_answer":"","guidance":"Existing evidence is enough."}')

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 10
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_extra_strong_model", lambda cfg: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: FakeSupervisorModel())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    graph.invoke(
        {"messages": [HumanMessage(content="What final entity answers the original question?")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "supervisor-finalizer-contract-prompt-test"}},
    )

    prompt = strong.finalizer_prompt.lower()
    assert "original question" in prompt
    assert "not an intermediate" in prompt
    assert "bare final answer" in prompt


def test_supervisor_finalizes_when_agent_ignores_prior_nudge(monkeypatch, tmp_path):
    class FakeBoundModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return _ai_with_calls([
                {
                    "id": f"call-{self.calls}",
                    "name": "echo_tool",
                    "args": {"text": str(self.calls)},
                }
            ])

    class FakeStrongModel:
        def __init__(self):
            self.bound = FakeBoundModel()

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            return AIMessage(content="Final Answer: fallback")

    class FakeSupervisorModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content='{"status":"nudge","best_answer":"backtick","guidance":"Stop. Existing evidence supports backtick."}')

    strong = FakeStrongModel()
    supervisor = FakeSupervisorModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 12
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_extra_strong_model", lambda cfg: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: supervisor)
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Unlambda question")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "supervisor-finalize-test"}},
    )

    assert result["messages"][-1].content == "Final Answer: backtick"
    assert supervisor.calls == 2
    assert strong.bound.calls == 2


def test_supervisor_overhead_leaves_room_for_hard_cap_fail_safe(monkeypatch, tmp_path):
    class FakeBoundModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return _ai_with_calls([
                {
                    "id": f"call-{self.calls}",
                    "name": "echo_tool",
                    "args": {"text": str(self.calls)},
                }
            ])

    class FakeStrongModel:
        def __init__(self):
            self.bound = FakeBoundModel()

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            return AIMessage(content="Final Answer: hard cap fallback")

    class FakeSupervisorModel:
        def invoke(self, messages):
            return AIMessage(content='{"status":"continue"}')

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 8
    cfg.budget_hard_cap = 5
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_extra_strong_model", lambda cfg: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: FakeSupervisorModel())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="force hard cap")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "supervisor-hard-cap-headroom-test"}},
    )

    assert result["messages"][-1].content == "Final Answer: hard cap fallback"


def test_supervisor_overhead_leaves_room_for_iteration_fail_safe(monkeypatch, tmp_path):
    class FakeBoundModel:
        def invoke(self, messages):
            return _ai_with_calls([
                {
                    "id": "call",
                    "name": "echo_tool",
                    "args": {"text": "loop"},
                }
            ])

    class FakeStrongModel:
        def bind_tools(self, tools):
            return FakeBoundModel()

        def invoke(self, messages):
            return AIMessage(content="Final Answer: iteration fallback")

    class FakeSupervisorModel:
        def invoke(self, messages):
            return AIMessage(content='{"status":"continue"}')

    cfg = Config.from_env()
    cfg.recursion_limit = 5
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_extra_strong_model", lambda cfg: FakeStrongModel())
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: FakeSupervisorModel())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="force iteration cap")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "supervisor-iteration-headroom-test"}},
    )

    assert result["messages"][-1].content == "Final Answer: iteration fallback"


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


def test_dedup_does_not_emit_warning_level_logs(caplog):
    """Routine dedup fires on many turns; WARNING floods stderr during normal runs.

    Regression guard: `[dedup]`, `[semantic_dedup]`, `[loop_breaker]` stay at INFO."""
    import logging

    node = _build_tool_node([echo_tool])
    prior_call = {"id": "old", "name": "echo_tool", "args": {"text": "x"}}
    state = {"messages": [
        HumanMessage(content="go"),
        _ai_with_calls([prior_call]),
        ToolMessage(tool_call_id="old", name="echo_tool", content="done"),
        _ai_with_calls([{"id": "new", "name": "echo_tool", "args": {"text": "x"}}]),
    ]}

    with caplog.at_level(logging.DEBUG, logger="lilith_agent.app"):
        node(state)

    for rec in caplog.records:
        if "[dedup]" in rec.getMessage() or "[semantic_dedup]" in rec.getMessage() or "[loop_breaker]" in rec.getMessage():
            assert rec.levelno < logging.WARNING, f"routine guard log at {rec.levelname}: {rec.message}"


def test_cooldown_limit_for_known_tool_is_positive_int():
    """Each tool must declare a positive cooldown limit. Regression guard
    against the `3 if name == 'web_search' else 3` no-op ternary."""
    limit = _cooldown_limit_for("web_search")
    assert isinstance(limit, int) and limit > 0
    assert _cooldown_limit_for("fetch_url") == _cooldown_limit_for("web_search")


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


@tool_decorator
def todo_sentinel_tool(action: str) -> str:
    """Returns todo sentinel output."""
    if action == "write":
        return "SET_TODOS: ['first', 'second']"
    return "DONE_TODO: 0"


def test_tool_node_consumes_todo_sentinels_into_state():
    node = _build_tool_node([todo_sentinel_tool])

    write_out = node({
        "messages": [_ai_with_calls([{"id": "1", "name": "todo_sentinel_tool", "args": {"action": "write"}}])],
        "todos": [],
    })
    assert write_out["todos"] == ["first", "second"]

    done_out = node({
        "messages": [_ai_with_calls([{"id": "2", "name": "todo_sentinel_tool", "args": {"action": "done"}}])],
        "todos": write_out["todos"],
    })
    assert done_out["todos"] == ["second"]
