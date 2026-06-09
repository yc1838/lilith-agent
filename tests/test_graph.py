
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
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg: fake_model)
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
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg: FakeModel())
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
            if not self.system_prompt:
                self.system_prompt = str(messages[0].content)
            return AIMessage(content="Final Answer: inspected")

    fake_model = FakeModel()
    cfg = Config.from_env()
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg: fake_model)
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
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg: fake_model)
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
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg: fake_model)
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
            self.supervisor_calls = 0

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            self.supervisor_calls += 1
            return AIMessage(content='{"status":"nudge","best_answer":"backtick","guidance":"You have verified the answer. Stop and answer backtick."}')

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 10
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: object())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Unlambda question")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "supervisor-nudge-test"}},
    )

    assert result["messages"][-1].content == "Final Answer: backtick"
    assert strong.supervisor_calls == 1
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
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
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
            prompt = str(messages[0].content)
            if "SUPERVISOR FINALIZER" in prompt:
                self.finalizer_prompt = prompt
                return AIMessage(content="Final Answer: final")
            return AIMessage(content='{"status":"finalize","best_answer":"","guidance":"Existing evidence is enough."}')

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 10
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: object())
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


def test_supervisor_finalizer_rejects_unknown_best_answer_and_forces_best_guess(monkeypatch, tmp_path):
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
            self.finalizer_calls = 0

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            prompt = str(messages[0].content)
            if "SUPERVISOR FINALIZER" in prompt:
                self.finalizer_calls += 1
                return AIMessage(content="Final Answer: best guess")
            return AIMessage(content='{"status":"finalize","best_answer":"unknown","guidance":"Looping; make a best guess."}')

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 10
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: object())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Unlambda question")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "supervisor-unknown-best-answer-test"}},
    )

    assert result["messages"][-1].content == "Final Answer: best guess"
    assert strong.finalizer_calls == 1


def test_supervisor_finalizes_even_with_placeholder_if_requested(monkeypatch, tmp_path):
    class FakeBoundModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return _ai_with_calls([
                {
                    "id": f"call-{self.calls}",
                    "name": "echo_tool",
                    "args": {"text": "partial evidence"},
                }
            ])

    class FakeStrongModel:
        def __init__(self):
            self.bound = FakeBoundModel()
            self.supervisor_calls = 0
            self.finalizer_calls = 0

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            prompt = str(messages[0].content)
            if "SUPERVISOR FINALIZER" in prompt:
                self.finalizer_calls += 1
                return AIMessage(content="Final Answer: finalizer output")
            self.supervisor_calls += 1
            if self.supervisor_calls == 1:
                return AIMessage(content='{"status":"nudge","best_answer":"","guidance":"Use the evidence to make a best guess."}')
            # On second call, it asks to finalize but with a placeholder answer. The code should allow finalization.
            return AIMessage(content='{"status":"finalize","best_answer":"Unknown","guidance":"You were already nudged. Provide your final answer based on the best available information."}')

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 12
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: object())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Question requiring a concrete answer")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "supervisor-finalize-after-nudge-test"}},
    )

    assert result["messages"][-1].content == "Final Answer: finalizer output"
    assert strong.finalizer_calls == 1
    assert strong.supervisor_calls == 2


def test_supervisor_forces_finalize_after_max_nudges(monkeypatch, tmp_path):
    class FakeBoundModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return _ai_with_calls([
                {
                    "id": f"call-{self.calls}",
                    "name": "echo_tool",
                    "args": {"text": "partial evidence"},
                }
            ])

    class FakeStrongModel:
        def __init__(self):
            self.bound = FakeBoundModel()
            self.supervisor_calls = 0
            self.finalizer_calls = 0

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            prompt = str(messages[0].content)
            if "SUPERVISOR FINALIZER" in prompt:
                self.finalizer_calls += 1
                return AIMessage(content="Final Answer: max nudges forced this")
            self.supervisor_calls += 1
            return AIMessage(content='{"status":"nudge","best_answer":"concrete candidate","guidance":"Check one more constraint before final answer."}')

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 20
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MAX_NUDGES", 5, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: object())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Question requiring more checking")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "supervisor-max-nudges-test"}},
    )

    assert strong.finalizer_calls == 0
    assert result["messages"][-1].content == "Final Answer: concrete candidate"
    assert strong.supervisor_calls == 5
    assert result["supervisor_decision"] == "finalize"


def test_final_answer_gets_supervisor_review_and_can_be_returned_for_revision(monkeypatch, tmp_path):
    class FakeBoundModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if any("FINAL ANSWER REVIEW FAILED" in str(getattr(m, "content", "")) for m in messages):
                return AIMessage(content="Final Answer: corrected answer")
            return AIMessage(content="Final Answer: wrong answer")

    class FakeStrongModel:
        def __init__(self):
            self.bound = FakeBoundModel()
            self.review_calls = 0

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            prompt = str(messages[0].content)
            if "FINAL ANSWER REVIEW" in prompt:
                self.review_calls += 1
                if self.review_calls == 1:
                    return AIMessage(content='{"status":"nudge","best_answer":"","guidance":"The proposed answer violates a constraint; fix it."}')
                return AIMessage(content='{"status":"finalize","best_answer":"corrected answer","guidance":"Approved."}')
            return AIMessage(content='{"status":"continue","best_answer":"","guidance":""}')

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 10
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: object())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Question with constraint")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "final-answer-review-revision-test"}},
    )

    assert strong.review_calls == 2
    assert "Final Answer: corrected answer" in result["messages"][-1].content
    assert all("wrong answer" != getattr(m, "content", "") for m in result["messages"][-1:])


def test_fail_safe_falls_back_to_supervisor_best_answer_when_empty(monkeypatch, tmp_path):
    """fail_safe model returning empty content must not propagate; supervisor_best_answer wins."""

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
            self.fail_safe_calls = 0
            self.supervisor_calls = 0

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            prompt = str(messages[0].content)
            if "EMERGENCY OVERRIDE" in prompt:
                self.fail_safe_calls += 1
                return AIMessage(content="")
            self.supervisor_calls += 1
            return AIMessage(content='{"status":"nudge","best_answer":"backtick","guidance":"Stop and answer backtick."}')

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 4
    cfg.budget_hard_cap = 2
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: object())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Question")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "fail-safe-best-answer-test"}},
    )

    last = result["messages"][-1]
    assert last.content
    assert "Final Answer:" in last.content
    assert "backtick" in last.content


def test_fail_safe_never_returns_empty_answer_without_best_answer(monkeypatch, tmp_path):
    """No supervisor_best_answer, fail_safe model empty: still produce non-empty descriptive Final Answer."""

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
            return AIMessage(content="")

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 4
    cfg.budget_hard_cap = 2
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 99, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: object())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Question")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "fail-safe-default-answer-test"}},
    )

    last = result["messages"][-1]
    content = str(getattr(last, "content", ""))
    assert content.strip(), "fail_safe must never propagate an empty answer"
    assert "Final Answer:" in content


def test_supervisor_review_auto_approves_after_fail_safe(monkeypatch, tmp_path):
    """fail_safe -> supervisor_review must auto-approve (no infinite loop back to model)."""

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
            self.review_calls = 0

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            prompt = str(messages[0].content)
            if "EMERGENCY OVERRIDE" in prompt:
                return AIMessage(content="Final Answer: best effort")
            if "FINAL ANSWER REVIEW" in prompt:
                self.review_calls += 1
                return AIMessage(content='{"status":"nudge","best_answer":"","guidance":"reject"}')
            return AIMessage(content='{"status":"continue","best_answer":"","guidance":""}')

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 4
    cfg.budget_hard_cap = 2
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 99, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: object())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Question")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "fail-safe-review-no-loop-test"}},
    )

    last = result["messages"][-1]
    assert "Final Answer: best effort" in str(getattr(last, "content", ""))
    assert strong.review_calls == 0


def test_tool_node_invokes_tool_successfully():
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


def test_supervisor_finalizes_when_agent_ignores_prior_nudge(monkeypatch, tmp_path):
    class FakeBoundModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            sup_count = sum(
                1 for m in messages
                if "SUPERVISOR:" in str(getattr(m, "content", ""))
            )
            if sup_count >= 2:
                return AIMessage(content="Final Answer: backtick")
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
            self.supervisor_calls = 0

        def bind_tools(self, tools):
            return self.bound

        def invoke(self, messages):
            self.supervisor_calls += 1
            return AIMessage(content='{"status":"nudge","best_answer":"backtick","guidance":"Stop. Existing evidence supports backtick."}')

    strong = FakeStrongModel()
    cfg = Config.from_env()
    cfg.recursion_limit = 12
    cfg.budget_hard_cap = 99
    cfg.budget_warn_at = 99
    cfg.compact_summarize = False
    monkeypatch.setenv("LILITH_HOME", str(tmp_path / ".lilith"))
    monkeypatch.setattr("lilith_agent.app._SUPERVISOR_MIN_TOOL_CALLS", 1, raising=False)
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
    monkeypatch.setattr("lilith_agent.app.get_cheap_model", lambda cfg: object())
    monkeypatch.setattr("lilith_agent.tools.build_tools", lambda cfg: [echo_tool])
    monkeypatch.setattr("lilith_agent.memory.extract_and_compress_facts", lambda messages, model: None)

    graph = build_react_agent(cfg)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Unlambda question")], "iterations": 0, "todos": []},
        {"configurable": {"thread_id": "supervisor-finalize-test"}},
    )

    assert result["messages"][-1].content == "Final Answer: backtick"
    assert strong.supervisor_calls == 2
    assert strong.bound.calls == 3


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
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg, thinking=True: strong)
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
    monkeypatch.setattr("lilith_agent.app.get_strong_model", lambda cfg: FakeStrongModel())
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


def test_tool_node_prints_semantic_dedup_queries(capsys):
    node = _build_tool_node([echo_tool], semantic_dedup_threshold=0.5)
    prior_call = {"id": "old", "name": "web_search", "args": {"query": "apollo moon landing transcript"}}
    new_call = {"id": "new", "name": "web_search", "args": {"query": "moon landing apollo transcript"}}
    state = {"messages": [
        HumanMessage(content="go"),
        _ai_with_calls([prior_call]),
        ToolMessage(tool_call_id="old", name="web_search", content="done"),
        _ai_with_calls([new_call]),
    ]}

    out = node(state)

    printed = capsys.readouterr().out
    assert "[tools] semantic_dedup score=1.00 tool=web_search" in printed
    assert "query='moon landing apollo transcript'" in printed
    assert "prior_query='apollo moon landing transcript'" in printed
    assert "REDUNDANT SEARCH PATH" in out["messages"][0].content


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
