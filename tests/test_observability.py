import json
from unittest.mock import patch, MagicMock

from lilith_agent.tools.ddg_search import web_search


def test_web_search_returns_error_when_ddgs_hangs(monkeypatch):
    """If DDGS blocks forever, web_search must still return within seconds."""
    import time as _time
    from lilith_agent.tools import ddg_search

    monkeypatch.setattr(ddg_search, "_HARD_TIMEOUT_S", 1)

    def _hang(*_a, **_kw):
        _time.sleep(30)
        return []

    fake_ddgs = MagicMock()
    fake_ddgs.__enter__.return_value.text.side_effect = _hang
    fake_ddgs.__exit__.return_value = False

    with patch("ddgs.DDGS") as ddgs_cls:
        ddgs_cls.return_value = fake_ddgs
        t0 = _time.monotonic()
        result = web_search("moon", max_results=3)
        elapsed = _time.monotonic() - t0

    assert elapsed < 3, f"web_search hung for {elapsed:.1f}s, hard timeout not enforced"
    assert "timed out" in result.lower() or "error" in result.lower()


def test_web_search_passes_timeout():
    fake_ddgs = MagicMock()
    fake_ddgs.__enter__.return_value.text.return_value = iter([])
    fake_ddgs.__exit__.return_value = False

    with patch("ddgs.DDGS") as ddgs_cls:
        ddgs_cls.return_value = fake_ddgs
        web_search("hello", max_results=3)

    assert ddgs_cls.called, "DDGS must be constructed"
    _, kwargs = ddgs_cls.call_args
    assert "timeout" in kwargs and kwargs["timeout"] > 0, (
        "web_search must pass a timeout to DDGS so a stalled network call fails fast"
    )


def test_setup_arize_skips_without_credentials(monkeypatch):
    from lilith_agent.observability import setup_arize

    monkeypatch.delenv("ARIZE_SPACE_ID", raising=False)
    monkeypatch.delenv("ARIZE_API_KEY", raising=False)

    assert setup_arize() is False


def test_jsonl_callback_captures_chat_model_payload(tmp_path):
    from lilith_agent.observability import JsonlTraceCallback
    from langchain_core.messages import HumanMessage, SystemMessage

    cb = JsonlTraceCallback(tmp_path / "trace.jsonl")
    messages = [[SystemMessage(content="sys"), HumanMessage(content="hi")]]
    cb.on_chat_model_start({"name": "ChatAnthropic"}, messages, run_id="r0")

    events = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "chat_model_start"
    assert events[0]["model"] == "ChatAnthropic"
    assert any(m["type"] == "system" and m["content"] == "sys" for m in events[0]["messages"][0])
    assert any(m["type"] == "human" and m["content"] == "hi" for m in events[0]["messages"][0])


def test_jsonl_callback_captures_chain_boundaries(tmp_path):
    from lilith_agent.observability import JsonlTraceCallback

    cb = JsonlTraceCallback(tmp_path / "trace.jsonl")
    cb.on_chain_start({"name": "agent"}, {"messages": []}, run_id="c1")
    cb.on_chain_end({"messages": ["ok"]}, run_id="c1")

    events = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "chain_start"
    assert events[0]["name"] == "agent"
    assert events[1]["event"] == "chain_end"
    assert events[1]["elapsed_s"] >= 0


def test_jsonl_callback_mirrors_non_error_events_at_debug(tmp_path, caplog):
    """Non-error LangGraph events are logged at DEBUG so they don't flood the console.

    Keeping INFO clean was the whole point of the level demotion — the full payload
    still lives in the .jsonl file, so the mirror is just a low-priority heartbeat.
    """
    from lilith_agent.observability import JsonlTraceCallback
    import logging as _logging

    cb = JsonlTraceCallback(tmp_path / "trace.jsonl")
    with caplog.at_level(_logging.DEBUG, logger="lilith_agent.trace"):
        cb.on_tool_start({"name": "tavily_search"}, "moon", run_id="r1")

    matches = [
        r for r in caplog.records
        if "tool_start" in r.getMessage() and "tavily_search" in r.getMessage()
    ]
    assert matches, "expected tool_start mirror record"
    assert all(r.levelno == _logging.DEBUG for r in matches)


def test_jsonl_callback_mirrors_errors_at_warning(tmp_path, caplog):
    """Errors stay visible: mirrored at WARNING so they punch through the default cutoff."""
    from lilith_agent.observability import JsonlTraceCallback
    import logging as _logging

    cb = JsonlTraceCallback(tmp_path / "trace.jsonl")
    with caplog.at_level(_logging.WARNING, logger="lilith_agent.trace"):
        cb.on_tool_error(RuntimeError("boom"), run_id="r1")

    matches = [r for r in caplog.records if "tool_error" in r.getMessage()]
    assert matches, "expected tool_error mirror record"
    assert all(r.levelno == _logging.WARNING for r in matches)


def test_jsonl_callback_writes_tool_and_llm_events(tmp_path):
    from lilith_agent.observability import JsonlTraceCallback

    path = tmp_path / "trace.jsonl"
    cb = JsonlTraceCallback(path)

    cb.on_tool_start({"name": "web_search"}, "moon perigee", run_id="r1")
    cb.on_tool_end("results...", run_id="r1")
    cb.on_tool_error(RuntimeError("boom"), run_id="r2")

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    events = [json.loads(l) for l in lines]
    assert events[0]["event"] == "tool_start"
    assert events[0]["name"] == "web_search"
    assert events[0]["input"] == "moon perigee"
    assert events[1]["event"] == "tool_end"
    assert events[2]["event"] == "tool_error"
    assert "boom" in events[2]["error"]
