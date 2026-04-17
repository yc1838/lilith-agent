"""Tests for `_compact_old_tool_messages`.

The compaction function runs on every model_node turn and is the only thing
keeping older tool outputs from overflowing the context window. The original
implementation head-truncated to 300 chars, which can amputate the exact line
containing the answer. This file tests the summarize-preferred variant: when
an LLM-backed summarizer is supplied, old long tool outputs are replaced by
the summarizer's result; the head-truncation path remains as a fallback for
cases where the summarizer is unavailable or fails.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lilith_agent.app import (
    _COMPACT_KEEP_RECENT,
    _COMPACT_MAX_CHARS,
    _COMPACT_SUMMARY_PREFIX,
    _compact_old_tool_messages,
)


def _long_tool_msg(name: str, content: str) -> ToolMessage:
    return ToolMessage(tool_call_id=f"tc-{name}", name=name, content=content)


def test_short_tool_messages_pass_through_unchanged():
    msgs = [
        HumanMessage("q"),
        _long_tool_msg("web_search", "short result"),
        AIMessage("ok"),
    ]
    out = _compact_old_tool_messages(msgs)
    assert out[1].content == "short result"


def test_recent_tool_messages_kept_verbatim_even_when_long():
    long = "X" * (_COMPACT_MAX_CHARS * 10)
    msgs = [_long_tool_msg("web_search", long) for _ in range(_COMPACT_KEEP_RECENT)]
    out = _compact_old_tool_messages(msgs)
    for m in out:
        assert m.content == long


def test_old_long_message_is_summarized_when_summarizer_provided():
    long = "X" * (_COMPACT_MAX_CHARS * 5)
    msgs = [
        _long_tool_msg("web_search", long),  # old
        *[_long_tool_msg("web_search", "short") for _ in range(_COMPACT_KEEP_RECENT)],
    ]

    def fake_summarizer(tool_name: str, content: str) -> str:
        return "SUMMARY_OF_FACTS_42"

    out = _compact_old_tool_messages(msgs, summarize_fn=fake_summarizer)
    assert "SUMMARY_OF_FACTS_42" in out[0].content
    assert out[0].content.startswith(_COMPACT_SUMMARY_PREFIX)
    # The old raw payload is gone — summarization replaced it.
    assert "X" * 1000 not in out[0].content


def test_summarizer_receives_tool_name_and_full_content():
    long = "alpha " * 500
    msgs = [
        _long_tool_msg("arxiv_search", long),
        *[_long_tool_msg("web_search", "short") for _ in range(_COMPACT_KEEP_RECENT)],
    ]
    recorded: dict = {}

    def fake_summarizer(tool_name: str, content: str) -> str:
        recorded["name"] = tool_name
        recorded["len"] = len(content)
        return "ok"

    _compact_old_tool_messages(msgs, summarize_fn=fake_summarizer)
    assert recorded["name"] == "arxiv_search"
    assert recorded["len"] == len(long)


def test_summarizer_failure_falls_back_to_head_truncation():
    long = "Y" * (_COMPACT_MAX_CHARS * 5)
    msgs = [
        _long_tool_msg("web_search", long),
        *[_long_tool_msg("web_search", "short") for _ in range(_COMPACT_KEEP_RECENT)],
    ]

    def broken_summarizer(tool_name: str, content: str) -> str:
        raise RuntimeError("llm offline")

    out = _compact_old_tool_messages(msgs, summarize_fn=broken_summarizer)
    content = out[0].content
    # Fallback marker from the original truncation path
    assert "COMPACTED" in content
    # First `max_chars` preserved verbatim
    assert content.startswith("Y" * _COMPACT_MAX_CHARS)


def test_summarizer_returning_empty_falls_back_to_truncation():
    long = "Z" * (_COMPACT_MAX_CHARS * 5)
    msgs = [
        _long_tool_msg("web_search", long),
        *[_long_tool_msg("web_search", "short") for _ in range(_COMPACT_KEEP_RECENT)],
    ]

    def empty_summarizer(tool_name: str, content: str) -> str:
        return ""

    out = _compact_old_tool_messages(msgs, summarize_fn=empty_summarizer)
    assert "COMPACTED" in out[0].content
    assert out[0].content.startswith("Z" * _COMPACT_MAX_CHARS)


def test_no_summarizer_uses_truncation_fallback():
    """Backwards-compat: the original (no summarize_fn) path must still truncate."""
    long = "W" * (_COMPACT_MAX_CHARS * 5)
    msgs = [
        _long_tool_msg("web_search", long),
        *[_long_tool_msg("web_search", "short") for _ in range(_COMPACT_KEEP_RECENT)],
    ]
    out = _compact_old_tool_messages(msgs)
    assert "COMPACTED" in out[0].content
    assert out[0].content.startswith("W" * _COMPACT_MAX_CHARS)


def test_already_summarized_message_is_not_resummarized():
    """If a prior pass already produced a `[COMPACTED SUMMARY] …` content,
    a second pass must skip it — otherwise we waste a cheap-model call every
    turn on the same already-shrunk payload.
    """
    prior_summary = _COMPACT_SUMMARY_PREFIX + "already shrunk facts"
    msgs = [
        _long_tool_msg("web_search", prior_summary),
        *[_long_tool_msg("web_search", "short") for _ in range(_COMPACT_KEEP_RECENT)],
    ]
    calls = {"n": 0}

    def fake_summarizer(tool_name: str, content: str) -> str:
        calls["n"] += 1
        return "should not run"

    out = _compact_old_tool_messages(msgs, summarize_fn=fake_summarizer)
    assert calls["n"] == 0
    assert out[0].content == prior_summary
