from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class Case:
    id: str
    raw: str
    expected_deterministic: str
    needs_llm: bool


# Regression table: 17 real-shape GAIA answer fixtures.
# CI runs assertions against the deterministic layer + gating layer only.
# No network / LLM calls — the LLM path is exercised with a mock further down.
CASES: list[Case] = [
    # ---- Already clean, short — must pass through untouched and skip LLM ----
    Case("bare_number",      "42",                  "42",                  False),
    Case("bare_word",        "Paris",               "Paris",               False),
    Case("year",             "1969",                "1969",                False),
    Case("short_phrase",     "Mount Everest",       "Mount Everest",       False),
    Case("abbrev_period",    "Mr.",                 "Mr.",                 False),
    Case("abbrev_dots",      "U.S.",                "U.S.",                False),
    Case("scene_descriptor", "INT. OFFICE - DAY",   "INT. OFFICE - DAY",   False),
    Case("large_number",     "3000",                "3000",                False),

    # ---- Clean after safe wrapper-strip — must skip LLM ----
    Case("markdown_bold",    "**42**",              "42",                  False),
    Case("double_quoted",    '"Paris"',             "Paris",               False),
    Case("backticked",       "`code`",              "code",                False),
    Case("final_answer",     "Final Answer: 42",    "42",                  False),
    Case("answer_colon",     "Answer: Paris",       "Paris",               False),
    Case("bold_and_prefix",  "**Final Answer: 42**","42",                  False),

    # ---- Verbose — deterministic leaves alone; gate must route to LLM ----
    Case("verbose_filler",   "The answer is 42",
         "The answer is 42", True),
    Case("verbose_approx",
         "Based on my calculations, the answer is approximately 1500",
         "Based on my calculations, the answer is approximately 1500",
         True),
    Case("long_sentence",
         "I found that the total number of papers published was 127 in 2019",
         "I found that the total number of papers published was 127 in 2019",
         True),

    # ---- Final-Answer tail extraction (the real GAIA regression pattern) ----
    # When a model produces a verbose preamble followed by `Final Answer: X`,
    # take X and drop the preamble. This is what the LLM formatter USED to do
    # unreliably — the deterministic layer now handles it safely.
    Case("tail_simple",
         "Some reasoning about it.\n\nFinal Answer: 142",
         "142", False),
    Case("tail_with_markdown",
         "**The object** is *Tritia gibbosula*, dated to 142 thousand years.\n\nFinal Answer: 142",
         "142", False),
    Case("tail_multi_word",
         "Based on the spreadsheet, the oldest title is shown below.\n\nFinal Answer: Time-Parking 2: Parallel Universe",
         "Time-Parking 2: Parallel Universe", False),
    Case("tail_case_insensitive",
         "Long reasoning paragraph.\n\nfinal answer: egalitarian",
         "egalitarian", False),
    Case("tail_last_wins",
         "Preliminary guess — Final Answer: 41.\nOn re-check it's actually 42.\nFinal Answer: 42",
         "42", False),
    Case("tail_bold_wrapper",
         "Reasoning...\n\nFinal Answer: **Paris**",
         "Paris", False),
    Case("tail_trailing_period",
         "Reasoning concluded.\n\nFinal Answer: dot.",
         "dot.", False),
]

_CASE_IDS = [c.id for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_deterministic_format_table(case: Case):
    from lilith_agent.runner import _deterministic_format

    assert _deterministic_format(case.raw) == case.expected_deterministic


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_gating_table(case: Case):
    from lilith_agent.runner import _needs_llm_formatter

    assert _needs_llm_formatter(case.expected_deterministic) is case.needs_llm


class _RaiseIfCalled:
    def __init__(self):
        self.called = False

    def invoke(self, _messages):  # pragma: no cover - should never run
        self.called = True
        raise AssertionError("LLM formatter was called when it shouldn't have been")


class _FakeModel:
    def __init__(self, response: str):
        self.response = response
        self.called = False

    def invoke(self, _messages):
        self.called = True

        class _Resp:
            pass

        r = _Resp()
        r.content = self.response
        return r


def test_short_clean_answer_bypasses_llm():
    from lilith_agent.runner import _final_formatting_cleanup

    model = _RaiseIfCalled()
    out = _final_formatting_cleanup(model, "What is 6*7?", "42")
    assert out == "42"
    assert model.called is False


def test_verbose_answer_invokes_llm_by_default():
    from lilith_agent.runner import _final_formatting_cleanup

    model = _FakeModel(response="42")
    out = _final_formatting_cleanup(
        model,
        "What is 6*7?",
        "Based on my calculations, the answer is approximately 42",
    )
    assert model.called is True
    assert out == "42"


def test_llm_formatter_disabled_returns_deterministic_only():
    from lilith_agent.runner import _final_formatting_cleanup

    model = _RaiseIfCalled()
    out = _final_formatting_cleanup(
        model,
        "What is 6*7?",
        "Based on my calculations, the answer is approximately 42",
        llm_formatter_enabled=False,
    )
    assert model.called is False
    # deterministic is a no-op on verbose text — raw is returned unchanged
    assert out == "Based on my calculations, the answer is approximately 42"


def test_llm_formatter_disabled_still_applies_deterministic_strip():
    from lilith_agent.runner import _final_formatting_cleanup

    model = _RaiseIfCalled()
    out = _final_formatting_cleanup(
        model,
        "Q",
        "**Final Answer: 42**",
        llm_formatter_enabled=False,
    )
    assert model.called is False
    assert out == "42"


def test_config_llm_formatter_enabled_defaults_on_and_is_env_overridable(monkeypatch):
    from lilith_agent.config import Config

    monkeypatch.delenv("GAIA_LLM_FORMATTER_ENABLED", raising=False)
    assert Config.from_env().llm_formatter_enabled is True

    monkeypatch.setenv("GAIA_LLM_FORMATTER_ENABLED", "false")
    assert Config.from_env().llm_formatter_enabled is False
