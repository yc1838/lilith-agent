# Gemini Rate-Limit Cooldown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Gemini 3 Flash / Gemini 3.1 Pro rate-limit protection for GAIA batch runs, including correct 429 detection, shared lane cooldown, per-question streak skip, cross-task batch pause, and daily-quota batch abort.

**Architecture:** `models.py` owns provider exception classification, Gemini lane cooldown state, per-question 429 scopes, and the shared batch-window counters. `runner.py` owns task-level retry, fresh ephemeral memory per retry, batch pause sleeps, non-success rate-limit answers, and diagnostic abort markers. Tests use fake models/graphs and monkeypatched sleep/time so no external APIs or real waiting are needed.

**Tech Stack:** Python, LangChain `BaseChatModel` / `Runnable`, Tenacity, pytest, standard-library `contextvars`, `collections.deque`, `time.monotonic`, and existing Lilith runner/checkpoint helpers.

**Design Source:** `docs/superpowers/specs/2026-05-06-gemini-rate-limit-cooldown-design.md`

---

## File Structure

- Modify: `src/lilith_agent/models.py`
  - Replace type-only retry classification with `is_retryable_rate_limit`.
  - Add optional `google.genai.errors.ClientError` support.
  - Add `RateLimitCooldownError`, `QuestionRateLimitStreakError`, and `BatchAbortRateLimitError`.
  - Add Gemini lane profiles and shared in-process cooldown registry.
  - Add per-question rate-limit context scope.
  - Add batch sliding-window helpers.
  - Update `_RetryWrapper` and `_BoundRetryWrapper` sync and async paths to use lane metadata, cooldown checks, and exhausted-retry signal conversion.

- Modify: `src/lilith_agent/runner.py`
  - Import rate-limit signals/helpers lazily inside `run_agent_on_questions`.
  - Factor one task invocation into a helper path so retry uses a fresh `ephemeral_memory()` block.
  - Catch `RateLimitCooldownError`, sleep, retry same task once, and avoid success checkpoints on rate-limit failure.
  - Catch `QuestionRateLimitStreakError`, append `AGENT ERROR: RATE LIMITED`, and continue.
  - Catch `BatchAbortRateLimitError`, append `AGENT ERROR: RATE LIMITED`, write `<checkpoint_dir>/rate_limit_abort.json`, and stop the batch.
  - Check the batch sliding window between tasks and sleep when it trips.

- Modify: `tests/test_models.py`
  - Add unit tests for retry predicates, cooldown registry behavior, daily quota classification, same-lane sharing, lane isolation, monotonic timing, bound wrapper propagation, async parity, and question streak.

- Modify: `tests/test_runner.py`
  - Add runner tests for retry-once behavior, fresh ephemeral memory block per retry, no success checkpoint on rate-limit failure, batch abort marker, and batch pause trigger.

---

## Task 1: Add Rate-Limit Exception Classification Tests

**Files:**
- Modify: `tests/test_models.py`
- Modify: `src/lilith_agent/models.py`

- [ ] **Step 1: Write failing tests for Gemini `ClientError` classification**

Add these imports near the top of `tests/test_models.py`:

```python
import pytest

from lilith_agent.models import is_retryable_rate_limit
```

Add this helper and tests to `tests/test_models.py`:

```python
def _make_genai_client_error(code: int):
    pytest.importorskip("google.genai.errors")
    from google.genai.errors import ClientError

    return ClientError(
        code,
        {
            "error": {
                "code": code,
                "status": "RESOURCE_EXHAUSTED" if code == 429 else "INVALID_ARGUMENT",
                "message": "test error",
            }
        },
    )


def test_genai_client_error_429_is_retryable():
    exc = _make_genai_client_error(429)

    assert is_retryable_rate_limit(exc) is True


def test_genai_client_error_400_is_not_retryable():
    exc = _make_genai_client_error(400)

    assert is_retryable_rate_limit(exc) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_models.py::test_genai_client_error_429_is_retryable tests/test_models.py::test_genai_client_error_400_is_not_retryable -v
```

Expected: FAIL because `is_retryable_rate_limit` is not defined/exported yet.

- [ ] **Step 3: Implement retryable predicate**

In `src/lilith_agent/models.py`, replace the `RETRY_EXCEPTIONS` block and old static retry-parameter dict with this shape.

Add imports at the top:

```python
import asyncio
import time
from typing import Any

from tenacity import AsyncRetrying, Retrying, retry_if_exception, stop_after_attempt, wait_exponential
```

Replace the optional exception setup with:

```python
try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = None

try:
    from google.genai.errors import ClientError as GenAIClientError
except ImportError:
    GenAIClientError = None

try:
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    AnthropicRateLimitError = None

try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = None
```

Add below the logger/constants:

```python
def is_retryable_rate_limit(exc: BaseException) -> bool:
    if ResourceExhausted is not None and isinstance(exc, ResourceExhausted):
        return True
    if GenAIClientError is not None and isinstance(exc, GenAIClientError):
        return getattr(exc, "code", None) == 429
    if AnthropicRateLimitError is not None and isinstance(exc, AnthropicRateLimitError):
        return True
    if OpenAIRateLimitError is not None and isinstance(exc, OpenAIRateLimitError):
        return True
    return False
```

Add retry parameter factories:

```python
def _tenacity_sleep(seconds: float) -> None:
    time.sleep(seconds)


async def _async_tenacity_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _base_retry_params() -> dict[str, Any]:
    return dict(
        retry=retry_if_exception(is_retryable_rate_limit),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: log.warning(
            f"LLM Rate Limit (429) hit. Retrying in {retry_state.next_action.sleep}s... "
            f"(Attempt {retry_state.attempt_number}/5)"
        ),
        reraise=True,
    )


def _sync_retry_params() -> dict[str, Any]:
    return {**_base_retry_params(), "sleep": _tenacity_sleep}


def _async_retry_params() -> dict[str, Any]:
    return {**_base_retry_params(), "sleep": _async_tenacity_sleep}
```

Update existing retry loops to call the factories immediately so the old static retry dict can be removed without leaving dead references:

```python
def _generate(self, *args, **kwargs):
    for attempt in Retrying(**_sync_retry_params()):
        with attempt:
            return self.inner._generate(*args, **kwargs)


async def _agenerate(self, *args, **kwargs):
    async for attempt in AsyncRetrying(**_async_retry_params()):
        with attempt:
            return await self.inner._agenerate(*args, **kwargs)
```

In `_BoundRetryWrapper`, update both retry loops the same way:

```python
def invoke(self, input, config=None, **kwargs):
    for attempt in Retrying(**_sync_retry_params()):
        with attempt:
            return self._bound.invoke(input, config=config, **kwargs)


async def ainvoke(self, input, config=None, **kwargs):
    async for attempt in AsyncRetrying(**_async_retry_params()):
        with attempt:
            return await self._bound.ainvoke(input, config=config, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_models.py::test_genai_client_error_429_is_retryable tests/test_models.py::test_genai_client_error_400_is_not_retryable -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lilith_agent/models.py tests/test_models.py
git commit -m "fix: classify gemini genai rate limits"
```

---

## Task 2: Add Gemini Lane Cooldown Tests and Registry

**Files:**
- Modify: `tests/test_models.py`
- Modify: `src/lilith_agent/models.py`

- [ ] **Step 1: Write failing cooldown tests**

Add these imports to `tests/test_models.py`:

```python
from langchain_core.outputs import ChatGeneration, ChatResult

from lilith_agent.models import RateLimitCooldownError, _reset_rate_limit_state_for_tests
```

Add this fixture so Tenacity never uses real wall-clock backoff in model tests:

```python
@pytest.fixture(autouse=True)
def _disable_retry_sleeps(monkeypatch):
    async def _no_async_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("lilith_agent.models._tenacity_sleep", lambda _seconds: None, raising=False)
    monkeypatch.setattr("lilith_agent.models._async_tenacity_sleep", _no_async_sleep, raising=False)
```

Add these fake models and tests:

```python
class _FailingGenerateModel:
    _llm_type = "failing"

    def __init__(self, exc: BaseException):
        self.exc = exc
        self.calls = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        raise self.exc

    async def _agenerate(self, *args, **kwargs):
        self.calls += 1
        raise self.exc

    def bind_tools(self, tools, **kwargs):
        def _raise(_msgs):
            raise self.exc

        return RunnableLambda(_raise)


class _SuccessfulGenerateModel:
    _llm_type = "success"

    def __init__(self):
        self.calls = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    async def _agenerate(self, *args, **kwargs):
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def bind_tools(self, tools, **kwargs):
        return RunnableLambda(lambda msgs: AIMessage(content="ok"))


def test_retry_wrapper_records_first_gemini_cooldown(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    sleeps = []
    monkeypatch.setattr("lilith_agent.models.time.sleep", sleeps.append)

    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(RateLimitCooldownError) as raised:
        wrapper._generate([])

    assert raised.value.provider == "google"
    assert raised.value.model == "gemini-3.1-pro"
    assert raised.value.cooldown_seconds == 60
    assert sleeps == []


def test_retry_wrapper_escalates_gemini_cooldowns(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(RateLimitCooldownError) as first:
        wrapper._generate([])
    with pytest.raises(RateLimitCooldownError) as second:
        wrapper._generate([])
    with pytest.raises(RateLimitCooldownError) as third:
        wrapper._generate([])

    assert first.value.cooldown_seconds == 60
    assert second.value.cooldown_seconds == 120
    assert third.value.cooldown_seconds == 300


def test_success_resets_lane_failure_counter(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    failing = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )
    success = _RetryWrapper.model_construct(
        inner=_SuccessfulGenerateModel(), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(RateLimitCooldownError):
        failing._generate([])
    success._generate([])
    with pytest.raises(RateLimitCooldownError) as raised:
        failing._generate([])

    assert raised.value.cooldown_seconds == 60
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_models.py::test_retry_wrapper_records_first_gemini_cooldown tests/test_models.py::test_retry_wrapper_escalates_gemini_cooldowns tests/test_models.py::test_success_resets_lane_failure_counter -v
```

Expected: FAIL because cooldown exceptions/state are not implemented.

- [ ] **Step 3: Implement lane exceptions and registry**

Add imports in `src/lilith_agent/models.py`:

```python
import time
from dataclasses import dataclass
```

Add constants and exceptions after `_NO_THINK`:

```python
_GEMINI_COOLDOWN_MODELS = {"gemini-3-flash-preview", "gemini-3.1-pro"}
_COOLDOWN_LADDER_SECONDS = (60, 120, 300)
_cooldown_until: dict[tuple[str, str], float] = {}
_rate_limit_exhaustions: dict[tuple[str, str], int] = {}


@dataclass
class RateLimitCooldownError(Exception):
    provider: str
    model: str
    cooldown_seconds: int
    original_error: str

    def __str__(self) -> str:
        return (
            f"rate limited provider={self.provider} model={self.model} "
            f"cooldown={self.cooldown_seconds}s original={self.original_error}"
        )


@dataclass
class QuestionRateLimitStreakError(Exception):
    count: int

    def __str__(self) -> str:
        return f"rate limited question streak reached {self.count}"


@dataclass
class BatchAbortRateLimitError(Exception):
    reason: str
    original_error: str

    def __str__(self) -> str:
        return f"rate limit batch abort: {self.reason}: {self.original_error}"


def _reset_rate_limit_state_for_tests() -> None:
    _cooldown_until.clear()
    _rate_limit_exhaustions.clear()
```

Add helpers:

```python
def _gemini_lane(provider: str | None, model_name: str | None) -> tuple[str, str] | None:
    if provider == "google" and model_name in _GEMINI_COOLDOWN_MODELS:
        return (provider, model_name)
    return None


def _cooldown_seconds_for_exhaustion(count: int) -> int:
    idx = min(max(count, 1), len(_COOLDOWN_LADDER_SECONDS)) - 1
    return _COOLDOWN_LADDER_SECONDS[idx]


def _sleep_active_cooldown(lane: tuple[str, str] | None) -> None:
    if lane is None:
        return
    remaining = _cooldown_until.get(lane, 0.0) - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


async def _sleep_active_cooldown_async(lane: tuple[str, str] | None) -> None:
    if lane is None:
        return
    remaining = _cooldown_until.get(lane, 0.0) - time.monotonic()
    if remaining > 0:
        await asyncio.sleep(remaining)


def _record_success(lane: tuple[str, str] | None) -> None:
    if lane is None:
        return
    _rate_limit_exhaustions[lane] = 0
    _cooldown_until.pop(lane, None)


def _record_exhausted_rate_limit(lane: tuple[str, str], exc: BaseException) -> RateLimitCooldownError:
    count = _rate_limit_exhaustions.get(lane, 0) + 1
    _rate_limit_exhaustions[lane] = count
    cooldown = _cooldown_seconds_for_exhaustion(count)
    _cooldown_until[lane] = time.monotonic() + cooldown
    return RateLimitCooldownError(
        provider=lane[0],
        model=lane[1],
        cooldown_seconds=cooldown,
        original_error=str(exc),
    )
```

Update `_RetryWrapper` fields and methods:

```python
class _RetryWrapper(BaseChatModel):
    inner: BaseChatModel
    provider: str | None = None
    model_name: str | None = None

    @property
    def _llm_type(self) -> str:
        return f"retry-{self.inner._llm_type}"

    def _generate(self, *args, **kwargs):
        lane = _gemini_lane(self.provider, self.model_name)
        _sleep_active_cooldown(lane)
        try:
            for attempt in Retrying(**_sync_retry_params()):
                with attempt:
                    result = self.inner._generate(*args, **kwargs)
                    _record_success(lane)
                    return result
        except Exception as exc:
            if lane is not None and is_retryable_rate_limit(exc):
                raise _record_exhausted_rate_limit(lane, exc) from exc
            raise
```

Leave `_agenerate` unchanged in this task. Task 8 adds failing async tests and then replaces `_agenerate` with the async version of this cooldown/retry logic.

Update `_build._wrap`:

```python
def _wrap(m):
    return _RetryWrapper(inner=m, provider=provider, model_name=model)
```

- [ ] **Step 4: Run cooldown tests**

Run:

```bash
pytest tests/test_models.py::test_retry_wrapper_records_first_gemini_cooldown tests/test_models.py::test_retry_wrapper_escalates_gemini_cooldowns tests/test_models.py::test_success_resets_lane_failure_counter -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lilith_agent/models.py tests/test_models.py
git commit -m "feat: add gemini cooldown registry"
```

---

## Task 3: Add Same-Lane Sharing, Lane Isolation, and Monotonic Tests

**Files:**
- Modify: `tests/test_models.py`
- Modify: `src/lilith_agent/models.py`

- [ ] **Step 1: Write failing sharing/isolation/monotonic tests**

Add tests:

```python
def test_same_gemini_lane_shares_cooldown(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    now = [1000.0]
    sleeps = []
    monkeypatch.setattr("lilith_agent.models.time.monotonic", lambda: now[0])
    monkeypatch.setattr("lilith_agent.models.time.sleep", sleeps.append)
    first = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )
    second = _RetryWrapper.model_construct(
        inner=_SuccessfulGenerateModel(), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(RateLimitCooldownError):
        first._generate([])
    now[0] = 1005.0
    second._generate([])

    assert sleeps == [55.0]


def test_other_gemini_lane_does_not_sleep(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    now = [1000.0]
    sleeps = []
    monkeypatch.setattr("lilith_agent.models.time.monotonic", lambda: now[0])
    monkeypatch.setattr("lilith_agent.models.time.sleep", sleeps.append)
    pro = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )
    flash = _RetryWrapper.model_construct(
        inner=_SuccessfulGenerateModel(), provider="google", model_name="gemini-3-flash-preview"
    )

    with pytest.raises(RateLimitCooldownError):
        pro._generate([])
    now[0] = 1005.0
    flash._generate([])

    assert sleeps == []


def test_unknown_google_model_does_not_get_gemini_cooldown(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-unknown"
    )

    with pytest.raises(type(exc)):
        wrapper._generate([])
```

- [ ] **Step 2: Run tests**

Run:

```bash
pytest tests/test_models.py::test_same_gemini_lane_shares_cooldown tests/test_models.py::test_other_gemini_lane_does_not_sleep tests/test_models.py::test_unknown_google_model_does_not_get_gemini_cooldown -v
```

Expected: PASS if Task 2 implementation is correct. If same-lane sleep fails, fix `_sleep_active_cooldown` to use `_cooldown_until[lane] - time.monotonic()`.

- [ ] **Step 3: Commit tests/fixes**

Run:

```bash
git add src/lilith_agent/models.py tests/test_models.py
git commit -m "test: cover gemini cooldown lane behavior"
```

---

## Task 4: Add Daily Quota / Batch Abort Classification

**Files:**
- Modify: `tests/test_models.py`
- Modify: `src/lilith_agent/models.py`

- [ ] **Step 1: Write failing tests for daily quota metadata**

Add helper and tests:

```python
def _make_genai_quota_error(details: list[dict]):
    pytest.importorskip("google.genai.errors")
    from google.genai.errors import ClientError

    return ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "quota exceeded",
                "details": details,
            }
        },
    )


def test_daily_quota_metadata_raises_batch_abort(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_quota_error(
        [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}],
            }
        ]
    )
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(BatchAbortRateLimitError) as raised:
        wrapper._generate([])

    assert "daily" in raised.value.reason.lower()


def test_long_retry_delay_raises_batch_abort(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_quota_error(
        [
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "900s",
            }
        ]
    )
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(BatchAbortRateLimitError) as raised:
        wrapper._generate([])

    assert "retry" in raised.value.reason.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_models.py::test_daily_quota_metadata_raises_batch_abort tests/test_models.py::test_long_retry_delay_raises_batch_abort -v
```

Expected: FAIL because metadata parsing is not implemented.

- [ ] **Step 3: Implement metadata parsing**

Add helpers to `src/lilith_agent/models.py`:

```python
def _iter_error_details(exc: BaseException) -> list[dict[str, Any]]:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        nested = details.get("error", {}).get("details", details.get("details", []))
        return nested if isinstance(nested, list) else []
    return details if isinstance(details, list) else []


def _parse_retry_delay_seconds(value: Any) -> float | None:
    if isinstance(value, str) and value.endswith("s"):
        try:
            return float(value[:-1])
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _batch_abort_reason(exc: BaseException) -> str | None:
    for detail in _iter_error_details(exc):
        retry_delay = _parse_retry_delay_seconds(detail.get("retryDelay"))
        if retry_delay is not None and retry_delay > 600:
            return f"retry delay {retry_delay:g}s exceeds batch threshold"
        violations = detail.get("violations", [])
        if isinstance(violations, list):
            for violation in violations:
                if isinstance(violation, dict) and "PerDay" in str(violation.get("quotaId", "")):
                    return f"daily quota exhausted: {violation.get('quotaId')}"
        if "PerDay" in str(detail.get("quotaId", "")):
            return f"daily quota exhausted: {detail.get('quotaId')}"
    return None
```

Update `_RetryWrapper._generate` exception path:

```python
if lane is not None and is_retryable_rate_limit(exc):
    reason = _batch_abort_reason(exc)
    if reason is not None:
        raise BatchAbortRateLimitError(reason=reason, original_error=str(exc)) from exc
    raise _record_exhausted_rate_limit(lane, exc) from exc
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_models.py::test_daily_quota_metadata_raises_batch_abort tests/test_models.py::test_long_retry_delay_raises_batch_abort -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lilith_agent/models.py tests/test_models.py
git commit -m "feat: abort batch on gemini daily quota"
```

---

## Task 5: Add Per-Question Streak Scope

**Files:**
- Modify: `tests/test_models.py`
- Modify: `src/lilith_agent/models.py`

- [ ] **Step 1: Write failing streak tests**

Add imports:

```python
from lilith_agent.models import rate_limit_question_scope
```

Add tests:

```python
def test_question_rate_limit_scope_raises_after_50_events():
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)

    with pytest.raises(QuestionRateLimitStreakError) as raised:
        with rate_limit_question_scope():
            for _ in range(50):
                try:
                    raise exc
                except BaseException as caught:
                    from lilith_agent.models import record_rate_limit_observation

                    record_rate_limit_observation(caught)

    assert raised.value.count == 50


def test_question_rate_limit_scope_resets_after_success():
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)

    with rate_limit_question_scope():
        from lilith_agent.models import record_rate_limit_observation, record_rate_limit_success

        for _ in range(49):
            record_rate_limit_observation(exc)
        record_rate_limit_success()
        for _ in range(49):
            record_rate_limit_observation(exc)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_models.py::test_question_rate_limit_scope_raises_after_50_events tests/test_models.py::test_question_rate_limit_scope_resets_after_success -v
```

Expected: FAIL because question scope helpers are missing.

- [ ] **Step 3: Implement contextvar scope**

Add imports in `models.py`:

```python
from contextlib import contextmanager
from contextvars import ContextVar
```

Add constants/state:

```python
_QUESTION_STREAK_LIMIT = 50
_question_rate_limit_streak: ContextVar[int | None] = ContextVar("question_rate_limit_streak", default=None)
```

Add helpers:

```python
@contextmanager
def rate_limit_question_scope():
    token = _question_rate_limit_streak.set(0)
    try:
        yield
    finally:
        _question_rate_limit_streak.reset(token)


def record_rate_limit_observation(exc: BaseException) -> None:
    current = _question_rate_limit_streak.get()
    if current is None or not is_retryable_rate_limit(exc):
        return
    current += 1
    _question_rate_limit_streak.set(current)
    if current >= _QUESTION_STREAK_LIMIT:
        raise QuestionRateLimitStreakError(count=current) from exc


def record_rate_limit_success() -> None:
    if _question_rate_limit_streak.get() is not None:
        _question_rate_limit_streak.set(0)
```

Update `_RetryWrapper._generate` so each retryable exception observed by tenacity records an observation. The 50-event budget is counted per lower-level retry attempt, not per outer GAIA task or per exhausted wrapper call. Use a Tenacity `after` callback or wrap the inner call in a `try/except` inside the attempt:

```python
try:
    result = self.inner._generate(*args, **kwargs)
except Exception as observed:
    record_rate_limit_observation(observed)
    raise
record_rate_limit_success()
_record_success(lane)
return result
```

Do the same in `_BoundRetryWrapper.invoke`.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_models.py::test_question_rate_limit_scope_raises_after_50_events tests/test_models.py::test_question_rate_limit_scope_resets_after_success -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lilith_agent/models.py tests/test_models.py
git commit -m "feat: track per-question rate limit streaks"
```

---

## Task 6: Add Batch Sliding Window Helpers

**Files:**
- Modify: `tests/test_models.py`
- Modify: `src/lilith_agent/models.py`

- [ ] **Step 1: Write failing batch-window tests**

Add imports:

```python
from lilith_agent.models import batch_rate_limit_pause_seconds, clear_batch_rate_limit_window
```

Add test:

```python
def test_batch_window_triggers_pause_after_70_rate_limits_in_100_outcomes():
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    from lilith_agent.models import record_rate_limit_observation, record_rate_limit_success

    for _ in range(70):
        record_rate_limit_observation(exc)
    for _ in range(30):
        record_rate_limit_success()

    assert batch_rate_limit_pause_seconds() == 300
    clear_batch_rate_limit_window()
    assert batch_rate_limit_pause_seconds() is None


def test_batch_window_does_not_trigger_below_threshold():
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    from lilith_agent.models import record_rate_limit_observation, record_rate_limit_success

    for _ in range(69):
        record_rate_limit_observation(exc)
    for _ in range(31):
        record_rate_limit_success()

    assert batch_rate_limit_pause_seconds() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_models.py::test_batch_window_triggers_pause_after_70_rate_limits_in_100_outcomes tests/test_models.py::test_batch_window_does_not_trigger_below_threshold -v
```

Expected: FAIL because batch-window helpers are missing.

- [ ] **Step 3: Implement sliding-window state**

Add import:

```python
from collections import deque
```

Add constants/state:

```python
_BATCH_WINDOW_SIZE = 100
_BATCH_WINDOW_RATE_LIMIT_THRESHOLD = 70
_BATCH_PAUSE_LADDER_SECONDS = (300, 600, 1200)
_batch_rate_limit_window: deque[bool] = deque(maxlen=_BATCH_WINDOW_SIZE)
_batch_pause_count = 0
```

Update `_reset_rate_limit_state_for_tests()`:

```python
def _reset_rate_limit_state_for_tests() -> None:
    global _batch_pause_count
    _cooldown_until.clear()
    _rate_limit_exhaustions.clear()
    _batch_rate_limit_window.clear()
    _batch_pause_count = 0
```

Update `record_rate_limit_observation` and `record_rate_limit_success`:

```python
def record_rate_limit_observation(exc: BaseException) -> None:
    if not is_retryable_rate_limit(exc):
        return
    _batch_rate_limit_window.append(True)
    current = _question_rate_limit_streak.get()
    if current is None:
        return
    current += 1
    _question_rate_limit_streak.set(current)
    if current >= _QUESTION_STREAK_LIMIT:
        raise QuestionRateLimitStreakError(count=current) from exc


def record_rate_limit_success() -> None:
    _batch_rate_limit_window.append(False)
    if _question_rate_limit_streak.get() is not None:
        _question_rate_limit_streak.set(0)
```

Add helpers:

```python
def batch_rate_limit_pause_seconds() -> int | None:
    global _batch_pause_count
    if len(_batch_rate_limit_window) < _BATCH_WINDOW_SIZE:
        return None
    if sum(_batch_rate_limit_window) < _BATCH_WINDOW_RATE_LIMIT_THRESHOLD:
        return None
    _batch_pause_count += 1
    idx = min(_batch_pause_count - 1, len(_BATCH_PAUSE_LADDER_SECONDS) - 1)
    return _BATCH_PAUSE_LADDER_SECONDS[idx]


def clear_batch_rate_limit_window() -> None:
    _batch_rate_limit_window.clear()
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_models.py::test_batch_window_triggers_pause_after_70_rate_limits_in_100_outcomes tests/test_models.py::test_batch_window_does_not_trigger_below_threshold -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lilith_agent/models.py tests/test_models.py
git commit -m "feat: track batch rate limit window"
```

---

## Task 7: Wire Bound Retry Wrapper to Lane Metadata

**Files:**
- Modify: `tests/test_models.py`
- Modify: `src/lilith_agent/models.py`

- [ ] **Step 1: Write failing bound-wrapper cooldown test**

Add test:

```python
def test_bound_retry_wrapper_raises_cooldown_for_gemini_lane(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    bound = wrapper.bind_tools([])

    with pytest.raises(RateLimitCooldownError) as raised:
        bound.invoke([("user", "hi")])

    assert raised.value.model == "gemini-3.1-pro"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_models.py::test_bound_retry_wrapper_raises_cooldown_for_gemini_lane -v
```

Expected: FAIL because `_BoundRetryWrapper` does not receive provider/model metadata.

- [ ] **Step 3: Update `_BoundRetryWrapper`**

Update `bind_tools` in `_RetryWrapper`:

```python
def bind_tools(self, tools: Any, **kwargs: Any):
    bound = self.inner.bind_tools(tools, **kwargs)
    return _BoundRetryWrapper(bound=bound, provider=self.provider, model_name=self.model_name)
```

Update `_BoundRetryWrapper`:

```python
class _BoundRetryWrapper(Runnable):
    def __init__(self, bound, provider: str | None = None, model_name: str | None = None):
        self._bound = bound
        self._provider = provider
        self._model_name = model_name

    def invoke(self, input, config=None, **kwargs):
        lane = _gemini_lane(self._provider, self._model_name)
        _sleep_active_cooldown(lane)
        try:
            for attempt in Retrying(**_sync_retry_params()):
                with attempt:
                    try:
                        result = self._bound.invoke(input, config=config, **kwargs)
                    except Exception as observed:
                        record_rate_limit_observation(observed)
                        raise
                    record_rate_limit_success()
                    _record_success(lane)
                    return result
        except Exception as exc:
            if lane is not None and is_retryable_rate_limit(exc):
                reason = _batch_abort_reason(exc)
                if reason is not None:
                    raise BatchAbortRateLimitError(reason=reason, original_error=str(exc)) from exc
                raise _record_exhausted_rate_limit(lane, exc) from exc
            raise
```

- [ ] **Step 4: Run bound-wrapper tests**

Run:

```bash
pytest tests/test_models.py::test_retry_wrapper_bind_tools_returns_runnable tests/test_models.py::test_retry_wrapper_bound_invoke_passes_through tests/test_models.py::test_bound_retry_wrapper_raises_cooldown_for_gemini_lane -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lilith_agent/models.py tests/test_models.py
git commit -m "feat: apply cooldown to bound model calls"
```

---

## Task 8: Add Async Retry/Cooldown Parity

**Files:**
- Modify: `tests/test_models.py`
- Modify: `src/lilith_agent/models.py`

- [ ] **Step 1: Write failing async parity tests**

Add tests:

```python
@pytest.mark.asyncio
async def test_async_retry_wrapper_raises_cooldown_for_gemini_lane(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.asyncio.sleep", AsyncMock())
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(RateLimitCooldownError) as raised:
        await wrapper._agenerate([])

    assert raised.value.model == "gemini-3.1-pro"


@pytest.mark.asyncio
async def test_async_bound_retry_wrapper_raises_cooldown_for_gemini_lane(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.asyncio.sleep", AsyncMock())
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    bound = wrapper.bind_tools([])

    with pytest.raises(RateLimitCooldownError) as raised:
        await bound.ainvoke([("user", "hi")])

    assert raised.value.model == "gemini-3.1-pro"
```

Add this import if missing:

```python
from unittest.mock import AsyncMock
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_models.py::test_async_retry_wrapper_raises_cooldown_for_gemini_lane tests/test_models.py::test_async_bound_retry_wrapper_raises_cooldown_for_gemini_lane -v
```

Expected: FAIL because `_agenerate` and `ainvoke` still use the old generic retry behavior.

- [ ] **Step 3: Implement async parity in wrappers**

Update `_RetryWrapper._agenerate`:

```python
async def _agenerate(self, *args, **kwargs):
    lane = _gemini_lane(self.provider, self.model_name)
    await _sleep_active_cooldown_async(lane)
    try:
        async for attempt in AsyncRetrying(**_async_retry_params()):
            with attempt:
                try:
                    result = await self.inner._agenerate(*args, **kwargs)
                except Exception as observed:
                    record_rate_limit_observation(observed)
                    raise
                record_rate_limit_success()
                _record_success(lane)
                return result
    except Exception as exc:
        if lane is not None and is_retryable_rate_limit(exc):
            reason = _batch_abort_reason(exc)
            if reason is not None:
                raise BatchAbortRateLimitError(reason=reason, original_error=str(exc)) from exc
            raise _record_exhausted_rate_limit(lane, exc) from exc
        raise
```

Update `_BoundRetryWrapper.ainvoke`:

```python
async def ainvoke(self, input, config=None, **kwargs):
    lane = _gemini_lane(self._provider, self._model_name)
    await _sleep_active_cooldown_async(lane)
    try:
        async for attempt in AsyncRetrying(**_async_retry_params()):
            with attempt:
                try:
                    result = await self._bound.ainvoke(input, config=config, **kwargs)
                except Exception as observed:
                    record_rate_limit_observation(observed)
                    raise
                record_rate_limit_success()
                _record_success(lane)
                return result
    except Exception as exc:
        if lane is not None and is_retryable_rate_limit(exc):
            reason = _batch_abort_reason(exc)
            if reason is not None:
                raise BatchAbortRateLimitError(reason=reason, original_error=str(exc)) from exc
            raise _record_exhausted_rate_limit(lane, exc) from exc
        raise
```

- [ ] **Step 4: Run async parity tests**

Run:

```bash
pytest tests/test_models.py::test_async_retry_wrapper_raises_cooldown_for_gemini_lane tests/test_models.py::test_async_bound_retry_wrapper_raises_cooldown_for_gemini_lane -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lilith_agent/models.py tests/test_models.py
git commit -m "feat: apply gemini cooldown to async model calls"
```

---

## Task 9: Wire Runner Retry-Once Behavior

**Files:**
- Modify: `tests/test_runner.py`
- Modify: `src/lilith_agent/runner.py`

- [ ] **Step 1: Write failing runner retry test**

Add imports to `tests/test_runner.py`:

```python
from langchain_core.messages import AIMessage

from lilith_agent.config import Config
from lilith_agent.models import RateLimitCooldownError
from lilith_agent.runner import run_agent_on_questions
```

Add shared runner-test fixtures, fake graph, and test:

```python
@pytest.fixture
def runner_test_config() -> Config:
    return Config(
        cheap_provider="google",
        cheap_model="gemini-3-flash-preview",
        strong_provider="google",
        strong_model="gemini-3.1-pro",
        extra_strong_provider="google",
        extra_strong_model="gemini-3.1-pro",
        vision_provider="fal",
        vision_model="gemini-3-flash-preview",
        fal_vision_api_key="",
        api_url="",
        checkpoint_dir="",
        whisper_model="base",
        anthropic_api_key="",
        google_api_key="",
        huggingface_api_key="",
        tavily_api_key="",
        lmstudio_base_url="",
        max_tokens=1024,
        llm_formatter_enabled=True,
    )


@pytest.fixture(autouse=True)
def _isolate_runner_model_setup(monkeypatch, runner_test_config):
    monkeypatch.setattr(Config, "from_env", classmethod(lambda cls: runner_test_config))
    monkeypatch.setattr("lilith_agent.models.get_cheap_model", lambda cfg: object())


class _GraphFailsOnceWithCooldown:
    def __init__(self):
        self.calls = 0
        self.thread_ids = []

    def invoke(self, state, config):
        self.calls += 1
        self.thread_ids.append(config["configurable"]["thread_id"])
        if self.calls == 1:
            raise RateLimitCooldownError(
                provider="google",
                model="gemini-3.1-pro",
                cooldown_seconds=12,
                original_error="429",
            )
        return {"messages": [AIMessage(content="Final Answer: 42")]}


def test_runner_retries_same_question_once_after_cooldown(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("lilith_agent.runner._final_formatting_cleanup", lambda model, question, raw, llm_formatter_enabled=True: raw)
    sleeps = []
    monkeypatch.setattr("lilith_agent.runner.time.sleep", sleeps.append)
    graph = _GraphFailsOnceWithCooldown()

    answers = run_agent_on_questions(
        graph,
        [{"task_id": "task-1", "question": "What is 6*7?"}],
        tmp_path,
    )

    assert graph.calls == 2
    assert graph.thread_ids == ["task-1", "task-1"]
    assert sleeps == [12]
    assert answers == [{"task_id": "task-1", "submitted_answer": "Final Answer: 42"}]
    assert (tmp_path / "task-1.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_runner.py::test_runner_retries_same_question_once_after_cooldown -v
```

Expected: FAIL because runner does not catch `RateLimitCooldownError` specially.

- [ ] **Step 3: Implement retry-once flow**

Add import at top of `runner.py`:

```python
import time
```

Inside `run_agent_on_questions`, add lazy imports near existing config imports:

```python
from lilith_agent.models import RateLimitCooldownError, rate_limit_question_scope
```

Create local helper inside `run_agent_on_questions` before the `for` loop:

```python
def _invoke_task_once(task_state: dict, task_id: str):
    from lilith_agent.memory import ephemeral_memory

    with rate_limit_question_scope():
        with ephemeral_memory():
            return graph.invoke(task_state, {"configurable": {"thread_id": task_id}})
```

Replace the existing `try/except` around `ephemeral_memory()` with:

```python
try:
    try:
        result = _invoke_task_once(state, task_id)
    except RateLimitCooldownError as exc:
        log_runner.warning(
            "[runner] task=%s rate limited provider=%s model=%s cooldown=%s",
            task_id,
            exc.provider,
            exc.model,
            exc.cooldown_seconds,
        )
        time.sleep(exc.cooldown_seconds)
        result = _invoke_task_once(state, task_id)
except RateLimitCooldownError as exc:
    log_runner.warning("[runner] task=%s rate limited after retry: %s", task_id, exc)
    answers.append({"task_id": task_id, "submitted_answer": "AGENT ERROR: RATE LIMITED"})
    continue
except Exception as exc:
    log_runner.warning("[runner] task=%s agent error: %s", task_id, exc)
    answers.append({"task_id": task_id, "submitted_answer": f"AGENT ERROR: {exc}"})
    continue
```

- [ ] **Step 4: Run test**

Run:

```bash
pytest tests/test_runner.py::test_runner_retries_same_question_once_after_cooldown -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lilith_agent/runner.py tests/test_runner.py
git commit -m "feat: retry gaia task once after cooldown"
```

---

## Task 10: Runner Rate-Limit Failure Does Not Checkpoint

**Files:**
- Modify: `tests/test_runner.py`
- Modify: `src/lilith_agent/runner.py`

- [ ] **Step 1: Write failing no-checkpoint test**

Add fake graph and test:

```python
class _GraphAlwaysCooldown:
    def __init__(self):
        self.calls = 0

    def invoke(self, state, config):
        self.calls += 1
        raise RateLimitCooldownError(
            provider="google",
            model="gemini-3.1-pro",
            cooldown_seconds=3,
            original_error="429",
        )


def test_runner_does_not_checkpoint_when_rate_limited_twice(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("lilith_agent.runner._final_formatting_cleanup", lambda model, question, raw, llm_formatter_enabled=True: raw)
    monkeypatch.setattr("lilith_agent.runner.time.sleep", lambda _: None)
    graph = _GraphAlwaysCooldown()

    answers = run_agent_on_questions(
        graph,
        [{"task_id": "task-rl", "question": "rate limited?"}],
        tmp_path,
    )

    assert graph.calls == 2
    assert answers == [{"task_id": "task-rl", "submitted_answer": "AGENT ERROR: RATE LIMITED"}]
    assert not (tmp_path / "task-rl.json").exists()
```

- [ ] **Step 2: Run test**

Run:

```bash
pytest tests/test_runner.py::test_runner_does_not_checkpoint_when_rate_limited_twice -v
```

Expected: PASS if Task 9 implementation is correct. If it fails by writing a checkpoint, ensure the rate-limit exception path `continue`s before formatting/checkpointing.

- [ ] **Step 3: Commit test/fix**

Run:

```bash
git add src/lilith_agent/runner.py tests/test_runner.py
git commit -m "test: avoid checkpoint on rate limit failure"
```

---

## Task 11: Runner Fresh Ephemeral Memory Per Retry

**Files:**
- Modify: `tests/test_runner.py`
- Modify: `src/lilith_agent/runner.py`

- [ ] **Step 1: Write failing fresh-memory test**

Add test:

```python
def test_runner_uses_fresh_ephemeral_memory_for_retry(monkeypatch, tmp_path: Path):
    graph = _GraphFailsOnceWithCooldown()
    events = []

    class _FakeEphemeralMemory:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")

    monkeypatch.setattr("lilith_agent.memory.ephemeral_memory", lambda: _FakeEphemeralMemory())
    monkeypatch.setattr("lilith_agent.runner._final_formatting_cleanup", lambda model, question, raw, llm_formatter_enabled=True: raw)
    monkeypatch.setattr("lilith_agent.runner.time.sleep", lambda _: None)

    run_agent_on_questions(
        graph,
        [{"task_id": "task-memory", "question": "What is isolated?"}],
        tmp_path,
    )

    assert events == ["enter", "exit", "enter", "exit"]
```

- [ ] **Step 2: Run test**

Run:

```bash
pytest tests/test_runner.py::test_runner_uses_fresh_ephemeral_memory_for_retry -v
```

Expected: PASS if `_invoke_task_once` imports and opens `ephemeral_memory()` per attempt. If it fails, ensure the helper wraps exactly one graph invocation.

- [ ] **Step 3: Commit**

Run:

```bash
git add src/lilith_agent/runner.py tests/test_runner.py
git commit -m "test: verify fresh memory on cooldown retry"
```

---

## Task 12: Runner Question Streak and Batch Abort Handling

**Files:**
- Modify: `tests/test_runner.py`
- Modify: `src/lilith_agent/runner.py`

- [ ] **Step 1: Write failing runner tests**

Add imports:

```python
import json

from lilith_agent.models import BatchAbortRateLimitError, QuestionRateLimitStreakError
```

Add fake graphs and tests:

```python
class _GraphQuestionStreak:
    def invoke(self, state, config):
        raise QuestionRateLimitStreakError(count=50)


class _GraphBatchAbort:
    def invoke(self, state, config):
        raise BatchAbortRateLimitError(reason="daily quota exhausted", original_error="429")


def test_runner_skips_question_on_rate_limit_streak(tmp_path: Path):
    answers = run_agent_on_questions(
        _GraphQuestionStreak(),
        [
            {"task_id": "task-streak", "question": "first"},
            {"task_id": "task-next", "question": "second"},
        ],
        tmp_path,
    )

    assert answers[0] == {"task_id": "task-streak", "submitted_answer": "AGENT ERROR: RATE LIMITED"}
    assert not (tmp_path / "task-streak.json").exists()


def test_runner_stops_batch_and_writes_abort_marker_on_daily_quota(tmp_path: Path):
    answers = run_agent_on_questions(
        _GraphBatchAbort(),
        [
            {"task_id": "task-abort", "question": "first"},
            {"task_id": "task-never", "question": "second"},
        ],
        tmp_path,
    )

    assert answers == [{"task_id": "task-abort", "submitted_answer": "AGENT ERROR: RATE LIMITED"}]
    marker = tmp_path / "rate_limit_abort.json"
    assert marker.exists()
    data = json.loads(marker.read_text())
    assert data["task_id"] == "task-abort"
    assert data["reason"] == "daily quota exhausted"
    assert not (tmp_path / "task-abort.json").exists()
    assert not (tmp_path / "task-never.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_runner.py::test_runner_skips_question_on_rate_limit_streak tests/test_runner.py::test_runner_stops_batch_and_writes_abort_marker_on_daily_quota -v
```

Expected: FAIL because runner does not catch these exceptions specially.

- [ ] **Step 3: Implement runner handlers**

Update runner lazy imports:

```python
from lilith_agent.models import (
    BatchAbortRateLimitError,
    QuestionRateLimitStreakError,
    RateLimitCooldownError,
    rate_limit_question_scope,
)
```

Add exception handlers before the generic `except Exception`:

```python
except QuestionRateLimitStreakError as exc:
    log_runner.warning("[runner] task=%s rate limit streak: %s", task_id, exc)
    answers.append({"task_id": task_id, "submitted_answer": "AGENT ERROR: RATE LIMITED"})
    continue
except BatchAbortRateLimitError as exc:
    log_runner.warning("[runner] task=%s batch abort rate limit: %s", task_id, exc)
    answers.append({"task_id": task_id, "submitted_answer": "AGENT ERROR: RATE LIMITED"})
    _write_checkpoint_atomic(
        checkpoint_root / "rate_limit_abort.json",
        {
            "task_id": task_id,
            "reason": exc.reason,
            "original_error": exc.original_error,
        },
    )
    return answers
```

Ensure these handlers catch failures from both the first invocation and the retry invocation.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_runner.py::test_runner_skips_question_on_rate_limit_streak tests/test_runner.py::test_runner_stops_batch_and_writes_abort_marker_on_daily_quota -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lilith_agent/runner.py tests/test_runner.py
git commit -m "feat: handle rate limit streak and batch abort"
```

---

## Task 13: Runner Batch Pause Between Tasks

**Files:**
- Modify: `tests/test_runner.py`
- Modify: `src/lilith_agent/runner.py`

- [ ] **Step 1: Write failing batch-pause test**

Add fake graph and test:

```python
class _GraphAlwaysSucceeds:
    def __init__(self):
        self.calls = 0

    def invoke(self, state, config):
        self.calls += 1
        return {"messages": [AIMessage(content=f"answer-{self.calls}")]}


def test_runner_pauses_batch_when_window_trips(monkeypatch, tmp_path: Path):
    pauses = [300, None]
    sleeps = []
    monkeypatch.setattr("lilith_agent.models.batch_rate_limit_pause_seconds", lambda: pauses.pop(0))
    monkeypatch.setattr("lilith_agent.models.clear_batch_rate_limit_window", lambda: None)
    monkeypatch.setattr("lilith_agent.runner.time.sleep", sleeps.append)
    monkeypatch.setattr("lilith_agent.runner._final_formatting_cleanup", lambda model, question, raw, llm_formatter_enabled=True: raw)

    answers = run_agent_on_questions(
        _GraphAlwaysSucceeds(),
        [
            {"task_id": "task-a", "question": "a"},
            {"task_id": "task-b", "question": "b"},
        ],
        tmp_path,
    )

    assert sleeps == [300]
    assert [answer["task_id"] for answer in answers] == ["task-a", "task-b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_runner.py::test_runner_pauses_batch_when_window_trips -v
```

Expected: FAIL because runner does not check batch pause helper.

- [ ] **Step 3: Implement pause check after each task**

Update imports:

```python
from lilith_agent.models import clear_batch_rate_limit_window, batch_rate_limit_pause_seconds
```

After appending each successful answer and after each non-batch-abort rate-limit answer, call:

```python
pause_seconds = batch_rate_limit_pause_seconds()
if pause_seconds is not None:
    log_runner.warning("[runner] pausing batch for %ss due to rate limit window", pause_seconds)
    time.sleep(pause_seconds)
    clear_batch_rate_limit_window()
```

To avoid duplicating this block, define a local helper inside `run_agent_on_questions`:

```python
def _maybe_pause_for_batch_rate_limit() -> None:
    pause_seconds = batch_rate_limit_pause_seconds()
    if pause_seconds is None:
        return
    log_runner.warning("[runner] pausing batch for %ss due to rate limit window", pause_seconds)
    time.sleep(pause_seconds)
    clear_batch_rate_limit_window()
```

Call `_maybe_pause_for_batch_rate_limit()` immediately before each `continue` for non-abort task failures and after appending successful answers.

- [ ] **Step 4: Run test**

Run:

```bash
pytest tests/test_runner.py::test_runner_pauses_batch_when_window_trips -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lilith_agent/runner.py tests/test_runner.py
git commit -m "feat: pause batch on high rate limit window"
```

---

## Task 14: Full Verification and Cleanup

**Files:**
- Modify only if verification reveals regressions: `src/lilith_agent/models.py`, `src/lilith_agent/runner.py`, `tests/test_models.py`, `tests/test_runner.py`

- [ ] **Step 1: Run focused test suite**

Run:

```bash
pytest tests/test_models.py tests/test_runner.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run broader tests**

Run:

```bash
pytest -q
```

Expected: all tests pass. If any test fails, classify it before accepting it: either fix the regression, or document the exact failing test name, command output, and why it is an existing environment-bound failure unrelated to this patch. Do not claim a full pass unless `pytest -q` exits 0.

- [ ] **Step 3: Inspect final diff**

Run:

```bash
git diff -- src/lilith_agent/models.py src/lilith_agent/runner.py tests/test_models.py tests/test_runner.py
```

Expected:

- No edits to `memory.py`.
- No edits to `app.py` web-search dedup.
- No embedding dependencies/schema changes.
- No modifications to unrelated untracked files.

- [ ] **Step 4: Commit any verification fixes**

If Step 1 or Step 2 required fixes, run:

```bash
git add src/lilith_agent/models.py src/lilith_agent/runner.py tests/test_models.py tests/test_runner.py
git commit -m "fix: stabilize gemini cooldown tests"
```

If no fixes were required, skip this commit.

---

## Self-Review Against Spec

- [ ] Correct Gemini 429 detection: Task 1.
- [ ] Non-429 `ClientError` does not retry: Task 1.
- [ ] Shared lane cooldown with 60/120/300 seconds: Task 2.
- [ ] Same-lane sharing and other-lane isolation: Task 3.
- [ ] Monotonic time for cooldown: Task 3.
- [ ] Daily/RPD batch abort: Task 4 and Task 12.
- [ ] Per-question 50-event streak: Task 5 and Task 12.
- [ ] Cross-task sliding window and batch pause: Task 6 and Task 13.
- [ ] Bound wrapper cooldown propagation: Task 7.
- [ ] Async `_agenerate` / `ainvoke` cooldown parity: Task 8.
- [ ] Fresh `ephemeral_memory()` retry: Task 9 and Task 11.
- [ ] No success checkpoint on rate-limit failure: Task 10.
- [ ] Unknown model does not get Gemini cooldown profile: Task 3.
- [ ] No `memory.py`, `app.py`, embedding, or unrelated-file changes: Task 14.
