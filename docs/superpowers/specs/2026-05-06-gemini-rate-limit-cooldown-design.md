---
title: Gemini Rate-Limit Cooldown Design
date: 2026-05-06
status: revised-for-review
---

# Gemini Rate-Limit Cooldown Design

## Context

Lilith currently wraps chat model calls in `_RetryWrapper` with tenacity exponential backoff for provider rate-limit exceptions. This protects a single model call, but it does not coordinate across calls or GAIA tasks. If Gemini quota pressure persists after retries are exhausted, the runner can move to the next task and immediately repeat the same retry ladder against the same overloaded model lane.

Current Gemini retry detection also needs tightening. The installed `langchain-google-genai` path catches `google.genai.errors.ClientError`, while `models.py` currently only registers `google.api_core.exceptions.ResourceExhausted` for Google. A cooldown layer must first classify real Gemini 429s correctly, without retrying non-429 client errors such as `400 INVALID_ARGUMENT`.

For current usage, only two Google text models need first-class cooldown behavior:

| Model | RPM | TPM | RPD |
| --- | ---: | ---: | ---: |
| `gemini-3-flash-preview` | 2000 | 3,000,000 | 100,000 |
| `gemini-3.1-pro` | 1000 | 5,000,000 | 50,000 |

Gemma, embeddings, image, audio, video, and grounding quotas are out of scope for this patch.

## Goals

1. Correctly detect retryable Gemini 429s from the actual GenAI client exception shape.
2. Retain per-call exponential backoff for transient single-call hiccups.
3. Add shared lane cooldown for persistent retry exhaustion.
4. Skip a single GAIA question when it hits the per-question 429 streak threshold.
5. Pause the whole batch when the recent cross-task 429 rate is high.
6. Stop the batch on daily-quota/RPD exhaustion instead of polling every few minutes.
7. Preserve normal non-rate-limit error handling.

## Non-goals

- No memory embedding implementation.
- No embedding cooldown lane.
- No Gemma-specific token cap or quota profile.
- No persistent cross-process rate-limit database.
- No external queue, scheduler, or distributed throttler.
- No change to web-search Jaccard dedup.

## Proposed Architecture

### Model lanes

Represent each rate-limited Gemini model as a lane keyed by provider and model:

- `google:gemini-3-flash-preview`
- `google:gemini-3.1-pro`

Unknown models should continue to use generic retry behavior. They must not inherit Gemini-specific cooldown profiles or Gemini quota assumptions.

### Retryable rate-limit predicate

Replace type-only retry classification with a custom predicate, for example `is_retryable_rate_limit(exc)`.

The predicate should return true for:

- `google.api_core.exceptions.ResourceExhausted`
- `google.genai.errors.ClientError` only when `exc.code == 429`
- Anthropic `RateLimitError`
- OpenAI `RateLimitError`

The predicate should return false for:

- `google.genai.errors.ClientError` with any non-429 code
- generic `ClientError`/`ValueError`/tool errors
- invalid request errors such as `400 INVALID_ARGUMENT`

Tenacity should use this predicate rather than `retry_if_exception_type(RETRY_EXCEPTIONS)`. This keeps 429 handling broad enough for real Gemini calls while preventing retries/cooldowns for deterministic bad requests.

### Quota metadata classification

When a retryable Gemini 429 is observed, inspect available metadata before deciding whether to apply a short cooldown.

Use defensive parsing against `exc.details` because Gemini error payloads may vary. The implementation should look for:

- `RetryInfo` details with `retryDelay`
- `QuotaFailure` details with a `quotaId`
- quota identifiers containing daily-scope markers such as `PerDay`

If `retryDelay` is greater than 600 seconds, or if a quota identifier indicates a daily/RPD limit, raise a batch-abort signal instead of applying the 60/120/300 second lane cooldown. Waiting 300 seconds does not fix `50K/day` exhaustion on `gemini-3.1-pro`.

If metadata is absent or unparseable, fall back to the normal lane cooldown ladder.

### In-process cooldown registry

Add a small module-level registry in `models.py`:

```text
(provider, model) -> cooldown_until_monotonic_seconds
(provider, model) -> consecutive_rate_limit_exhaustions
```

Use `time.monotonic()` for all cooldown calculations.

Before invoking the wrapped model, `_RetryWrapper` checks whether its lane has an active cooldown. If active, it sleeps until the cooldown expires, then proceeds with the existing tenacity retry loop.

After a successful call, the lane's consecutive rate-limit exhaustion count resets to zero.

Cooldown state is shared across `_RetryWrapper` instances for the same lane and isolated across different lanes. A cooldown on `google:gemini-3.1-pro` must not delay a `google:gemini-3-flash-preview` call.

### Cooldown trigger and duration

The cooldown layer sits on top of the existing tenacity retry ladder. A normal exhausted call still spends time inside tenacity first. With the current retry settings, the rough pre-cooldown wait sequence is approximately:

```text
4s -> 8s -> 16s -> 32s -> 60s
```

Only after tenacity exhausts retryable 429 attempts should the wrapper record lane exhaustion and raise a rate-limit signal.

Use this conservative escalating cooldown for repeated failures on the same lane:

| Consecutive exhausted rate limits | Cooldown |
| ---: | ---: |
| 1 | 60 seconds |
| 2 | 120 seconds |
| 3+ | 300 seconds |

If Gemini supplies a shorter explicit retry delay for non-daily quota pressure, a future patch can prefer it. This patch only needs the daily-vs-short-cooldown distinction.

### Rate-limit signals

Introduce project-level exceptions:

- `RateLimitCooldownError`
  - Raised when one model invocation exhausts tenacity retries on a retryable, non-daily rate limit.
  - Carries provider, model, cooldown seconds, and original exception text.

- `QuestionRateLimitStreakError`
  - Raised when the current GAIA question reaches the per-question 429 streak threshold.
  - Used to skip the current question without burning more model calls.

- `BatchAbortRateLimitError`
  - Raised when metadata indicates daily/RPD exhaustion or a retry delay longer than 600 seconds.
  - Used to stop the batch rather than polling through every remaining task.

Non-rate-limit exceptions should keep their existing behavior and should not be rewritten as cooldown errors.

### Per-question streak threshold

The runner should execute each task inside a rate-limit question scope. Within that scope, `_RetryWrapper` records each observed retryable 429 event for the current question.

If one question accumulates 50 consecutive retryable 429 events, abort that question with `QuestionRateLimitStreakError`.

The streak resets when:

- the current question finishes,
- a model call succeeds,
- a non-rate-limit error occurs,
- the runner moves to the next GAIA task.

This protects against a single hard question or loop burning retry ladders indefinitely.

### Cross-task sliding window

Maintain an in-process sliding window over recent model-call outcomes during a GAIA batch. One outcome means one lower-level model request attempt observed by the retry wrapper, including tenacity retry attempts; it does not mean one GAIA task.

```text
window size: 100 model-call outcomes
pause condition: >= 70 rate-limited outcomes in the window
```

When the pause condition is met, the runner should pause the whole batch with exponential backoff:

| Consecutive batch pauses | Pause |
| ---: | ---: |
| 1 | 300 seconds |
| 2 | 600 seconds |
| 3+ | 1200 seconds |

After a batch pause, clear the sliding window so the same old failures do not immediately trigger another pause.

This is separate from lane cooldown. Lane cooldown handles the next call on a specific model; the sliding window handles the case where many tasks across the batch are hitting the same quota wall.

### Runner behavior

`runner.py::run_agent_on_questions` is currently sequential. This design assumes sequential batch execution and uses synchronous sleeps. If the runner later becomes concurrent, cooldown and pause sleeps must be revisited.

For each GAIA task:

1. Invoke the graph normally inside the existing `with ephemeral_memory():` task isolation block.
2. If a `RateLimitCooldownError` escapes:
   - log the task, model, and cooldown duration,
   - sleep for the indicated cooldown,
   - retry the same task once inside a new `with ephemeral_memory():` block.
3. If the retry succeeds, format and checkpoint the answer normally.
4. If the retry still fails with rate-limit exhaustion, append an `AGENT ERROR: RATE LIMITED` answer for this run and continue to the next question.
5. If `QuestionRateLimitStreakError` escapes, append an answer beginning with `AGENT ERROR: RATE LIMITED` and continue to the next question.
6. If `BatchAbortRateLimitError` escapes:
   - do not write a normal success checkpoint for the current task,
   - append an in-memory answer beginning with `AGENT ERROR: RATE LIMITED`,
   - write a non-success diagnostic marker at `<checkpoint_dir>/rate_limit_abort.json`,
   - stop the batch and return answers collected so far.

The `AGENT ERROR:` prefix matches the existing runner convention and ensures any shared checkpoint guard continues to treat the value as non-success.

## Data Flow

```text
GAIA task
  -> runner opens ephemeral_memory()
  -> runner invokes graph
    -> model node calls Gemini lane
      -> _RetryWrapper checks lane cooldown
      -> tenacity retries using is_retryable_rate_limit
      -> each observed 429 updates question scope + batch window
      -> exhausted non-daily 429 records lane cooldown
      -> RateLimitCooldownError bubbles up
  -> runner waits cooldown
  -> runner retries same GAIA task once in a fresh ephemeral_memory()
  -> success checkpoints answer OR failure continues without success checkpoint
```

For daily/RPD exhaustion:

```text
Gemini 429 payload indicates PerDay or retryDelay > 600s
  -> BatchAbortRateLimitError
  -> runner writes diagnostic marker
  -> runner stops batch
```

## Error Handling

- Rate-limit exhaustion should be logged at warning level with provider, model, cooldown, and task id when available.
- Generic graph/model/tool errors should continue through the existing `AGENT ERROR` path.
- A failed retry after cooldown should not crash the whole batch.
- A successful call after cooldown should reset that model lane's consecutive rate-limit count.
- Non-429 `google.genai.errors.ClientError` must not trigger retry, cooldown, question streak, or batch pause.
- Process restart resets in-process cooldown, streak, and sliding-window state by design.

## Testing Plan

Add focused unit tests without hitting external APIs:

1. `is_retryable_rate_limit` returns true for `google.genai.errors.ClientError` with code `429`.
2. `is_retryable_rate_limit` returns false for `google.genai.errors.ClientError` with code `400`.
3. `_RetryWrapper` sleeps when a lane cooldown is active.
4. `_RetryWrapper` records a 60-second cooldown after first exhausted retryable error.
5. Consecutive exhausted rate-limit errors escalate to 120 seconds and then 300 seconds.
6. Successful calls reset the consecutive lane failure counter.
7. Two `_RetryWrapper` instances on the same lane share cooldown.
8. Cooldown on `gemini-3.1-pro` does not delay `gemini-3-flash-preview`.
9. Cooldown uses `time.monotonic()` rather than wall-clock time.
10. Daily quota metadata or `retryDelay > 600s` raises `BatchAbortRateLimitError`.
11. `run_agent_on_questions` retries the same task once after `RateLimitCooldownError`.
12. `run_agent_on_questions` opens a fresh `ephemeral_memory()` block for the retry.
13. `run_agent_on_questions` does not write a normal success checkpoint when both attempts are rate-limited.
14. A 50-event per-question 429 streak skips the current question.
15. A 100-outcome sliding window with at least 70 rate-limited outcomes triggers a batch pause.
16. Unknown/non-Google models do not receive Gemini-specific cooldown profiles.

Tests should monkeypatch sleep/time functions so the suite does not actually wait.

## Risks and Mitigations

- **Risk: sleeping in tests slows the suite.**
  - Mitigation: inject or monkeypatch sleep/time functions in tests.

- **Risk: synchronous sleep blocks future concurrent runners.**
  - Mitigation: document sequential runner assumption; current `run_agent_on_questions` loops tasks sequentially.

- **Risk: cooldown is only in-process.**
  - Mitigation: acceptable for current local batch runs; process restart resetting cooldown is a deliberate non-goal.

- **Risk: same-question retry duplicates some work.**
  - Mitigation: retry only once and use a fresh ephemeral memory store, matching existing per-task isolation.

- **Risk: Gemini error payloads vary.**
  - Mitigation: classify 429s by `exc.code` first; parse quota metadata defensively; fall back to short cooldown only when daily quota cannot be detected.

## Implementation Boundaries

Expected files:

- `src/lilith_agent/models.py`
- `src/lilith_agent/runner.py`
- `tests/test_models.py`
- `tests/test_runner.py` or a new focused runner test file

Do not modify:

- `memory.py`
- `app.py` web-search dedup logic
- embedding dependencies or schema
- unrelated untracked files
