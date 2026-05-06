---
title: Gemini Rate-Limit Cooldown Design
date: 2026-05-06
status: approved-for-planning
---

# Gemini Rate-Limit Cooldown Design

## Context

Lilith currently wraps chat model calls in `_RetryWrapper` with tenacity exponential backoff for provider rate-limit exceptions. This protects a single model call, but it does not coordinate across calls or GAIA tasks. If Gemini quota pressure persists after retries are exhausted, the runner can move to the next task and immediately repeat the same retry ladder against the same overloaded model lane.

For current usage, only two Google text models need first-class cooldown behavior:

| Model | RPM | TPM | RPD |
| --- | ---: | ---: | ---: |
| `gemini-3-flash-preview` | 2000 | 3,000,000 | 100,000 |
| `gemini-3.1-pro` | 1000 | 5,000,000 | 50,000 |

Gemma, embeddings, image, audio, video, and grounding quotas are out of scope for this patch.

## Goals

1. Avoid hammering Gemini after retry exhaustion.
2. Cool down the specific provider/model lane that is rate-limited.
3. Let GAIA batch runs wait, retry the same question once, then continue if still blocked.
4. Keep existing per-call exponential backoff behavior.
5. Preserve normal non-rate-limit error handling.

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

Unknown models should continue to use the existing retry wrapper without profile-specific behavior. They may still share generic retry handling, but they should not get special cooldown assumptions from this design.

### In-process cooldown registry

Add a small module-level registry in `models.py`:

```text
(provider, model) -> cooldown_until_monotonic_seconds
(provider, model) -> consecutive_rate_limit_exhaustions
```

Before invoking the wrapped model, `_RetryWrapper` checks whether its lane has an active cooldown. If active, it sleeps until the cooldown expires, then proceeds with the existing tenacity retry loop.

After a successful call, the lane's consecutive rate-limit exhaustion count resets to zero.

### Rate-limit exhaustion signal

Introduce a project-level exception such as `RateLimitCooldownError`. `_RetryWrapper` should raise it only when all retry attempts were exhausted due to a recognized retryable rate-limit exception.

The exception should carry enough information for callers and logs:

- provider
- model
- cooldown seconds
- original exception string

Non-rate-limit exceptions should keep their existing behavior and should not be rewritten as cooldown errors.

### Cooldown duration

Because both target models have high RPM, exhausted 429s are likely burst, token, or daily-quota pressure rather than ordinary request spacing. Use a conservative escalating cooldown for repeated failures on the same lane:

| Consecutive exhausted rate limits | Cooldown |
| ---: | ---: |
| 1 | 60 seconds |
| 2 | 120 seconds |
| 3+ | 300 seconds |

If the provider exception exposes an explicit retry delay later, a future patch can prefer that delay. This design does not require parsing provider-specific retry metadata.

### Runner behavior

`runner.py::run_agent_on_questions` should distinguish rate-limit cooldown failures from generic agent errors.

For each GAIA task:

1. Invoke the graph normally.
2. If a `RateLimitCooldownError` escapes:
   - log the task, model, and cooldown duration,
   - sleep for the indicated cooldown,
   - retry the same task once from a fresh ephemeral memory context.
3. If the retry succeeds, format and checkpoint the answer normally.
4. If the retry still fails with rate-limit exhaustion, append an `AGENT ERROR: RATE LIMITED` answer for this run and continue to the next question.
5. Do not write a normal success checkpoint for rate-limited failures.

This keeps reruns possible after quota resets and avoids falsely treating a quota failure as a completed answer.

## Data Flow

```text
GAIA task
  -> runner invokes graph
    -> model node calls Gemini lane
      -> _RetryWrapper checks lane cooldown
      -> tenacity retries per call
      -> exhausted 429 records lane cooldown
      -> RateLimitCooldownError bubbles up
  -> runner waits cooldown
  -> runner retries same GAIA task once
  -> success checkpoints answer OR failure continues without success checkpoint
```

## Error Handling

- Rate-limit exhaustion should be logged at warning level with provider, model, and cooldown.
- Generic graph/model/tool errors should continue through the existing `AGENT ERROR` path.
- A failed retry after cooldown should not crash the whole batch.
- A successful call after cooldown should reset that model lane's consecutive rate-limit count.

## Testing Plan

Add focused unit tests without hitting external APIs:

1. `_RetryWrapper` sleeps when a lane cooldown is active.
2. `_RetryWrapper` records a 60-second cooldown after first exhausted retryable error.
3. Consecutive exhausted rate-limit errors escalate to 120 seconds and then 300 seconds.
4. Successful calls reset the consecutive failure counter.
5. `run_agent_on_questions` retries the same task once after `RateLimitCooldownError`.
6. `run_agent_on_questions` does not write a normal checkpoint when both attempts are rate-limited.
7. Unknown/non-Google models do not receive Gemini-specific cooldown profiles.

## Risks and Mitigations

- **Risk: sleeping in tests slows the suite.**
  - Mitigation: inject or monkeypatch sleep/time functions in tests.

- **Risk: cooldown is only in-process.**
  - Mitigation: acceptable for current local batch runs; cross-process persistence is a non-goal.

- **Risk: same-question retry duplicates some work.**
  - Mitigation: retry only once and use a fresh ephemeral memory store, matching existing per-task isolation.

- **Risk: provider raises a different 429 exception shape.**
  - Mitigation: reuse the existing `RETRY_EXCEPTIONS` list so cooldown behavior follows current retry classification.

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
