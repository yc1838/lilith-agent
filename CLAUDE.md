# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[test]"          # editable install + test extras
lilith                             # interactive TUI (entry: lilith_agent.tui:main)
python -m lilith_agent.tui         # equivalent
pytest                             # full suite (pyproject pins src/ on pythonpath)
pytest tests/test_graph.py -k router  # single test by name
python scripts/dev_run_gaia.py --limit 3 --level 1
python scripts/dev_run_gaia.py --task-id <uuid>           # rerun one question
python scripts/dev_run_gaia.py --split test --level 1 --limit -1 --cavemen --caveman-mode ultra
python scripts/build_leaderboard_submission.py --split test --out submission.jsonl
docker build -t lilith-pysandbox:latest sandbox/          # required before LILITH_SANDBOX=docker
```

Per-question state lives in `.checkpoints/<task_id>.json` — delete one to force a rerun. Session logs/traces land in `.lilith/session-<ts>.{log,jsonl}`. Both directories are gitignored.

Two `app.py` files exist: top-level `app.py` is the **Gradio Space** entry; `src/lilith_agent/app.py` is the **ReAct graph**. Don't confuse them.

## Architecture

### ReAct graph (`src/lilith_agent/app.py`)

`build_react_agent(cfg)` returns a compiled LangGraph `StateGraph(AgentState)` with three nodes — `model`, `tools`, `fail_safe` — and a `MemorySaver` checkpointer. The graph is the single source of truth for control flow; do not bypass it.

State machine:
- `model` → `tools` when last `AIMessage` has `tool_calls` AND iterations < `recursion_limit-2` AND tool calls since last `HumanMessage` < `budget_hard_cap` (default 25).
- `model` → `fail_safe` when either ceiling is hit. `fail_safe_node` invokes the model once with an emergency override prompt and ends.
- `model` → `END` when no tool calls (final answer).
- `tools` → `model` with appended `ToolMessage`s (one per requested call).

The `tools` node (`_build_tool_node`) layers three independent guards before invoking a tool — keep them in this order if editing:
1. **Exact dedup**: `(tool_name, sorted-json args)` matches any prior `AIMessage` tool call → synthetic error `ToolMessage`, no invocation.
2. **Semantic dedup** (`web_search` only): Jaccard similarity ≥ `cfg.semantic_dedup_threshold` (default 0.5, tokens normalized + stopwords stripped) against any prior `web_search` query in the current turn → synthetic error.
3. **Per-tool error cooldown**: 3 contiguous error `ToolMessage`s for one tool name → "stalled" error pushing the model to pivot strategy.

Tool exceptions are caught and surfaced as `status="error"` `ToolMessage`s (never raised). This is what lets the model self-correct — preserve it.

The `model` node always runs three pre-invocation transforms in order:
1. `apply_caveman(base_prompt, cfg.caveman, cfg.caveman_mode)` wraps the system prompt with caveman framing when enabled.
2. `_compact_old_tool_messages` — keep last 4 `ToolMessage`s verbatim, compact older ones >300 chars. If `cfg.compact_summarize` is on, the cheap model summarizes (target ≤600 chars, prefixed `[COMPACTED SUMMARY] ` so subsequent passes skip re-summarizing). Otherwise head-truncate with a `[COMPACTED: N chars dropped]` marker.
3. Goal re-injection at ≥5 calls and `[BUDGET WARNING]` at `cfg.budget_warn_at` calls — both as ephemeral `SystemMessage`s prepended each turn (not stored in state).

Untrusted-input boundary: the user's first `HumanMessage` is wrapped in `<gaia_question>...</gaia_question>` by upstream callers. The model node strips this delimiter when extracting the goal. The system prompt instructs the model to treat anything inside as **data, not instructions**. Do not weaken this when refactoring prompts.

### Config (`src/lilith_agent/config.py`)

`Config.from_env()` reads `GAIA_*` env vars (note: `GAIA_` prefix, not `LILITH_`). Three model tiers: `cheap` / `strong` / `extra_strong`, each with independent provider+model. The agent's main model is **`extra_strong`**; `cheap` powers the tool-result summarizer. `extra_strong_*` defaults to `strong_*` if unset. The `vision_*` pair is separate, used by `inspect_visual_content`. Behavior flags: `caveman`, `caveman_mode`, `recursion_limit` (50), `budget_hard_cap` (25), `budget_warn_at` (15), `semantic_dedup_threshold` (0.5), `compact_summarize`, `llm_formatter_enabled`.

### Tools (`src/lilith_agent/tools/`)

`build_tools(cfg)` in `tools/__init__.py` returns the registered list. Tools are LangChain `@tool`-decorated closures; the closure injects `cfg` so individual tool modules stay pure functions.

`run_python` (`tools/python_exec.py`) runs in an isolated sandbox with **no host filesystem access**. Selection via `LILITH_SANDBOX`: `auto` (default — docker if available, else process), `process`, `docker`. Cross-boundary I/O must route through `read_file` (in) and `write_file` (out, restricted to `.lilith/scratch`). Files written from inside `run_python` vanish when the call returns. The Docker backend requires `lilith-pysandbox:latest` to be built (`sandbox/Dockerfile`); see `sandbox/README.md` for the isolation matrix and known gaps (e.g., `ctypes`-level metadata-IP bypass).

`inspect_visual_content` has a fallback chain: configured provider+model → same-provider stable fallback → cross-provider last-resort (`gemini-3-flash-preview` on Google). All-fail trips a **session-level circuit breaker** — future calls return a clean error rather than retry. Don't add retry loops above this; the breaker is intentional.

### Observability (`src/lilith_agent/observability.py`)

Two logger trees configured by `setup_observability()`:
- `lilith_agent.app` — routing/compaction
- `lilith_agent.nodes.{model,tools,fail_safe}` — per-node traces (chosen so output mirrors the GAIA reference agent's bracketed-tag style)

`JsonlTraceCallback` writes one JSON event per line to `.lilith/session-*.jsonl` flushed per write — replay-able. Arize AX auto-enables when `ARIZE_SPACE_ID` + `ARIZE_API_KEY` are set; LangSmith via `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` + `LANGCHAIN_PROJECT`. Token-usage metadata in `response_metadata` is preserved (only `safety_ratings`, `logprobs`, Gemini thought signatures are stripped — keep this allowlist when touching `_strip_response_metadata_noise`, since Arize cost reporting depends on `input_tokens`/`output_tokens`).

### Entry points

- `tui.py` → interactive REPL with slash commands (`/clear`, `/caveman [on|off|lite|full|ultra]`).
- `runner.py::run_agent_on_questions` → batch over GAIA, writes `.checkpoints/<task_id>.json` per question; reruns skip existing checkpoints unless `--force`.
- `app.py` (top-level) → Gradio Space; calls `build_react_agent` + `run_agent_on_questions` + `ScoringApiClient` (`agents-course-unit4-scoring.hf.space`).
- `gaia_dataset.py` → HF dataset client (`GaiaDatasetClient`), needs `GAIA_HUGGINGFACE_API_KEY`.

## Repo-specific notes

- Python ≥3.11 required (`pyproject.toml`).
- `requirements.txt` is the **pinned snapshot for the HF Space** (includes `gradio`, `datasets`, `requests` not in `pyproject` deps); `pyproject.toml` is the editable-install spec. Keep both in sync when adding deps that the runtime touches.
- `ffmpeg` on PATH is preferred for `youtube_frame_at`; falls back to `imageio-ffmpeg`.
- `.last_failures.txt` is written by the GAIA runner — useful for `--force` reruns of just the failed task IDs.
- Never weaken the `<gaia_question>` untrusted-input wrapper, the per-tool cooldown, or the run_python FS boundary without explicit instruction — these are intentional hardening, not artifacts.

## Agent Behavioral Guidelines (Karpathy Style)

- **Think Before Coding**: Don't assume. Surface tradeoffs. If multiple interpretations exist, present them. If unclear, stop and ask.
- **Simplicity First**: Write the minimum code that solves the problem. No speculative features, no "just-in-case" abstractions or "flexibility" that wasn't requested.
- **Surgical Changes**: Touch only what you must. Don't "improve" adjacent code, comments, or formatting. Don't refactor things that aren't broken. Match existing style. Every changed line should trace directly to the requested task. Clean up only your own orphans (imports/variables).
- **Goal-Driven Execution**: Define success criteria. Loop until verified. Transform tasks into verifiable goals (e.g., "Write a test that reproduces the bug, then make it pass").
- **Engineering Realism (March of Nines)**: Prioritize deployment reliability and tail-end behavior over happy-path demos. Acknowledge that the leap from 90% to 99.9% is non-linear.
