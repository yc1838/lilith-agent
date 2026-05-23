---
title: Lilith Agent
emoji: 🦋
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# Lilith Agent

🦋 A ReAct research assistant built on LangGraph. Lilith plans, calls tools, and answers open-ended research questions from a TUI or a batch runner over the GAIA benchmark.

Currently 85% on gaig level 1 benchmark: https://huggingface.co/spaces/agents-course/Students_leaderboard


## Features

- **Explicit ReAct graph** — tool-call dedup, per-tool error feedback, recursion cap, iteration fail-safe
- **Three-layer persistent memory** — short-term thread checkpoints, long-term semantic facts (LangMem), episodic task experiences; inspired by the Engram memory architecture
- **Tool belt** — web search, URL fetch, sandboxed Python, file I/O, PDF, audio/video transcription, YouTube frame extraction, vision (Gemini + FAL fallbacks), arXiv, CrossRef, todos
- **Multi-provider routing** — cheap / strong / extra-strong model tiers with independent provider+model config
- **Observability** — per-session JSONL trace + rotating log file, optional Arize AX + LangSmith tracing
- **Caveman mode** — compresses the system prompt so the model responds tersely (lite / full / ultra)

## Install

```bash
pip install -e .
# or for a pinned snapshot:
pip install -r requirements.txt
```

Also need `ffmpeg` on PATH for YouTube frame extraction. If missing, `imageio-ffmpeg` (bundled via deps) is used as a fallback.

## Configure

Copy `.env.example` (or create `.env`) with at least:

```bash
GAIA_ANTHROPIC_API_KEY=sk-ant-...
GAIA_GOOGLE_API_KEY=...
GAIA_TAVILY_API_KEY=tvly-...
GAIA_FAL_VISION_API_KEY=fal-...        # optional, for FAL vision
GAIA_HUGGINGFACE_API_KEY=hf_...        # optional, for GAIA dataset

# Optional tracing
ARIZE_SPACE_ID=...
ARIZE_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT="Lilith Agent"
```

Model routing (all optional, shown with defaults):

```bash
GAIA_CHEAP_PROVIDER=google
GAIA_CHEAP_MODEL=gemini-3-flash-preview
GAIA_STRONG_PROVIDER=anthropic
GAIA_STRONG_MODEL=claude-sonnet-4-6
GAIA_EXTRA_STRONG_PROVIDER=anthropic
GAIA_EXTRA_STRONG_MODEL=claude-sonnet-4-6
GAIA_VISION_PROVIDER=fal
GAIA_VISION_MODEL=gemini-3-flash-preview
GAIA_CAVEMAN=true
GAIA_CAVEMAN_MODE=full
GAIA_RECURSION_LIMIT=50
GAIA_BUDGET_HARD_CAP=25
GAIA_BUDGET_WARN_AT=15
GAIA_SEMANTIC_DEDUP_THRESHOLD=0.5
```

## Run

### Interactive TUI

```bash
lilith
# or
python -m lilith_agent.tui
```

The TUI prints the logo, caveman status, and the trace file path. Type your question at the `lilith 🦋 >` prompt.

**Slash commands**:

| Command | Effect |
| --- | --- |
| `/clear` | Wipe conversation memory, start a new thread |
| `/memory list` | Show all stored facts and recent episodic experiences |
| `/memory forget <id>` | Delete a fact by ID prefix |
| `/memory reflect` | Manually trigger long-term memory extraction for the current thread |
| `/caveman` | Toggle caveman on/off |
| `/caveman off` / `/caveman on` | Explicit on/off |
| `/caveman lite` | Lightest — keep articles & full sentences, cut fluff |
| `/caveman full` | Default — drop articles, fragments OK (classic caveman) |
| `/caveman ultra` | Heaviest — abbreviations, arrows for causality |
| `exit` / `quit` | Leave |

### Batch run over GAIA

```bash
python scripts/dev_run_gaia.py --limit 3 --level 1
python scripts/dev_run_gaia.py --task-id c61d22de-5f6c-4958-a7f6-5e9707bd3466
```

```bash
# Runs all level-one test questions with caveman mode. Rerun without --force to resume.
python scripts/dev_run_gaia.py --split test --level 1 --limit -1 --cavemen --caveman-mode ultra
```

```bash
# Without caveman mode — set GAIA_CAVEMAN=false in .env beforehand.
python scripts/dev_run_gaia.py --split test --level 1 --limit 5
```

### Build a leaderboard submission

```bash
python scripts/build_leaderboard_submission.py --split test --out submission.jsonl --pad-missing
# Upload submission.jsonl to https://huggingface.co/spaces/gaia-benchmark/leaderboard/submit
```

Per-question checkpoints land in `.checkpoints/<task_id>.json`. Reruns skip existing checkpoints by default. To overwrite fresh answers, use `--force` on the selected scope:

```bash
# Rerun and overwrite one task.
python scripts/dev_run_gaia.py --split test --task-id <task_id> --force

# Rerun and overwrite all level-one test tasks.
python scripts/dev_run_gaia.py --split test --level 1 --limit -1 --force
```

After any rerun, rebuild `submission.jsonl`; the builder reads the latest checkpoint files. The GAIA leaderboard expects the full test split: 93 level-1 rows, 159 level-2 rows, and 49 level-3 rows. Use `--pad-missing` so unanswered tasks are emitted as blank placeholders and the file has the required 301 rows:

```bash
python scripts/build_leaderboard_submission.py \
  --checkpoint-dir .checkpoints \
  --split test \
  --out submission.jsonl \
  --pad-missing

wc -l submission.jsonl  # should print 301
```

## Tools

All tools live under [src/lilith_agent/tools/](src/lilith_agent/tools/) and are registered in [\_\_init\_\_.py](src/lilith_agent/tools/__init__.py):

| Tool | Purpose |
| --- | --- |
| `web_search`, `fetch_url` | Primary web search + page fetch |
| `run_python` | Sandboxed Python subprocess (bs4, pandas, trafilatura, pypdf, …) |
| `read_file`, `ls`, `grep`, `glob_files`, `write_file` | Local filesystem |
| `transcribe_audio` | faster-whisper |
| `youtube_transcript` | Spoken-word captions only |
| `youtube_frame_at` | Download + extract one frame at a timestamp (PNG) |
| `inspect_pdf` | PDF → text |
| `inspect_visual_content` | Multimodal vision (Gemini + FAL moondream/llava fallbacks) |
| `arxiv_search`, `crossref_search`, `count_journal_articles`, `filter_entities` | Academic metadata |
| `todos` | High-level planning |
| `search_memory` | Query Lilith's long-term memory (facts + episodes) by keyword |

### Vision fallback chain

`inspect_visual_content` tries in order: configured provider+model → same-provider stable fallback → cross-provider last-resort (`gemini-3-flash-preview` on Google). If **all** fail, it trips a session-level circuit breaker so future calls return a clean error message instead of looping.

## Observability

- **Logs**: `.lilith/session-<timestamp>.log` (WARNING+ to stderr, INFO+ to file)
- **Trace**: `.lilith/session-<timestamp>.jsonl` — full LLM/tool/chain events, flushed per line, replay-able
- **Arize AX**: auto-enabled when `ARIZE_SPACE_ID` + `ARIZE_API_KEY` are set
- **LangSmith**: set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` + `LANGCHAIN_PROJECT`

## Project layout

```text
src/lilith_agent/
  app.py             # ReAct graph, model routing, caveman prompt wrapping
  tui.py             # interactive loop, slash commands, rich output
  runner.py          # batch runner over GAIA questions
  memory.py          # three-layer memory: checkpoints, semantic facts, episodic
  config.py          # Config.from_env(), model + API key + feature flags
  observability.py   # logging, Arize setup, JsonlTraceCallback
  models.py          # provider -> chat model builder
  gaia_dataset.py    # HF GAIA dataset loader
  tools/             # LangChain @tool wrappers + impls
scripts/
  dev_run_gaia.py    # CLI to run against real GAIA questions
.checkpoints/        # per-question answers (gitignored)
.lilith/             # session logs, JSONL traces, long_term_memory.sqlite (gitignored)
```

## Memory system

Lilith uses a three-layer persistent memory architecture loosely inspired by the [Engram memory model](https://arxiv.org/abs/2501.12599):

| Layer | Storage | Role |
| --- | --- | --- |
| **Short-term** (thread checkpoints) | `.lilith/threads.sqlite` via LangGraph `SqliteSaver` | Preserves full conversation state across restarts within a thread |
| **Long-term semantic** (facts) | `.lilith/long_term_memory.sqlite` | Extracts and deduplicates user preferences, names, project details using LangMem; injected into the system prompt on new queries |
| **Episodic** (task experiences) | `.lilith/long_term_memory.sqlite` | Summarises past tool trajectories — what failed, what worked — so Lilith avoids repeating mistakes |

The semantic layer uses LangMem as a memory governance engine (extraction, conflict resolution, forgetting) while SQLite provides local, auditable, migratable persistence. Long-term memory extraction runs automatically after each conversation and can be triggered manually with `/memory reflect`. The agent can also call `search_memory` during reasoning when the system-prompt injection has been truncated.

For batch GAIA runs each question gets an isolated ephemeral memory store (`MemoryStore(":memory:")`) so questions cannot contaminate each other.

## Testing

```bash
pytest
```

Memory tests can use the `ephemeral_memory()` context manager for isolated in-memory stores:

```python
from lilith_agent.memory import ephemeral_memory

def test_something():
    with ephemeral_memory() as store:
        store.add_episode("task", "summary", "success")
        assert len(store.get_recent_episodes()) == 1
    # store discarded on exit, no disk writes
```
