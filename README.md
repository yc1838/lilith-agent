# Lilith Agent

🦋 A ReAct research assistant built on LangGraph. Lilith plans, calls tools, and answers open-ended research questions from a TUI or a batch runner over the GAIA benchmark.

## Features

- **Explicit ReAct graph** — tool-call dedup, per-tool error feedback, recursion cap, iteration fail-safe
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
GAIA_RECURSION_LIMIT=100
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

Per-question checkpoints land in `.checkpoints/<task_id>.json` — delete one to force a rerun.

## Tools

All tools live under [src/lilith_agent/tools/](src/lilith_agent/tools/) and are registered in [\_\_init\_\_.py](src/lilith_agent/tools/__init__.py):

| Tool | Purpose |
| --- | --- |
| `tavily_search`, `fetch_url` | Primary web search + page fetch |
| `run_python` | Sandboxed Python subprocess (bs4, pandas, trafilatura, pypdf, …) |
| `read_file`, `ls`, `grep`, `glob_files`, `write_file` | Local filesystem |
| `transcribe_audio` | faster-whisper |
| `youtube_transcript` | Spoken-word captions only |
| `youtube_frame_at` | Download + extract one frame at a timestamp (PNG) |
| `inspect_pdf` | PDF → text |
| `inspect_visual_content` | Multimodal vision (Gemini + FAL moondream/llava fallbacks) |
| `arxiv_search`, `crossref_search`, `count_journal_articles`, `filter_entities` | Academic metadata |
| `write_todos`, `mark_todo_done` | High-level planning |

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
  config.py          # Config.from_env(), model + API key + feature flags
  observability.py   # logging, Arize setup, JsonlTraceCallback
  models.py          # provider -> chat model builder
  gaia_dataset.py    # HF GAIA dataset loader
  tools/             # LangChain @tool wrappers + impls
scripts/
  dev_run_gaia.py    # CLI to run against real GAIA questions
.checkpoints/        # per-question answers (gitignored)
.lilith/             # session logs + JSONL traces (gitignored)
```

## Testing

```bash
pytest
```
