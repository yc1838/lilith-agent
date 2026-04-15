# Lilith Agent — Architecture

Three views: **system** (entry → graph → outputs), **ReAct graph** (state machine), **tool belt** (tool taxonomy + dependencies).

## System overview

```mermaid
flowchart TB
    subgraph Entry["Entry points"]
        TUI["tui.py<br/>(lilith CLI)"]
        CLI["scripts/dev_run_gaia.py<br/>(batch runner)"]
        APP["app.py<br/>(Gradio · HF Space)"]
    end

    subgraph Config["Config & models"]
        CFG["config.py<br/>Config.from_env()"]
        MDL["models.py<br/>cheap / strong / extra_strong<br/>+ NoThink / Retry wrappers"]
        ENV[".env<br/>API keys · provider+model · caveman"]
        ENV --> CFG --> MDL
    end

    subgraph Core["ReAct core"]
        BUILD["app.py :: build_react_agent(cfg)"]
        GRAPH["LangGraph<br/>StateGraph(AgentState)"]
        RUN["runner.py<br/>run_agent_on_questions()"]
        BUILD --> GRAPH
        RUN --> GRAPH
    end

    subgraph Data["GAIA data sources"]
        DS["gaia_dataset.py<br/>GaiaDatasetClient<br/>(HF dataset)"]
        SCORE["ScoringApiClient<br/>(agents-course-unit4-<br/>scoring.hf.space)"]
    end

    subgraph Obs["Observability"]
        LOG[".lilith/session-*.log"]
        TRACE[".lilith/session-*.jsonl<br/>JsonlTraceCallback"]
        ARIZE["Arize AX<br/>(optional)"]
        LS["LangSmith<br/>(optional)"]
    end

    TUI --> BUILD
    TUI --> Obs
    CLI --> BUILD
    CLI --> RUN
    CLI --> DS
    APP --> BUILD
    APP --> RUN
    APP --> SCORE

    MDL --> BUILD
    GRAPH --> Obs

    CKPT[".checkpoints/<task_id>.json"]
    RUN --> CKPT
```

## ReAct graph (state machine)

```mermaid
stateDiagram-v2
    [*] --> model
    model --> tools: AIMessage has tool_calls<br/>AND iterations &lt; limit<br/>AND calls &lt; BUDGET_HARD_CAP (25)
    model --> fail_safe: iterations ≥ recursion_limit − 2<br/>OR calls ≥ BUDGET_HARD_CAP
    model --> [*]: no tool_calls (final answer)
    tools --> model: results appended as ToolMessages
    fail_safe --> [*]: emergency summary
    note right of model
        compact old ToolMessages
        (keep last 4 verbatim,
        truncate older to 300 chars)
        inject BUDGET WARNING at 15 calls
        apply caveman prompt wrapper
    end note
    note left of tools
        per-call guards:
        1. exact (name,args) dedup
        2. semantic dedup (Jaccard ≥ 0.5) for tavily
        3. per-tool error cooldown (3 fails → freeze)
    end note
```

## Tool belt

```mermaid
flowchart LR
    subgraph Web["Web & search"]
        WS["web_search<br/>(DDG → Tavily fallback)"]
        FU["fetch_url<br/>(trafilatura)"]
    end

    subgraph Code["Code & files"]
        RP["run_python<br/>(sandboxed subprocess)"]
        RF["read_file"]
        LS_T["ls · grep · glob_files · write_file"]
    end

    subgraph Media["Media"]
        AUD["transcribe_audio<br/>(faster-whisper)"]
        PDF["inspect_pdf"]
        VIS["inspect_visual_content<br/>(Gemini → FAL → cross-provider)"]
        YT_T["youtube_transcript"]
        YT_F["youtube_frame_at<br/>(ffmpeg)"]
    end

    subgraph Academic["Academic"]
        AX["arxiv_search"]
        CR["crossref_search"]
        CJ["count_journal_articles"]
        FE["filter_entities"]
    end

    subgraph Plan["Planning"]
        WT["write_todos"]
        MD["mark_todo_done"]
    end

    REG["tools/__init__.py<br/>build_tools(cfg)"]
    REG --> Web
    REG --> Code
    REG --> Media
    REG --> Academic
    REG --> Plan

    VIS -. fallback chain .-> VIS
    YT_F --> VIS
    CR --> FE
```

## Key sources

| Concern | File |
|---|---|
| ReAct graph + routing + dedup | [src/lilith_agent/app.py](src/lilith_agent/app.py) |
| Batch runner + checkpoints | [src/lilith_agent/runner.py](src/lilith_agent/runner.py) |
| Config loader | [src/lilith_agent/config.py](src/lilith_agent/config.py) |
| Model provider factory | [src/lilith_agent/models.py](src/lilith_agent/models.py) |
| Tool registry | [src/lilith_agent/tools/__init__.py](src/lilith_agent/tools/__init__.py) |
| Logging + Arize + JSONL trace | [src/lilith_agent/observability.py](src/lilith_agent/observability.py) |
| HF dataset client | [src/lilith_agent/gaia_dataset.py](src/lilith_agent/gaia_dataset.py) |
| Gradio Space entry | [app.py](app.py) |
| Interactive TUI | [src/lilith_agent/tui.py](src/lilith_agent/tui.py) |
| GAIA batch CLI | [scripts/dev_run_gaia.py](scripts/dev_run_gaia.py) |
