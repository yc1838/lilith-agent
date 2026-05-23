# A Month with GAIA: How Throwing Away a Working Agent Got Me to 85% on Level 1

I spent a month on the GAIA benchmark. I built two agents. The first one passed the certificate threshold within 48 hours. I deleted it anyway. Then the replacement got stuck for three weeks. Today it scored **85% on Level 1** and **~60% on Level 2** (53/86) on the [students' leaderboard](https://huggingface.co/spaces/agents-course/Students_leaderboard).

This is the story of why throwing away a working thing was the right call, and why a single architectural change at the very end of the month was worth the entire plateau in between.

## Attempt 1: The Multi-Node Graph

My first agent lived for four days. Created April 9, abandoned April 13. The architecture was the kind of thing that looks impressive in a diagram:

```
Question → Perception → Planner → Router → S1/S2 Executors → Verifier → Formatter → Answer
```

Six nodes, each with a single role. A planner that emitted a 3–5 step JSON plan with explicit tier hints. A rule-based router that read each step's `tier` field and dispatched it to a cheap or strong model. A verifier that could reject the answer and loop back to the planner up to twice. A formatter that stripped prose to make the output GAIA-grader friendly.

It scored:
- **April 10: 55% on Level 1** (first scored attempt — already over the 30% certificate threshold)
- **April 11: 60% Level 1, 45% Level 2**

Then I started "improving" it. I tweaked the planner prompt. I added a heuristic to the router. I changed how the verifier interpreted ambiguous answers. Each change felt like a refinement.

Each change made the agent worse. Within a couple of days I had taken Level 1 down to somewhere in the 30s or low 40s. The planner was emitting plans the executor couldn't carry out. The verifier was rejecting correct answers because the formatter had already mangled them. The router was sending hard reasoning to the cheap model and trivial lookups to the strong model. Everything was technically configurable and nothing actually worked.

I sat with it for a day, read some posts about why the ReAct pattern keeps winning over hand-built planning graphs, and decided the architecture itself was the problem. The agent was over-specified. Too many opinions baked into the structure, not enough room for the model to do what it's actually good at.

I deleted the repo. The certificate-passing agent is preserved at [yc1838/huggingface-agents-course-final](https://github.com/yc1838/huggingface-agents-course-final) as a museum piece.

## Attempt 2: Lilith

April 13. New repo. The premise was minimalist: **one graph, one model node, one tools node, one fail-safe**. Pure ReAct. The model decides what to do; the tools node executes; the loop continues until the model emits a final answer or hits a budget cap.

I named it Lilith because I wanted it to feel like an agent, not a script. The Hugging Face Space banner is a butterfly. Pink-to-purple gradient. The TUI prompt is `lilith >`. None of this matters for accuracy. All of it mattered for me showing up.

The graph:

```
              ┌──────────────┐
HumanMessage→ │    model     │ →END (no tool calls)
              └──────┬───────┘
                     │ has tool calls?
                     ▼
              ┌──────────────┐
              │    tools     │ → back to model
              └──────┬───────┘
                     │ recursion / budget hit?
                     ▼
              ┌──────────────┐
              │  fail_safe   │ → END
              └──────────────┘
```

That's it. Three nodes. The model picks tools. The tools node runs them and shoves the results back as `ToolMessage`s. If the model loops, `fail_safe` gives it one emergency override prompt and forces it to commit.

The first version of this got me **back to roughly where the old agent had been** — somewhere around 50%. Then it sat there. For three weeks.

## The Long Plateau

This is the part of the story that doesn't fit on a slide. Most of the work was unglamorous. I added tools, refined error handling, tuned prompts, fixed bugs, and the score barely moved. I'd ship something that felt like a real win and Level 1 would tick up two points and slide back down on the next run.

The tool belt grew to:

- `web_search`, `fetch_url` — primary research
- `run_python` — sandboxed Python with bs4, pandas, trafilatura, pypdf
- `read_file`, `ls`, `grep`, `glob_files`, `write_file` — local filesystem
- `transcribe_audio` — `faster-whisper`
- `youtube_transcript`, `youtube_frame_at` — GAIA has more YouTube questions than you'd expect
- `inspect_pdf` — PDF → text
- `inspect_visual_content` — multimodal vision with a fallback chain
- `arxiv_search`, `crossref_search`, `count_journal_articles`, `filter_entities` — academic
- `search_memory` — query Lilith's long-term memory

There was one rule I held to across all of them: **fail loud, fail recoverable**. Every tool wraps its body in a `try` that catches the exception and returns a `ToolMessage` with `status="error"` rather than raising. This sounds boring. It's the single most important design decision in the agent. It's what lets the model self-correct. Half the times the agent looks broken, it's actually mid-recovery.

Three of the plateau-period changes deserve their own war stories.

## War Story 1: The Sandbox

`run_python` was the first tool I wrote and the most dangerous. The model generates a string, and that string runs as Python. A model with access to the host filesystem could read arbitrary credentials in one tool call. I shipped this naively in the early days and then re-read my own system prompt — which told the model to *be creative, try things* — and realized those instructions were incompatible with shell-equivalent power.

Most of April 17 went to fixing this. Two backends:

**Docker backend** (production):
- `--read-only` root filesystem
- `--cap-drop=ALL` (no Linux capabilities)
- `tmpfs` for scratch space
- Metadata IP (`169.254.169.254`) blocked at the socket layer
- ~1 second container boot per call. Worth it.

**Process backend** (dev fallback):
- Scrubbed environment (no `GAIA_*` keys leak in)
- `setrlimit` for CPU/memory
- 200k-char output cap
- Subprocess with `stdin`-fed code, never `/tmp` files

A switch (`LILITH_SANDBOX=auto|process|docker`) picks at runtime. Cross-boundary I/O routes through `read_file` (in) and `write_file` (out, restricted to `.lilith/scratch`). Files written from inside `run_python` vanish when the call returns. By design.

Two non-decisions, both deliberate:
1. I did *not* sandbox `web_search` or `fetch_url`. They run fixed library code, not LLM strings. SSRF guards on `fetch_url` are the right boundary there.
2. The Docker backend has a known gap: `ctypes`-level metadata-IP bypass is theoretically possible. I documented it in `sandbox/README.md` rather than chase perfect.

This work didn't move the score. It just stopped me from waking up to a deleted home directory.

## War Story 2: The Spiral

April 27. I asked Lilith to find a file in `~/code/Orbit-Daily`. Easy. Maybe ten seconds.

What happened:
- Model called `find_files(name='lilith.txt', root='code/Orbit-Daily')`. Tool error: relative path doesn't exist (the agent's cwd was elsewhere).
- Model dropped the `~` and tried the relative path again. Same error.
- Model escalated to `find_files(root='/')`. Thirty-second disk scan. Timeout.
- Model switched to `run_python` and tried listing the user home from inside the sandbox. But `run_python`'s sandbox has its own cwd, divorced from the host. Nothing.
- Model panicked. Tried `find_files(root='/')` again. Cooldown kicked in: synthetic error told the model to pivot.
- Model didn't pivot.
- `GraphRecursionError(50)`. Question failed.

I'd already shipped `~` expansion in commits `96fa5bb` and `0dc5c75`. The plumbing existed. The model didn't know.

The fix took thirty minutes. I added one line to each filesystem-tool docstring: *"Paths starting with `~` are expanded to the user's home directory."* I added Directive 7 to the system prompt:

> **FILESYSTEM SEARCH STRATEGY.** Filesystem tools operate on the HOST filesystem. The `run_python` sandbox is separate. Use `~/...` for home. If exact-name search fails, BROADEN via `grep` or a shorter name pattern — do NOT escalate to `root='/'`.

That was it. No code change. Pure instruction-surface engineering. I added a test that asserted the docstrings contained the word "tilde." Spirals stopped.

The lesson I keep relearning: **what the model knows about its tools is a feature of the prompt, not the code.** Adding capabilities without telling the model is the most expensive form of dead code.

## War Story 3: Quota as Architecture

By early May, my Gemini bill was a graph that only knew how to go up. Worse: I was hitting daily quota walls on the Pro tier and burning the lane on routine summarization.

The diagnosis: I had one model doing all the work. The strongest tier ran every turn. Every tool result, no matter how trivial, got the heavyweight treatment.

The redesign was a week of work starting May 6:

- **Three model tiers**, each with independent provider+model:
  - `cheap` (Gemini Flash Lite) — tool-result summarization, memory extraction
  - `strong` (Gemini Flash) — default ReAct executor
  - `extra_strong` (Gemini Pro / Claude Sonnet) — late-stage rescue when Flash stalls
- **Shared Gemini cooldown registry** — exponential backoff with full jitter, capped at 60s, applied at the provider level so all tiers share the lane awareness
- **Per-question rate-limit streak tracking** — abort the batch on Gemini daily quota error rather than burn through 50 questions silently failing
- **Compaction with memory** — older `ToolMessage`s above 300 chars get summarized by the cheap model into ≤600-char snippets prefixed `[COMPACTED SUMMARY] ` so subsequent passes skip re-summarization

That `[COMPACTED SUMMARY] ` prefix is a one-character optimization that cut my cheap-model spend by maybe 40%. Without it, every model turn re-summarized the same already-summarized message.

This work didn't move the score either. It just made it sustainable to keep running experiments.

## War Story 4: Memory

Around April 26 I added a three-layer persistent memory architecture, loosely inspired by the [Engram memory model](https://arxiv.org/abs/2501.12599):

| Layer | Storage | Role |
|---|---|---|
| **Short-term** | `.lilith/threads.sqlite` via LangGraph's `SqliteSaver` | Conversation state across restarts within a thread |
| **Long-term semantic** | `.lilith/long_term_memory.sqlite` via LangMem | Facts, names, preferences — extracted, deduplicated, conflict-resolved |
| **Episodic** | same SQLite | Past task trajectories — what failed, what worked, summarized |

Episodic was the surprise. After every batch question, a summarizer reads the trace and writes a 2–3 sentence episode: *"Tried to find paper X by author Y; arxiv_search by title failed; succeeded with crossref_search by author + year."* Future questions that match (via keyword search) get those episodes injected into the system prompt as hints.

For batch GAIA runs each question gets an isolated `MemoryStore(":memory:")` so questions can't contaminate each other. This matters more than I expected — the validation set has near-duplicate questions, and without isolation, an episode from question 14 leaks into question 15 and biases the answer.

Memory moved the score a little. Not as much as I'd hoped.

## The Day Everything Changed

May 10. This morning the agent was sitting at **65% on Level 1, 53% on Level 2**. Better than the predecessor's best, finally, but still not great. The Level 1 questions it was missing were mostly not failures of reasoning — I'd watch the trace and the agent had clearly figured out the answer. Then it would emit a final answer like:

> The answer is **Georgette**.

GAIA grades on exact match. `"The answer is Georgette."` is wrong. `"Georgette"` is right.

Other variants: trailing periods. Units when only the number was asked for. A leading "Final Answer: " from a system prompt instruction the agent had decided to mirror. Capitalization mismatches. Years written as `"2002"` vs `"2002 (published)"`.

The fix was a single architectural addition layered on top of the ReAct loop:

1. **Supervisor review node** — after the model emits a candidate `Final Answer`, a separate node calls the strong model with the original question and the candidate, asking: *Is this concrete? Does it answer the question? Strip the prose.*
2. **Format-strip finalizer** — the supervisor's output is then run through a deterministic normalizer (lowercase, strip articles, strip trailing punctuation, normalize whitespace).
3. **Fail-safe fallback to `supervisor_best_answer`** — if the agent hits the budget cap with no committed answer, the supervisor picks the best candidate from the trace.

Commits `a04db57`, `2d3214d`, and `2741135`. Within four hours. The very next commit message is `b7d4c53 mark!!! got me 85% on level 1!!!`. The exclamation marks are load-bearing.

**Level 1: 65% → 85%. Level 2: 53% → ~60% (53/86).** Both moved. The Level 1 jump was bigger because Level 1's failures were dominated by formatting; Level 2's failures are still dominated by reasoning, but the supervisor node and format-strip finalizer recovered enough borderline answers to lift it 7 points.

## What I'd Tell Past Me

**Don't optimize a working architecture you don't yet trust.** The old agent passed the certificate at 55%, then I tinkered it down to the 30s. Tinkering felt productive. It was destruction. If you don't have strong intuitions about *why* a thing works, your changes will, on average, make it worse. Either invest in the intuitions or stop touching it.

**ReAct is not a baseline. It's a strong default.** I built the multi-node graph because ReAct seemed too simple to be enough. It is enough. The simplicity is what lets the model think. Every node I'd added to the predecessor was constraining the model's degrees of freedom in a place I didn't actually understand the constraints.

**Tools fail visibly. Errors are the model's input.** If a tool raises, the model can't see what went wrong. If it returns a `status="error"` `ToolMessage`, the model can read the error and try something else. This single discipline turns half your "the agent is broken" moments into "the agent is debugging itself."

**The instruction surface is the cheapest lever.** The spiral fix was thirty minutes of docstring edits. If your agent is doing something dumb, the first question is: *did I tell it not to?*

**Cooldowns are architecture, not afterthought.** Rate limits are a property of the system, like memory or disk. Design for them on day one. A shared cooldown registry across tiers is not optional once you cross ~50 questions/day.

**Compaction is not truncation.** Truncating the tail of a tool message will, with probability one, cut off the line that contained the answer. Summarize instead. Mark the summary so you don't re-summarize.

**A score that has plateaued for weeks can move 20 points in an afternoon.** Three weeks of plumbing turned out to be load-bearing for a four-hour architectural change at the end. The plateau was not wasted; the supervisor node would have shown a much smaller delta on top of the unstable old version. Most of the climb is invisible until the last step.

**Personality matters for the human, not the model.** Naming the agent Lilith and giving it a butterfly emoji didn't make the agent smarter. It made me show up to debug it for one more day. That's enough.

## What's Next

85% on Level 1 and ~60% on Level 2 is satisfying. The next frontier is the rest of Level 2 — the failures there are not formatting; they're real multi-source synthesis problems. The supervisor review node helps; a *supervisor synthesis* node, one that doesn't just judge the answer but stitches together facts from across the trace, might help more.

The agent is at [github.com/yc1838/lilith-agent](https://github.com/yc1838/lilith-agent). The abandoned predecessor is [here](https://github.com/yc1838/huggingface-agents-course-final), kept up as a reminder that working is not the same as good.
