from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

log = logging.getLogger(__name__)
log_runner = logging.getLogger("lilith_agent.nodes.runner")

try:
    from lilith_agent.tools.vision import reset_vision_state
except ImportError:
    def reset_vision_state(): pass


_TRACE_TOOL_OUTPUT_MAX = 400  # chars per tool result kept in reasoning_trace
_TRACE_AI_TEXT_MAX = 800      # chars per AI message text kept in reasoning_trace

# Deterministic formatter: strip only obvious wrapping. No language-level
# rewrites — unit conversion, filler removal, scene-descriptor stripping all
# live in the LLM formatter (see _final_formatting_cleanup), because a regex
# cannot safely tell apart "Mr." / "U.S." / "INT. OFFICE - DAY" from trailing
# filler.
_PREFIX_PATTERNS = (
    re.compile(r"^\s*final\s+answer\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE),
)
# Matches the LAST `Final Answer:` marker anywhere in the text and captures
# everything after it. Used to rescue answers where the model produced a
# verbose preamble followed by the canonical tail, e.g.
#   "...reasoning paragraph...\n\nFinal Answer: 142"
# The cheap-LLM formatter used to do this unreliably (sometimes dropping the
# tail and keeping the preamble). A literal-marker extraction is safe because
# the model only uses that phrase when it means "this is the bare answer."
_FINAL_ANSWER_TAIL = re.compile(r"(?is).*\bfinal\s+answer\s*:\s*(.+?)\s*$")
_WRAPPERS = ("**", '"', "'", "`")

_FILLER_PHRASES = (
    "the answer is",
    "based on",
    "i found",
    "i believe",
    "i think",
    "approximately",
    "my calculation",
    "in conclusion",
    "to summarize",
)
_LLM_FORMATTER_LEN_GATE = 40


def _wrap_user_question(text: str) -> str:
    """Wrap untrusted user/benchmark text in an XML-style delimiter.

    Scrubs inner `<gaia_question>` / `</gaia_question>` occurrences so an
    adversarial question can't close the wrapper early and inject a fake
    system section. Paired with a system-prompt assertion that the model
    should treat only text inside the single outer tag pair as the task.
    """
    safe = text.replace("</gaia_question>", "&lt;/gaia_question&gt;")
    safe = safe.replace("<gaia_question>", "&lt;gaia_question&gt;")
    return f"<gaia_question>\n{safe}\n</gaia_question>"


def _strip_symmetric_wrap(s: str) -> str:
    """Strip matched wrapping (only when both ends match and inner is clean)."""
    for w in _WRAPPERS:
        if len(s) >= 2 * len(w) and s.startswith(w) and s.endswith(w):
            inner = s[len(w): -len(w)]
            if w not in inner:
                return inner
    return s


def _deterministic_format(raw: str) -> str:
    """Safe pre-pass: strip prefixes and symmetric wrappers; never mutate content."""
    s = raw.strip()
    for _ in range(3):
        before = s
        s = _strip_symmetric_wrap(s).strip()
        for pat in _PREFIX_PATTERNS:
            s = pat.sub("", s, count=1).strip()
        m = _FINAL_ANSWER_TAIL.match(s)
        if m:
            tail = m.group(1).strip()
            if tail and tail != s:
                s = tail
        if s == before:
            break
    return s


def _needs_llm_formatter(s: str) -> bool:
    """Gate: short + no filler → already clean, skip the LLM call."""
    if len(s) >= _LLM_FORMATTER_LEN_GATE:
        return True
    lower = s.lower()
    return any(p in lower for p in _FILLER_PHRASES)


def _write_checkpoint_atomic(path: Path, data: dict) -> None:
    """Serialize first, then rename. A crash mid-serialize leaves the prior file intact.

    Why: resume logic silently skips checkpoints that fail to parse, so a truncated
    JSON from an interrupted write would cause a failed question to "succeed" as
    blank. os.replace is atomic within a filesystem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, path)


def _render_reasoning_trace(messages: list) -> str:
    """Render a compact human-readable trace of the agent's steps for leaderboard submission."""
    lines: list[str] = []
    step = 0
    for m in messages:
        if isinstance(m, AIMessage):
            text = getattr(m, "content", "")
            if isinstance(text, list):
                text = "".join(c.get("text", "") for c in text if isinstance(c, dict) and c.get("type") == "text")
            text = (text or "").strip()
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    step += 1
                    name = tc.get("name", "?")
                    args = tc.get("args") or {}
                    try:
                        args_str = json.dumps(args, ensure_ascii=False, default=str)
                    except Exception:
                        args_str = repr(args)
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "…"
                    lines.append(f"Step {step} [tool] {name}({args_str})")
            if text:
                if len(text) > _TRACE_AI_TEXT_MAX:
                    text = text[:_TRACE_AI_TEXT_MAX] + "…"
                lines.append(f"[think] {text}")
        elif isinstance(m, ToolMessage):
            out = str(getattr(m, "content", ""))
            if len(out) > _TRACE_TOOL_OUTPUT_MAX:
                out = out[:_TRACE_TOOL_OUTPUT_MAX] + f"…[+{len(out)-_TRACE_TOOL_OUTPUT_MAX} chars]"
            lines.append(f"[result {getattr(m, 'name', '?')}] {out}")
    return "\n".join(lines)


def run_agent_on_questions(graph: Any, questions: list[dict], checkpoint_dir: str | Path, client: Any = None) -> list[dict]:
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    answers: list[dict] = []

    from lilith_agent.config import Config
    from lilith_agent.models import get_cheap_model
    cfg = Config.from_env()
    cheap_model = get_cheap_model(cfg)

    total = len(questions)
    for idx, question in enumerate(questions, start=1):
        reset_vision_state()
        task_id = question.get("task_id")
        prompt = question.get("question")
        if not task_id or not prompt:
            continue

        file_name = question.get("file_name")
        if file_name and client:
            file_path = client.download_file(task_id, dest_dir=checkpoint_root / "files")
            if file_path:
                prompt += f"\n\n[Attached File Path: {file_path.absolute()}]"

        # Scoring rules removed from here to reduce per-turn context bloat.
        # They are now applied in a final post-processing step.

        checkpoint_path = checkpoint_root / f"{task_id}.json"
        if checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text())
                log_runner.info("[runner] task=%s (%d/%d) skipped (checkpoint exists)", task_id, idx, total)
                answers.append(
                    {
                        "task_id": task_id,
                        "submitted_answer": _submitted_answer_from_checkpoint(checkpoint),
                    }
                )
                continue
            except Exception:
                pass

        log_runner.info(
            "[runner] task=%s (%d/%d) starting q=%r",
            task_id, idx, total, (prompt[:160] + "…") if len(prompt) > 160 else prompt,
        )

        state = {
            "messages": [HumanMessage(content=_wrap_user_question(prompt))],
            "iterations": 0
        }

        try:
            result = graph.invoke(state, {"configurable": {"thread_id": task_id}})
        except Exception as exc:
            log_runner.warning("[runner] task=%s agent error: %s", task_id, exc)
            answers.append(
                {
                    "task_id": task_id,
                    "submitted_answer": f"AGENT ERROR: {exc}",
                }
            )
            continue

        last_m = result["messages"][-1]
        raw_content = getattr(last_m, "content", "")
        if isinstance(raw_content, list):
            submitted_answer = "".join([c.get("text", "") for c in raw_content if isinstance(c, dict) and c.get("type") == "text"])
        else:
            submitted_answer = str(raw_content)
        
        submitted_answer = submitted_answer.strip()
        
        submitted_answer = _final_formatting_cleanup(
            cheap_model,
            prompt,
            submitted_answer,
            llm_formatter_enabled=cfg.llm_formatter_enabled,
        )
        
        reasoning_trace = _render_reasoning_trace(result["messages"])

        checkpoint = {
            "task_id": task_id,
            "question": prompt,
            "submitted_answer": submitted_answer,
            "reasoning_trace": reasoning_trace,
        }
        
        if submitted_answer and not submitted_answer.startswith("AGENT ERROR"):
            _write_checkpoint_atomic(checkpoint_path, checkpoint)

        log_runner.info(
            "[runner] task=%s (%d/%d) answer=%r",
            task_id, idx, total,
            (submitted_answer[:160] + "…") if len(submitted_answer) > 160 else submitted_answer,
        )
        answers.append({"task_id": task_id, "submitted_answer": submitted_answer.strip()})

    return answers


def _final_formatting_cleanup(
    model: Any,
    question: str,
    raw_answer: str,
    *,
    llm_formatter_enabled: bool = True,
) -> str:
    """Two-stage post-processor: safe deterministic strip, then LLM fallback when needed.

    The deterministic pass always runs (cheap, rule-based, no semantic rewrites).
    The LLM pass runs only when (a) `llm_formatter_enabled` is True AND (b) the
    deterministic output still looks unstructured (long or contains filler phrases).
    Short, clean answers skip the LLM entirely, which avoids the known failure mode
    where the cheap model mutates a correct value (drops a digit, reinterprets units).
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    determ = _deterministic_format(raw_answer)

    if not llm_formatter_enabled:
        log.info("formatter: deterministic-only (flag disabled), len=%d", len(determ))
        return determ

    if not _needs_llm_formatter(determ):
        log.info("formatter: deterministic-only (gate bypass), len=%d", len(determ))
        return determ

    log.info("formatter: invoking LLM, in_len=%d", len(determ))

    instructions = (
        "You are a benchmark scoring assistant. Your task is to extract the EXACT answer "
        "from a researcher's conclusion based on strict formatting rules.\n\n"
        "SCORING RULES:\n"
        "1. Remove all conversational filler (e.g., 'The answer is...', 'Based on...', 'I found...').\n"
        "2. Strip all reasoning, context, and explanations. Output ONLY the core value.\n"
        "3. If the answer is a location, remove scene descriptors (INT., EXT., - DAY).\n"
        "4. Strip all trailing punctuation (., !).\n"
        "5. Honor requested units (e.g., if asked 'how many thousands', '3000' becomes '3').\n"
        "6. Output ONLY the bare text of the answer. No intro, no outro."
    )

    prompt = (
        f"Original Question: {question}\n\n"
        f"Researcher's Conclusion: {determ}\n\n"
        "Format the 'Researcher's Conclusion' into the bare answer required by the rules above."
    )

    try:
        resp = model.invoke([SystemMessage(content=instructions), HumanMessage(content=prompt)])
        content = resp.content
        if isinstance(content, list):
            cleaned = "".join([c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"])
        else:
            cleaned = str(content)

        cleaned = cleaned.strip()
        if not cleaned:
            log.warning("formatter: LLM returned empty, falling back to deterministic")
            return determ
        log.info("formatter: LLM returned out_len=%d", len(cleaned))
        return cleaned
    except Exception as e:
        log.warning("formatter: LLM call failed (%s), falling back to deterministic", e)
        return determ


def _submitted_answer_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    return checkpoint.get("submitted_answer") or checkpoint.get("final_answer", "")
