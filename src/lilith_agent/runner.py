from __future__ import annotations

import json
import logging
import os
import re
import time
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
_ASSIGNMENT_PREFIX = re.compile(r"^\s*(?:x|y|answer|result)\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
_COMMA_GROUPED_INTEGER = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+$")
_SCALAR_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
# Matches "to N decimal places" or "to the nearest tenth/hundredth/thousandth"
_DECIMAL_PLACES_RE = re.compile(
    r"(?:to|rounded?\s+to|nearest)\s+"
    r"(?:(\d+)\s+decimal\s+place[s]?|the\s+nearest\s+(tenth|hundredth|thousandth|ten[-\s]?thousandth))",
    re.IGNORECASE,
)
_PRECISION_WORDS = {
    "tenth": 1, "hundredth": 2, "thousandth": 3,
    "ten-thousandth": 4, "ten thousandth": 4,
}
def _required_decimal_places(question: str) -> int | None:
    """Return the number of decimal places the question demands, or None."""
    m = _DECIMAL_PLACES_RE.search(question)
    if not m:
        return None
    if m.group(1):
        return int(m.group(1))
    word = m.group(2).lower().replace(" ", "-")
    return _PRECISION_WORDS.get(word)


def _apply_decimal_precision(s: str, places: int) -> str:
    """Reformat a numeric string to exactly `places` decimal places."""
    try:
        value = float(s)
        return f"{value:.{places}f}"
    except (ValueError, OverflowError):
        return s


_ANSWER_CONTRACT_QUESTION_MARKERS = (
    "country", "countries", "capital", "arrival", "time", "meter", "metre",
    "label", "score", "passenger", "title", "author", "date", "year",
    "how many",
)
_GIVE_UP_PHRASES = (
    "unknown",
    "i don't know",
    "cannot determine",
    "could not determine",
    "unable to determine",
    "not enough information",
    "could not complete",
    "why it failed",
)


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


def _is_safe_llm_formatter_output(source: str, cleaned: str) -> bool:
    if not cleaned:
        return False
    if cleaned == source:
        return True
    start = source.find(cleaned)
    if start == -1:
        return False
    end = start + len(cleaned)
    before = source[start - 1] if start > 0 else ""
    after = source[end] if end < len(source) else ""
    if before.isalnum() or before == "_":
        return False
    if after.isalnum() or after == "_":
        return False
    return True


def _expand_to_source_token(source: str, cleaned: str) -> str | None:
    if not cleaned:
        return None
    start = source.find(cleaned)
    if start == -1:
        return None
    end = start + len(cleaned)
    token_start = start
    while token_start > 0 and (source[token_start - 1].isalnum() or source[token_start - 1] == "_"):
        token_start -= 1
    token_end = end
    while token_end < len(source) and (source[token_end].isalnum() or source[token_end] == "_"):
        token_end += 1
    if token_start == start and token_end == end:
        return None
    token = source[token_start:token_end].strip()
    return token or None


def _normalize_gaia_submission(question: str, answer: str) -> str:
    s = _deterministic_format(answer).strip()
    if s.startswith("**"):
        candidate = s.lstrip("*").strip()
        if candidate and "**" not in candidate:
            s = candidate
    if s.endswith("**"):
        candidate = s.rstrip("*").strip()
        if candidate and "**" not in candidate:
            s = candidate

    match = _ASSIGNMENT_PREFIX.match(s)
    if match:
        candidate = match.group(1).strip()
        if _SCALAR_NUMBER.fullmatch(candidate):
            s = candidate

    if _COMMA_GROUPED_INTEGER.fullmatch(s):
        s = s.replace(",", "")

    if ";" in s:
        s = re.sub(r"\s*;\s*", "; ", s).strip()

    required_places = _required_decimal_places(question)
    if required_places is not None and _SCALAR_NUMBER.fullmatch(s):
        s = _apply_decimal_precision(s, required_places)

    return s


def _is_give_up_answer(answer: str) -> bool:
    s = answer.strip().lower()
    if not s:
        return True
    if s.startswith("agent error"):
        return False
    if s in {"unknown", "n/a", "not found"}:
        return True
    if len(s) <= 240 and any(phrase in s for phrase in _GIVE_UP_PHRASES):
        return True
    return False


def _question_has_answer_contract_marker(question: str) -> bool:
    q = question.lower()
    return any(re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", q) for marker in _ANSWER_CONTRACT_QUESTION_MARKERS)


def _needs_answer_contract_check(question: str, answer: str) -> bool:
    if _is_give_up_answer(answer):
        return False
    q = question.lower()
    if not _question_has_answer_contract_marker(question):
        return False
    if _SCALAR_NUMBER.fullmatch(answer.strip()) and not any(marker in q for marker in ("time", "arrival", "date", "year")):
        return False
    return True


def _parse_contract_response(content: Any) -> dict[str, str]:
    if isinstance(content, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        text = str(content or "")
    text = text.strip()
    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {"status": "ok", "submitted_answer": "", "reason": ""}
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return {"status": "ok", "submitted_answer": "", "reason": ""}
    if not isinstance(parsed, dict):
        return {"status": "ok", "submitted_answer": "", "reason": ""}
    status = str(parsed.get("status", "ok") or "ok").strip().lower()
    if status not in {"ok", "repair"}:
        status = "ok"
    return {
        "status": status,
        "submitted_answer": str(parsed.get("submitted_answer", "") or "").strip(),
        "reason": str(parsed.get("reason", "") or "").strip(),
    }


def _repair_supported_by_context(question: str, reasoning_trace: str, repaired: str) -> bool:
    context = f"{question}\n{reasoning_trace}".lower()
    pieces = [
        piece.strip(" \t\r\n.,;:()[]{}\"'`")
        for piece in re.split(r"\s*(?:,|;|\band\b)\s*", repaired)
    ]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        return False
    return all(piece.lower() in context for piece in pieces)


def _apply_answer_contract(
    model: Any,
    question: str,
    answer: str,
    reasoning_trace: str,
    *,
    enabled: bool = True,
) -> str:
    if not enabled or not _needs_answer_contract_check(question, answer):
        return answer
    from langchain_core.messages import SystemMessage, HumanMessage

    prompt = (
        "You are a GAIA benchmark answer contract verifier. Check whether the submitted "
        "answer answers the ORIGINAL question, not an intermediate hop. If the answer type "
        "matches the question, return JSON {\"status\":\"ok\"}. If the answer is clearly "
        "the wrong type and the evidence trace contains the correct final answer, return "
        "JSON {\"status\":\"repair\",\"submitted_answer\":\"...\",\"reason\":\"...\"}. "
        "Do not guess. Do not repair unless the replacement appears in the evidence trace."
    )
    user = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"SUBMITTED ANSWER:\n{answer}\n\n"
        f"EVIDENCE TRACE:\n{reasoning_trace[-4000:]}"
    )
    try:
        response = model.invoke([SystemMessage(content=prompt), HumanMessage(content=user)])
    except Exception as exc:
        log.warning("answer_contract: verifier failed (%s), keeping original answer", exc)
        return answer
    decision = _parse_contract_response(getattr(response, "content", ""))
    if decision["status"] != "repair":
        return answer
    repaired = _normalize_gaia_submission(question, decision["submitted_answer"])
    if not repaired:
        return answer
    if not _repair_supported_by_context(question, reasoning_trace, repaired):
        log.warning("answer_contract: rejected unsupported repair %r", repaired)
        return answer
    log.info("answer_contract: repaired answer %r -> %r", answer, repaired)
    return repaired


def _parse_recovery_response(content: Any) -> dict[str, str]:
    if isinstance(content, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        text = str(content or "")
    text = text.strip()
    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {"status": "keep", "submitted_answer": "", "reason": ""}
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return {"status": "keep", "submitted_answer": "", "reason": ""}
    if not isinstance(parsed, dict):
        return {"status": "keep", "submitted_answer": "", "reason": ""}
    status = str(parsed.get("status", "keep") or "keep").strip().lower()
    if status not in {"keep", "answer"}:
        status = "keep"
    return {
        "status": status,
        "submitted_answer": str(parsed.get("submitted_answer", "") or "").strip(),
        "reason": str(parsed.get("reason", "") or "").strip(),
    }


def _apply_give_up_recovery(
    model: Any,
    question: str,
    answer: str,
    reasoning_trace: str,
    *,
    enabled: bool = True,
) -> str:
    if not enabled or not _is_give_up_answer(answer):
        return answer
    from langchain_core.messages import SystemMessage, HumanMessage

    prompt = (
        "You are a GAIA benchmark give-up recovery verifier. The submitted answer is "
        "empty, unknown, or a failure summary. If the evidence trace contains a concrete "
        "answer to the original question, return JSON {\"status\":\"answer\","
        "\"submitted_answer\":\"...\",\"reason\":\"...\"}. Otherwise return JSON "
        "{\"status\":\"keep\"}. Do not guess. The submitted_answer must appear in the trace."
    )
    user = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"CURRENT SUBMITTED ANSWER:\n{answer}\n\n"
        f"EVIDENCE TRACE:\n{reasoning_trace[-4000:]}"
    )
    try:
        response = model.invoke([SystemMessage(content=prompt), HumanMessage(content=user)])
    except Exception as exc:
        log.warning("give_up_recovery: verifier failed (%s), keeping original answer", exc)
        return answer
    decision = _parse_recovery_response(getattr(response, "content", ""))
    if decision["status"] != "answer":
        return answer
    recovered = _normalize_gaia_submission(question, decision["submitted_answer"])
    if not recovered:
        return answer
    if not _repair_supported_by_context(question, reasoning_trace, recovered):
        log.warning("give_up_recovery: rejected unsupported answer %r", recovered)
        return answer
    log.info("give_up_recovery: recovered answer %r -> %r", answer, recovered)
    return recovered


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
    from lilith_agent.models import (
        BatchAbortRateLimitError,
        QuestionRateLimitStreakError,
        RateLimitCooldownError,
        batch_rate_limit_pause_seconds,
        clear_batch_rate_limit_window,
        get_cheap_model,
        rate_limit_question_scope,
    )
    cfg = Config.from_env()
    cheap_model = get_cheap_model(cfg)

    total = len(questions)
    print(f"[runner] starting batch total={total} checkpoint_dir={checkpoint_root}", flush=True)

    def _invoke_task_once(task_state: dict, task_id: str):
        from lilith_agent.memory import ephemeral_memory

        with rate_limit_question_scope():
            with ephemeral_memory():
                return graph.invoke(task_state, {"configurable": {"thread_id": task_id}})

    def _maybe_pause_for_batch_rate_limit() -> None:
        pause_seconds = batch_rate_limit_pause_seconds()
        if pause_seconds is None:
            return
        print(f"[runner] pausing batch seconds={pause_seconds} reason=rate_limit_window", flush=True)
        log_runner.warning("[runner] pausing batch for %ss due to rate limit window", pause_seconds)
        time.sleep(pause_seconds)
        clear_batch_rate_limit_window()

    for idx, question in enumerate(questions, start=1):
        reset_vision_state()
        task_id = question.get("task_id")
        prompt = question.get("question")
        if not task_id or not prompt:
            print(f"[runner] skipping invalid question idx={idx} task_id={task_id!r}", flush=True)
            continue

        file_name = question.get("file_name")
        if file_name and client:
            print(f"[runner] task={task_id} downloading file={file_name}", flush=True)
            file_path = client.download_file(task_id, dest_dir=checkpoint_root / "files")
            if file_path:
                print(f"[runner] task={task_id} file_path={file_path.absolute()}", flush=True)
                prompt += f"\n\n[Attached File Path: {file_path.absolute()}]"
            else:
                print(f"[runner] task={task_id} file_download_missing file={file_name}", flush=True)

        # Scoring rules removed from here to reduce per-turn context bloat.
        # They are now applied in a final post-processing step.

        checkpoint_path = checkpoint_root / f"{task_id}.json"
        if checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text())
                log_runner.info("[runner] task=%s (%d/%d) skipped (checkpoint exists)", task_id, idx, total)
                print(f"[runner] task={task_id} ({idx}/{total}) skipped checkpoint={checkpoint_path}", flush=True)
                answers.append(
                    {
                        "task_id": task_id,
                        "submitted_answer": _submitted_answer_from_checkpoint(checkpoint),
                    }
                )
                continue
            except Exception:
                print(f"[runner] task={task_id} checkpoint unreadable path={checkpoint_path}", flush=True)
                pass

        log_runner.info(
            "[runner] task=%s (%d/%d) starting q=%r",
            task_id, idx, total, (prompt[:160] + "…") if len(prompt) > 160 else prompt,
        )
        print(f"[runner] task={task_id} ({idx}/{total}) starting", flush=True)

        state = {
            "messages": [HumanMessage(content=_wrap_user_question(prompt))],
            "iterations": 0
        }

        try:
            try:
                result = _invoke_task_once(state, task_id)
            except RateLimitCooldownError as exc:
                print(
                    f"[runner] task={task_id} rate_limited provider={exc.provider} model={exc.model} cooldown={exc.cooldown_seconds}",
                    flush=True,
                )
                log_runner.warning(
                    "[runner] task=%s rate limited provider=%s model=%s cooldown=%s",
                    task_id,
                    exc.provider,
                    exc.model,
                    exc.cooldown_seconds,
                )
                time.sleep(exc.cooldown_seconds)
                print(f"[runner] task={task_id} retrying after cooldown", flush=True)
                result = _invoke_task_once(state, task_id)
        except RateLimitCooldownError as exc:
            print(f"[runner] task={task_id} rate_limited_after_retry error={exc}", flush=True)
            log_runner.warning("[runner] task=%s rate limited after retry: %s", task_id, exc)
            answers.append({"task_id": task_id, "submitted_answer": "AGENT ERROR: RATE LIMITED"})
            _maybe_pause_for_batch_rate_limit()
            continue
        except QuestionRateLimitStreakError as exc:
            print(f"[runner] task={task_id} rate_limit_streak error={exc}", flush=True)
            log_runner.warning("[runner] task=%s rate limit streak: %s", task_id, exc)
            answers.append({"task_id": task_id, "submitted_answer": "AGENT ERROR: RATE LIMITED"})
            _maybe_pause_for_batch_rate_limit()
            continue
        except BatchAbortRateLimitError as exc:
            print(f"[runner] task={task_id} batch_abort_rate_limit reason={exc.reason}", flush=True)
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
            _maybe_pause_for_batch_rate_limit()
            continue
        except Exception as exc:
            print(f"[runner] task={task_id} agent_error type={type(exc).__name__} error={exc}", flush=True)
            log_runner.warning("[runner] task=%s agent error: %s", task_id, exc)
            answers.append(
                {
                    "task_id": task_id,
                    "submitted_answer": f"AGENT ERROR: {exc}",
                }
            )
            _maybe_pause_for_batch_rate_limit()
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
        submitted_answer = _normalize_gaia_submission(prompt, submitted_answer)

        reasoning_trace = _render_reasoning_trace(result["messages"])
        submitted_answer = _apply_answer_contract(
            cheap_model,
            prompt,
            submitted_answer,
            reasoning_trace,
            enabled=cfg.answer_contract_enabled,
        )
        submitted_answer = _apply_give_up_recovery(
            cheap_model,
            prompt,
            submitted_answer,
            reasoning_trace,
            enabled=cfg.give_up_recovery_enabled,
        )

        checkpoint = {
            "task_id": task_id,
            "question": prompt,
            "submitted_answer": submitted_answer,
            "reasoning_trace": reasoning_trace,
        }
        
        if submitted_answer and not submitted_answer.startswith("AGENT ERROR"):
            _write_checkpoint_atomic(checkpoint_path, checkpoint)
            print(f"[runner] task={task_id} checkpoint_written path={checkpoint_path}", flush=True)

        log_runner.info(
            "[runner] task=%s (%d/%d) answer=%r",
            task_id, idx, total,
            (submitted_answer[:160] + "…") if len(submitted_answer) > 160 else submitted_answer,
        )
        answer_preview = (submitted_answer[:160] + "…") if len(submitted_answer) > 160 else submitted_answer
        print(f"[runner] task={task_id} ({idx}/{total}) answer={answer_preview!r}", flush=True)
        answers.append({"task_id": task_id, "submitted_answer": submitted_answer.strip()})
        _maybe_pause_for_batch_rate_limit()

    print(f"[runner] finished batch produced={len(answers)}", flush=True)
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
        "You are a strict benchmark scoring assistant. Your ONLY job is to extract the EXACT final answer "
        "from a researcher's conclusion based on strict formatting rules.\n\n"
        "CRITICAL SCORING RULES:\n"
        "1. Remove ALL conversational filler, narrative text, or explanations (e.g., 'The answer is...', 'Based on...', 'I found...', 'The value is').\n"
        "2. Output ONLY the core value. NOT A FULL SENTENCE. NO EXPLANATIONS. NO ADDITIONAL CONTEXT.\n"
        "3. If the answer is a location, remove scene descriptors (INT., EXT., - DAY).\n"
        "4. Strip all trailing punctuation (., !).\n"
        "5. Honor requested units carefully.\n"
        "6. Output ONLY the bare text of the answer. No intro, no outro, no reasoning.\n"
        "7. STRICT MATH & PRECISION: If the question requires a specific number of decimal places or rounding, you MUST strictly apply it exactly as the Original Question specifies.\n"
        "8. ANTI-AUTOCORRECT (CRITICAL): NEVER correct spelling, grammar, or typos from the Researcher's Conclusion. If the conclusion contains 'Ploybius', you MUST output 'Ploybius'. Do not fix it to 'Polybius'. Preserve the conclusion's spelling verbatim.\n"
        "9. EXACT ENTITY NAMING: Do not expand, formalize, or translate names. If the conclusion uses a common name (e.g., 'Brunei'), DO NOT output the official state name ('Brunei Darussalam'). Match the conclusion's form.\n"
        "10. STRIP VARIABLES: If the question asks for a mathematical value, output JUST the number. If the conclusion says 'x = 2', 'x_2', or 'The value is 2', output ONLY '2'.\n"
        "11. EXACT PRECISION (NO ROUNDING): Do not round numbers, alter decimal places, or change units unless the Original Question explicitly instructs. If the conclusion is '1.456', do not round to '1.46'."
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
        if not _is_safe_llm_formatter_output(determ, cleaned):
            expanded = _expand_to_source_token(determ, cleaned)
            if expanded:
                log.warning("formatter: expanded unsafe LLM substring to source token")
                return expanded
            log.warning("formatter: rejected unsafe LLM rewrite, falling back to deterministic")
            return determ
        log.info("formatter: LLM returned out_len=%d", len(cleaned))
        return cleaned
    except Exception as e:
        log.warning("formatter: LLM call failed (%s), falling back to deterministic", e)
        return determ


def _submitted_answer_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    return checkpoint.get("submitted_answer") or checkpoint.get("final_answer", "")
