from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

try:
    from lilith_agent.tools.vision import reset_vision_state
except ImportError:
    def reset_vision_state(): pass


_TRACE_TOOL_OUTPUT_MAX = 400  # chars per tool result kept in reasoning_trace
_TRACE_AI_TEXT_MAX = 800      # chars per AI message text kept in reasoning_trace


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

    for question in questions:
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
                answers.append(
                    {
                        "task_id": task_id,
                        "submitted_answer": _submitted_answer_from_checkpoint(checkpoint),
                    }
                )
                continue
            except Exception:
                pass

        state = {
            "messages": [HumanMessage(content=prompt)],
            "iterations": 0
        }
        
        try:
            result = graph.invoke(state, {"configurable": {"thread_id": task_id}})
        except Exception as exc:
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
        
        submitted_answer = _final_formatting_cleanup(cheap_model, prompt, submitted_answer)
        
        reasoning_trace = _render_reasoning_trace(result["messages"])

        checkpoint = {
            "task_id": task_id,
            "question": prompt,
            "submitted_answer": submitted_answer,
            "reasoning_trace": reasoning_trace,
        }
        
        if submitted_answer and not submitted_answer.startswith("AGENT ERROR"):
            _write_checkpoint_atomic(checkpoint_path, checkpoint)
            
        answers.append({"task_id": task_id, "submitted_answer": submitted_answer.strip()})

    return answers


def _final_formatting_cleanup(model: Any, question: str, raw_answer: str) -> str:
    """Post-processor to ensure the agent's raw output follows strict GAIA scoring rules."""
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # We use a focused, one-shot prompt to extract exactly what the benchmark expects.
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
        f"Researcher's Conclusion: {raw_answer}\n\n"
        "Format the 'Researcher's Conclusion' into the bare answer required by the rules above."
    )
    
    try:
        resp = model.invoke([SystemMessage(content=instructions), HumanMessage(content=prompt)])
        # Properly extract text string from response (handles both simple strings and complex list-of-dicts)
        content = resp.content
        if isinstance(content, list):
            cleaned = "".join([c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"])
        else:
            cleaned = str(content)
            
        cleaned = cleaned.strip()
        # Fallback security: if the model returned nothing or an error, keep the original
        return cleaned if cleaned else raw_answer
    except Exception as e:
        print(f"Warning: Formatting cleanup failed: {e}")
        return raw_answer


def _submitted_answer_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    return checkpoint.get("submitted_answer") or checkpoint.get("final_answer", "")
