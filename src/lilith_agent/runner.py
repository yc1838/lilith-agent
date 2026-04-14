from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

try:
    from lilith_agent.tools.vision import reset_vision_state
except ImportError:
    def reset_vision_state(): pass


def run_agent_on_questions(graph: Any, questions: list[dict], checkpoint_dir: str | Path, client: Any = None) -> list[dict]:
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    answers: list[dict] = []

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

        # Apply GAIA strict formatting rules
        prompt += (
            "\n\n--- GAIA BENCHMARK SCORING RULES ---\n"
            "This is a GAIA benchmark evaluation. Your response will be scored via exact string matching.\n"
            "You MUST format your final generated message to be ONLY the bare, exact answer and nothing else.\n"
            "CRITICAL RULES:\n"
            "1. Remove all conversational filler (e.g., 'The answer is...', 'Based on...', 'I found...').\n"
            "2. DO NOT output paragraphs of reasoning in your final message. Put reasoning in your thoughts or earlier tool calls.\n"
            "3. If the answer is a location, remove scene descriptors like 'INT.', 'EXT.', '- DAY', '- NIGHT'.\n"
            "4. Strip all trailing punctuation (., !).\n"
            "5. If asked 'how many thousands', and your calculated result is 17000, output '17'.\n"
            "6. Output ONLY the core answer itself. No other text."
        )

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
        
        checkpoint = {
            "task_id": task_id,
            "question": prompt,
            "submitted_answer": submitted_answer
        }
        
        if submitted_answer and not submitted_answer.startswith("AGENT ERROR"):
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True))
            
        answers.append({"task_id": task_id, "submitted_answer": submitted_answer})

    return answers


def _submitted_answer_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    return checkpoint.get("submitted_answer") or checkpoint.get("final_answer", "")
