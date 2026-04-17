"""Convert .checkpoints/<task_id>.json files into a GAIA leaderboard JSONL.

The official scorer (https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py)
expects:
    {"task_id": ..., "model_answer": ..., "reasoning_trace": ...}

It does NOT search for "FINAL ANSWER:" — it normalizes whitespace + punctuation
and exact-matches. Submit the bare answer only.

Usage:
    python scripts/build_leaderboard_submission.py \\
        --checkpoint-dir .checkpoints \\
        --out submission.jsonl

    # Filter to only test-split task_ids that the agent actually answered:
    python scripts/build_leaderboard_submission.py --split test --level all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from lilith_agent.gaia_dataset import GaiaDatasetClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default=".checkpoints")
    p.add_argument("--out", default="submission.jsonl")
    p.add_argument("--split", default=None, help="If set, only include task_ids from this split (test/validation)")
    p.add_argument("--config", default="2023_all")
    p.add_argument("--include-errors", action="store_true",
                   help="Include answers starting with 'AGENT ERROR' (default: skip)")
    p.add_argument("--pad-missing", action="store_true",
                   help="When --split is set, emit placeholder rows for any task_ids "
                        "in that split that lack a real answer, so the submission "
                        "has the exact count the leaderboard expects.")
    p.add_argument("--pad-answer", default="",
                   help="Placeholder string for padded rows (default: empty string)")
    p.add_argument("--levels", default=None,
                   help="Comma-separated levels to restrict to, e.g. '1' or '1,2'. "
                        "Only active with --split.")
    args = p.parse_args()

    ckpt_root = Path(args.checkpoint_dir)
    if not ckpt_root.exists():
        print(f"No checkpoints at {ckpt_root}", file=sys.stderr)
        sys.exit(1)

    allowed_ids: set[str] | None = None
    split_questions: list[dict] = []
    if args.split:
        token = os.getenv("HF_TOKEN") or os.getenv("GAIA_HUGGINGFACE_API_KEY")
        client = GaiaDatasetClient(config=args.config, split=args.split, level=None, token=token)
        split_questions = client.get_questions()
        if args.levels:
            wanted = {s.strip() for s in args.levels.split(",") if s.strip()}
            split_questions = [q for q in split_questions if str(q.get("Level")) in wanted]
        allowed_ids = {q["task_id"] for q in split_questions}
        print(f"Restricting to {len(allowed_ids)} task_ids from split={args.split}"
              + (f" levels={args.levels}" if args.levels else ""))

    written = 0
    padded = 0
    skipped_errors = 0
    skipped_missing_split = 0
    skipped_blank = 0
    covered_ids: set[str] = set()

    out_path = Path(args.out)
    with out_path.open("w") as fh:
        for ckpt_path in sorted(ckpt_root.glob("*.json")):
            try:
                data = json.loads(ckpt_path.read_text())
            except Exception as e:
                print(f"skip {ckpt_path.name}: {e}", file=sys.stderr)
                continue

            task_id = data.get("task_id") or ckpt_path.stem
            answer = (data.get("submitted_answer") or data.get("final_answer") or "").strip()

            if allowed_ids is not None and task_id not in allowed_ids:
                skipped_missing_split += 1
                continue
            if not answer:
                skipped_blank += 1
                continue
            if answer.startswith("AGENT ERROR") and not args.include_errors:
                skipped_errors += 1
                continue

            record = {"task_id": task_id, "model_answer": answer}
            # Optional: include any reasoning the checkpoint stored
            trace = data.get("reasoning_trace") or data.get("reasoning")
            if trace:
                record["reasoning_trace"] = trace

            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            covered_ids.add(task_id)

        if args.pad_missing:
            if not split_questions:
                print("--pad-missing requires --split; skipping pad step", file=sys.stderr)
            else:
                for q in split_questions:
                    tid = q["task_id"]
                    if tid in covered_ids:
                        continue
                    record = {
                        "task_id": tid,
                        "model_answer": args.pad_answer,
                        "reasoning_trace": "no-attempt placeholder",
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    padded += 1

    print(f"Wrote {written} real records to {out_path}")
    if padded: print(f"  padded {padded} missing task_ids with placeholder='{args.pad_answer}'")
    if skipped_errors: print(f"  skipped {skipped_errors} AGENT ERROR rows (use --include-errors to keep)")
    if skipped_blank: print(f"  skipped {skipped_blank} blank-answer rows")
    if skipped_missing_split: print(f"  skipped {skipped_missing_split} rows not in split={args.split}")
    print(f"Total rows in file: {written + padded}")


if __name__ == "__main__":
    main()
