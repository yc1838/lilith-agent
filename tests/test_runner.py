from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilith_agent.runner import _write_checkpoint_atomic


def test_atomic_write_produces_no_tmp_leftover_on_success(tmp_path: Path):
    dest = tmp_path / "abc123.json"
    _write_checkpoint_atomic(dest, {"task_id": "abc123", "submitted_answer": "42"})
    assert dest.exists()
    assert json.loads(dest.read_text())["submitted_answer"] == "42"
    # No .tmp sibling left behind
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_does_not_corrupt_existing_file_on_serialization_failure(tmp_path: Path):
    dest = tmp_path / "abc123.json"
    dest.write_text(json.dumps({"task_id": "abc123", "submitted_answer": "good"}))

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        _write_checkpoint_atomic(dest, {"task_id": "abc123", "submitted_answer": Unserializable()})

    # Existing file must still be intact, not truncated or partial.
    data = json.loads(dest.read_text())
    assert data["submitted_answer"] == "good"
    assert list(tmp_path.glob("*.tmp")) == []
