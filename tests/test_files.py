from __future__ import annotations

from pathlib import Path

import pytest

from lilith_agent.tools.files import _resolve_safe_write_path, set_write_root, write_file


def test_relative_path_inside_root_is_allowed(tmp_path: Path):
    set_write_root(tmp_path)
    p = _resolve_safe_write_path("notes.txt")
    assert p == (tmp_path / "notes.txt").resolve()


def test_nested_relative_path_allowed(tmp_path: Path):
    set_write_root(tmp_path)
    p = _resolve_safe_write_path("sub/dir/notes.txt")
    assert p == (tmp_path / "sub" / "dir" / "notes.txt").resolve()


def test_absolute_path_rejected(tmp_path: Path):
    set_write_root(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        _resolve_safe_write_path("/etc/passwd")


def test_dotdot_escape_rejected(tmp_path: Path):
    set_write_root(tmp_path)
    with pytest.raises(ValueError, match="escape"):
        _resolve_safe_write_path("../outside.txt")


def test_deep_dotdot_escape_rejected(tmp_path: Path):
    set_write_root(tmp_path)
    with pytest.raises(ValueError, match="escape"):
        _resolve_safe_write_path("sub/../../outside.txt")


def test_write_file_inside_root_succeeds(tmp_path: Path):
    set_write_root(tmp_path)
    result = write_file("note.txt", "hello")
    assert "Successfully wrote" in result
    assert (tmp_path / "note.txt").read_text() == "hello"


def test_write_file_outside_root_returns_error_without_writing(tmp_path: Path):
    target = tmp_path / "outside.txt"
    set_write_root(tmp_path / "scratch")
    result = write_file(str(target), "hello")
    assert result.startswith("ERROR")
    assert not target.exists()
