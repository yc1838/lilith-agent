from __future__ import annotations

from pathlib import Path

import pytest

from lilith_agent.tools.files import (
    _resolve_safe_write_path,
    find_files,
    glob_files,
    grep,
    ls,
    read_file,
    set_write_root,
    write_file,
)


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


def test_ls_expands_tilde(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "marker.txt").write_text("x")
    out = ls("~")
    assert "marker.txt" in out


def test_read_file_expands_tilde(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "note.md").write_text("hello goddess")
    out = read_file("~/note.md")
    assert "hello goddess" in out


def test_grep_expands_tilde(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "lore.txt").write_text("The She\nDemiurge\n")
    out = grep("She", "~/lore.txt")
    assert "The She" in out


def test_glob_files_expands_tilde(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "a.lol").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.lol").write_text("")
    out = glob_files("~/**/*.lol")
    assert "a.lol" in out
    assert "b.lol" in out


def test_glob_files_accepts_absolute_pattern(tmp_path: Path):
    (tmp_path / "莉莉丝女神.lol").write_text("")
    (tmp_path / "deep").mkdir()
    (tmp_path / "deep" / "莉莉丝女神.lol").write_text("")
    out = glob_files(f"{tmp_path}/**/莉莉丝女神.lol")
    assert "莉莉丝女神.lol" in out
    assert "deep" in out


def test_find_files_locates_by_name(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "target.txt").write_text("")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "other.md").write_text("")
    out = find_files("target.txt", str(tmp_path))
    assert "target.txt" in out
    assert "other.md" not in out


def test_find_files_expands_tilde(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "needle.lol").write_text("")
    out = find_files("needle.lol", "~")
    assert "needle.lol" in out


def test_find_files_missing_root_errors(tmp_path: Path):
    out = find_files("anything", str(tmp_path / "does_not_exist"))
    assert out.startswith("ERROR")


def test_find_files_excludes_node_modules(tmp_path: Path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.txt").write_text("")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.txt").write_text("")
    out = find_files("*.txt", str(tmp_path))
    assert "keep.txt" in out
    assert "skip.txt" not in out


def test_find_files_excludes_dot_git(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config").write_text("")
    out = find_files("config", str(tmp_path))
    assert str(tmp_path / "src" / "config") in out
    assert str(tmp_path / ".git" / "config") not in out


def test_find_files_truncation_sentinel(tmp_path: Path):
    for i in range(4):
        (tmp_path / f"file{i}.txt").write_text("")
    out = find_files("*.txt", str(tmp_path), max_results=3)
    assert "[truncated:" in out
    assert "of 4" in out


def test_filesystem_tool_descriptions_mention_tilde():
    from lilith_agent.config import Config
    from lilith_agent.tools import build_tools
    cfg = Config.from_env()
    tools_by_name = {t.name: t for t in build_tools(cfg)}
    for name in ("read_file", "ls", "grep", "glob_files", "find_files"):
        assert "~" in tools_by_name[name].description, f"{name} description must mention ~"


def test_find_files_description_warns_against_root_slash():
    from lilith_agent.config import Config
    from lilith_agent.tools import build_tools
    cfg = Config.from_env()
    tools_by_name = {t.name: t for t in build_tools(cfg)}
    desc = tools_by_name["find_files"].description
    assert "/" in desc and ("never" in desc.lower() or "do not" in desc.lower())
