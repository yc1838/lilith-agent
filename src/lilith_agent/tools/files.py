import glob as _glob
import os
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
from docx import Document
from pypdf import PdfReader


# Agent writes are confined to this directory. Defaults to `.lilith/scratch`
# under the CWD; override with set_write_root() at startup if you want a
# per-run tmpfs-style root. The LLM must not be able to change this.
_WRITE_ROOT: Path = Path(".lilith/scratch").resolve()


def set_write_root(root: str | Path) -> None:
    """Set the sandboxed root directory for write_file. Call once at startup."""
    global _WRITE_ROOT
    _WRITE_ROOT = Path(root).resolve()
    _WRITE_ROOT.mkdir(parents=True, exist_ok=True)


def _resolve_safe_write_path(path: str) -> Path:
    """Resolve `path` against the write root. Raises ValueError on unsafe input.

    Rejects absolute paths and any relative path that resolves outside the root
    (e.g. via `..`). Returns a fully resolved Path inside the root.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")

    p = Path(path)
    if p.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {path!r}")

    root = _WRITE_ROOT.resolve()
    candidate = (root / p).resolve()

    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"path would escape write root: {path!r}")

    return candidate


def read_file(
    path: str, 
    start_line: Optional[int] = None, 
    end_line: Optional[int] = None, 
    max_chars: int = 20000
) -> str:
    """
    Reads a file with optional line-based chunking.
    Used to prevent context overflow for large files.
    """
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return f"ERROR: File {path} does not exist."
    if not p.is_file():
        return f"ERROR: {path} is not a file."

    ext = p.suffix.lower()
    try:
        if ext in (".txt", ".md", ".py", ".sh", ".json", ".yaml", ".yml"):
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            if start_line is not None or end_line is not None:
                start = (start_line - 1) if start_line else 0
                end = end_line if end_line else len(lines)
                text = "\n".join(lines[start:end])
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n...[truncated]"
                return text
            text = "\n".join(lines)
        elif ext == ".csv":
            df = pd.read_csv(p)
            text = df.to_csv(index=False)
        elif ext in (".xlsx", ".xls"):
            dfs = pd.read_excel(p, sheet_name=None)
            text = "\n\n".join(
                f"--- Sheet: {name} ---\n{df.to_csv(index=False)}"
                for name, df in dfs.items()
            )
        elif ext == ".pdf":
            reader = PdfReader(str(p))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        elif ext == ".docx":
            doc = Document(str(p))
            text = "\n".join(para.text for para in doc.paragraphs)
        else:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return f"[binary file, {p.stat().st_size} bytes]"
    except Exception as e:
        return f"ERROR reading {path}: {e}"
        
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def ls(path: str = ".") -> str:
    """List contents of a directory."""
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return f"ERROR: Path {path} does not exist."
    if not p.is_dir():
        return f"ERROR: {path} is not a directory."
    
    try:
        items = []
        for item in p.iterdir():
            suffix = "/" if item.is_dir() else ""
            items.append(f"{item.name}{suffix}")
        return "\n".join(sorted(items))
    except Exception as e:
        return f"ERROR listing {path}: {e}"


def grep(pattern: str, path: str, ignore_case: bool = True) -> str:
    """Search for a pattern in a file and return matching lines with line numbers."""
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return f"ERROR: File {path} does not exist."
    
    try:
        matches = []
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines, 1):
            if ignore_case:
                if pattern.lower() in line.lower():
                    matches.append(f"Line {i}: {line}")
            else:
                if pattern in line:
                    matches.append(f"Line {i}: {line}")
        
        if not matches:
            return "No matches found."
        return "\n".join(matches)
    except Exception as e:
        return f"ERROR grepping {path}: {e}"


def glob_files(pattern: str, root_dir: str = ".") -> str:
    """Find files matching a glob pattern."""
    pattern = os.path.expanduser(pattern)
    root_dir = os.path.expanduser(root_dir)
    try:
        if os.path.isabs(pattern):
            matches = _glob.glob(pattern, recursive=True)
        else:
            matches = _glob.glob(pattern, root_dir=root_dir, recursive=True)

        if not matches:
            return "No files found matching the pattern."
        return "\n".join(sorted(matches))
    except Exception as e:
        return f"ERROR globbing {pattern}: {e}"


def find_files(name: str, root: str = ".", max_results: int = 200) -> str:
    """Locate files by name under `root` using unix `find`. Fast deep search."""
    root = os.path.expanduser(root)
    if not Path(root).exists():
        return f"ERROR: Path {root} does not exist."
    try:
        result = subprocess.run(
            ["find", str(root), "-name", name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = [ln for ln in result.stdout.splitlines() if ln]
        if not lines:
            return "No files found."
        return "\n".join(lines[:max_results])
    except subprocess.TimeoutExpired:
        return "ERROR: find timed out (>30s)."
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file inside the sandboxed write root.

    The agent may only write under the configured write root (default
    `.lilith/scratch`). Absolute paths and `..`-escapes are rejected.
    """
    try:
        safe = _resolve_safe_write_path(path)
    except ValueError as exc:
        return f"ERROR writing to {path}: {exc}"

    try:
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {safe}."
    except Exception as e:
        return f"ERROR writing to {path}: {e}"
