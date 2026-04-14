import fnmatch
from pathlib import Path
from typing import Optional

import pandas as pd
from docx import Document
from pypdf import PdfReader


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
    p = Path(path)
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
    p = Path(path)
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
    p = Path(path)
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
    p = Path(root_dir)
    try:
        # Use rglob for recursive search if pattern starts with **/
        if pattern.startswith("**/"):
            matches = list(p.rglob(pattern[3:]))
        else:
            matches = list(p.glob(pattern))
            
        if not matches:
            return "No files found matching the pattern."
        return "\n".join(str(m) for m in sorted(matches))
    except Exception as e:
        return f"ERROR globbing {pattern}: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file. Used for offloading context or saving intermediate results."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}."
    except Exception as e:
        return f"ERROR writing to {path}: {e}"
