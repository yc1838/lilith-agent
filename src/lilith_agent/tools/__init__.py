"""GAIA agent tools. Registry assembled by build_tools()."""
from __future__ import annotations

from typing import Optional
from langchain_core.tools import BaseTool, tool

from lilith_agent.config import Config
from lilith_agent.tools.audio import transcribe_audio as _transcribe_audio
from lilith_agent.tools.files import (
    read_file as _read_file,
    ls as _ls,
    grep as _grep,
    glob_files as _glob_files,
    write_file as _write_file
)
from lilith_agent.tools.todos import (
    write_todos as _write_todos,
    mark_todo_done as _mark_todo_done,
    read_todos as _read_todos
)
from lilith_agent.tools.pdf import inspect_pdf as _inspect_pdf
from lilith_agent.tools.python_exec import run_python as _run_python
from lilith_agent.tools.search import web_search as _web_search
from lilith_agent.tools.vision import inspect_visual_content as _inspect_visual_content
from lilith_agent.tools.web import fetch_url as _fetch_url
from lilith_agent.tools.youtube import (
    youtube_transcript as _youtube_transcript,
    youtube_frame_at as _youtube_frame_at,
)
from lilith_agent.tools.academic import (
    arxiv_search as _arxiv_search, 
    crossref_search as _crossref_search,
    count_journal_articles as _count_journal_articles
)
from lilith_agent.tools.filters import filter_entities as _filter_entities


def build_tools(cfg: Config) -> list[BaseTool]:
    """Build the list of LangChain tools, injecting config via closures."""

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the web for general information using DuckDuckGo (primary) and Tavily (fallback).
        Returns a list of title/url/snippet results. This is your primary web search tool."""
        return _web_search(query, api_key=cfg.tavily_api_key, max_results=max_results)

    @tool
    def fetch_url(url: str, max_chars: int = 8000) -> str:
        """Fetch a URL and return its extracted main text (trafilatura)."""
        return _fetch_url(url, max_chars=max_chars)

    @tool
    def run_python(code: str, timeout: int = 30) -> str:
        """Run Python code in a sandboxed subprocess and return stdout + last expression.

        ISOLATED FROM THE HOST FILESYSTEM. This tool runs inside a container (or a
        scrubbed subprocess with cwd pinned to a throwaway tmpdir). It CANNOT open
        files from the user's workspace — paths like './notes.md', '.lilith/...',
        or any relative path resolve against the sandbox's own empty scratch dir,
        not the repo. Any file written here vanishes when the call returns.

        To move data across the boundary:
        - To READ a workspace file, call `read_file` first and pass the returned
          text into `run_python` as a string literal in `code`.
        - To PERSIST output the user can see, call `write_file` — do NOT try to
          write from inside `run_python`.

        AVAILABLE LIBRARIES: requests, beautifulsoup4 (bs4), pandas, trafilatura, openpyxl, faster-whisper, pypdf.
        CRITICAL: If you use this tool to scrape websites, you MUST include a 'User-Agent' header.
        Example: headers = {'User-Agent': 'Mozilla/5.0...'}
        """
        return _run_python(code, timeout=timeout)

    @tool
    def read_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None, max_chars: int = 20000) -> str:
        """Read a local file with optional line-based chunking (start_line/end_line). Returns text."""
        return _read_file(path, start_line=start_line, end_line=end_line, max_chars=max_chars)

    @tool
    def ls(path: str = ".") -> str:
        """List contents of a directory. Use this to find available files."""
        return _ls(path)

    @tool
    def grep(pattern: str, path: str) -> str:
        """Search for a pattern in a file. Returns matching lines with line numbers."""
        return _grep(pattern, path)

    @tool
    def glob_files(pattern: str) -> str:
        """Find files matching a glob pattern (e.g., '**/*.csv')."""
        return _glob_files(pattern)

    @tool
    def write_file(path: str, content: str) -> str:
        """Persist text to a file the user can open afterwards.

        This is the ONLY tool that writes to the user's workspace. `run_python`
        runs in an isolated sandbox with no host-filesystem access, so files it
        creates are lost when the call returns — always route persisted output
        through this tool instead.

        `path` is relative to the agent's write root (`.lilith/scratch` by
        default); absolute paths and `..`-escapes are rejected.
        """
        return _write_file(path, content)

    @tool
    def write_todos(todos: list[str]) -> str:
        """Initialize or overwrite the current task list (Todo list). Use for high-level planning."""
        return _write_todos(todos)

    @tool
    def mark_todo_done(index: int) -> str:
        """Mark a specific todo as complete by its position."""
        return _mark_todo_done(index)

    @tool
    def transcribe_audio(path: str) -> str:
        """Transcribe an audio file using faster-whisper. Returns the transcript text."""
        return _transcribe_audio(path, model_size=cfg.whisper_model)

    @tool
    def inspect_pdf(url_or_path: str, query: str) -> str:
        """Extract text from a PDF (local or remote). Recommended for reading academic papers or long documents."""
        return _inspect_pdf(url_or_path, query=query)

    @tool
    def inspect_visual_content(file_path_or_url: str, prompt: str) -> str:
        """Use Gemini Multimodal Vision to analyze an image (PNG, JPG) or video (MP4). Use this for counting objects on screen or identifying visual details."""
        return _inspect_visual_content(file_path_or_url, prompt=prompt, cfg=cfg)

    @tool
    def youtube_transcript(url: str) -> str:
        """Fetch the spoken-word transcript (captions) for a YouTube video URL.
        Does NOT capture on-screen text overlays — use `youtube_frame_at` + `inspect_visual_content` for those."""
        return _youtube_transcript(url)

    @tool
    def youtube_frame_at(url: str, timestamp_seconds: float) -> str:
        """Download a YouTube video and extract ONE frame at the given timestamp as a local PNG.
        Returns the local PNG path — pass it to `inspect_visual_content` to read on-screen text or visual details.
        REQUIRED for questions about what is visible on screen at a specific time in a YouTube video
        (e.g. a phrase shown as a title card, or an object visible at 0:30).
        """
        return _youtube_frame_at(url, timestamp_seconds=timestamp_seconds)

    @tool
    def arxiv_search(query: str, max_results: int = 5) -> str:
        """Search arXiv for papers. Returns a list of titles, authors, and summaries. Use this for modern academic pre-prints."""
        return _arxiv_search(query, max_results=max_results)

    @tool
    def crossref_search(filter_str: str, rows: int = 100, cursor: str = "*") -> str:
        """Search CrossRef API for bibliographic metadata.
        Use filter strings like 'issn:0028-0836,type:journal-article,from-pub-date:2020,until-pub-date:2020'.
        Returns a JSON list of works. Pair with filter_entities to prune broad results.
        """
        return _crossref_search(filter_str, rows=rows, cursor=cursor)

    @tool
    def count_journal_articles(journal_name: str, year: int, is_research_only: bool = True) -> str:
        """High-precision tool to count articles in a journal for a given year.
        Use this for definitive bibliographic metric counting (e.g., 'articles in Nature in 2020').
        It handles complex filtering and official source scraping internally.
        """
        return _count_journal_articles(journal_name, year, is_research_only=is_research_only)

    @tool
    def filter_entities(entities: list[dict], keep_conditions: list[str] = None, remove_conditions: list[str] = None) -> str:
        """Filter a list of entities based on conditions. Returns a metadata-rich JSON.
        Use this when a prior tool (like crossref_search) returns a 'broad' list that needs pruning (e.g., removing book reviews).
        """
        return _filter_entities(entities, keep_conditions=keep_conditions, remove_conditions=remove_conditions)

    return [
        web_search,
        fetch_url,
        run_python,
        read_file,
        transcribe_audio,
        youtube_transcript,
        youtube_frame_at,
        inspect_pdf,
        inspect_visual_content,
        arxiv_search,
        crossref_search,
        count_journal_articles,
        filter_entities,
        ls,
        grep,
        glob_files,
        write_file,
        write_todos,
        mark_todo_done,
    ]
