"""DuckDuckGo web search – zero API key required.

Uses the ddgs library (pip install ddgs).
Falls back gracefully if the library is not installed.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

log = logging.getLogger(__name__)

_HARD_TIMEOUT_S = 15


def _run_ddgs(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS
    with DDGS(timeout=10) as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo. Returns title/url/snippet results."""
    try:
        import ddgs  # noqa: F401
    except ImportError:
        return "ERROR: ddgs not installed. Run: pip install ddgs"

    # Run in a worker thread with a hard wall-clock timeout: DDGS' pagination
    # loop can stall beyond its per-request timeout and block the agent.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web_search")
    try:
        future = pool.submit(_run_ddgs, query, max_results)
        try:
            results = future.result(timeout=_HARD_TIMEOUT_S)
        except FuturesTimeout:
            log.warning("[web_search] hard timeout after %ss, query=%r", _HARD_TIMEOUT_S, query)
            return f"ERROR: web_search timed out after {_HARD_TIMEOUT_S}s"
        finally:
            # Don't block on the possibly-stuck worker thread on shutdown.
            pool.shutdown(wait=False, cancel_futures=True)
    except Exception as e:
        log.warning("[web_search] DuckDuckGo error: %s", e)
        return f"ERROR: DuckDuckGo search failed: {e}"

    if not results:
        return "No results."

    lines: list[str] = []
    for r in results:
        title = r.get("title", "No Title")
        url = r.get("href", "No URL")
        body = r.get("body", "No snippet available")
        lines.append(f"- {title} ({url})\n  {body}")

    import json
    metadata = {
        "value": len(results),
        "data_source": "duckduckgo",
        "record_type": "web-snippet",
        "type_strictness": "broad",
        "note": "Web snippets are unstructured and may contain irrelevant or duplicate information."
    }
    res_text = "\n".join(lines)
    return f"{res_text}\n\nMETADATA:\n{json.dumps(metadata, indent=2)}"
