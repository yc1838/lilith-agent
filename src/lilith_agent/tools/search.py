from __future__ import annotations

from ddgs import DDGS
from tavily import TavilyClient, UsageLimitExceededError
from tavily.errors import ForbiddenError

_SEARCH_RECOVERY_HINT = (
    "SEARCH_HINT: no useful results. Try: (1) rephrase with alternative keywords or narrower scope; "
    "(2) for academic/scientific queries, use arxiv_search or crossref_search; "
    "(3) for removed or historical pages, search via https://web.archive.org."
)
_SEARCH_LOG_PREVIEW_CHARS = 240


def _search_preview(text: str) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= _SEARCH_LOG_PREVIEW_CHARS:
        return compact
    return compact[: _SEARCH_LOG_PREVIEW_CHARS - 1] + "…"


def _print_search_log(message: str, *args) -> None:
    text = message % args if args else message
    print(text, flush=True)


def _ddg_search(query: str, max_results: int) -> str:
    results = DDGS().text(query, max_results=max_results)
    lines = [
        f"- {r.get('title', '')} ({r.get('href', '')})\n  {r.get('body', '')}"
        for r in results
    ]
    return "\n".join(lines) if lines else "No results."


def web_search(query: str, api_key: str, max_results: int = 5) -> str:
    # 1. Try DuckDuckGo first
    _print_search_log("[web_search] start provider=duckduckgo query=%r max_results=%d", query, max_results)
    try:
        ddg_results = _ddg_search(query, max_results)
    except Exception as e:
        ddg_results = f"DDG ERROR: {e}"
        _print_search_log(
            "[web_search] provider=duckduckgo status=error query=%r error=%r",
            query,
            str(e),
        )

    if ddg_results and "No results." not in ddg_results and not ddg_results.startswith("DDG ERROR"):
        out = f"[Source: DuckDuckGo]\n{ddg_results}"
        _print_search_log(
            "[web_search] provider=duckduckgo status=success chars=%d preview=%r",
            len(out),
            _search_preview(out),
        )
        return out

    _print_search_log(
        "[web_search] provider=duckduckgo status=empty_or_failed chars=%d preview=%r",
        len(ddg_results or ""),
        _search_preview(ddg_results or ""),
    )

    # 2. Fallback to Tavily if DDG failed or returned nothing
    if not api_key:
        return f"[Source: DuckDuckGo (Empty)]\n{ddg_results}\n[Tavily fallback skipped: no key]\n{_SEARCH_RECOVERY_HINT}"

    client = TavilyClient(api_key=api_key)
    _print_search_log("[web_search] start provider=tavily query=%r max_results=%d", query, max_results)
    try:
        res = client.search(query=query, max_results=max_results)
        lines: list[str] = []
        for item in res.get("results", []):
            lines.append(f"- {item['title']} ({item['url']})\n  {item['content']}")
        if not lines:
            out = f"[Source: Tavily (DDG fallback: {ddg_results})]\nNo results.\n{_SEARCH_RECOVERY_HINT}"
            _print_search_log(
                "[web_search] provider=tavily status=empty chars=%d preview=%r",
                len(out),
                _search_preview(out),
            )
            return out
        tavily_results = "\n".join(lines)
        out = f"[Source: Tavily (DDG fallback: {ddg_results})]\n{tavily_results}"
        _print_search_log(
            "[web_search] provider=tavily status=success chars=%d preview=%r",
            len(out),
            _search_preview(out),
        )
        return out
    except (UsageLimitExceededError, ForbiddenError) as e:
        _print_search_log("[web_search] provider=tavily status=limit_or_forbidden query=%r error=%r", query, str(e))
        return f"[Source: DuckDuckGo (Empty)]\n{ddg_results}\n[Tavily fallback failed: limit/forbidden: {e}]\n{_SEARCH_RECOVERY_HINT}"
    except Exception as e:
        _print_search_log("[web_search] provider=tavily status=error query=%r error=%r", query, str(e))
        return f"[Source: DuckDuckGo (Empty)]\n{ddg_results}\n[Tavily fallback failed: {e}]\n{_SEARCH_RECOVERY_HINT}"
