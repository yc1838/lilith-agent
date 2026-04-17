from __future__ import annotations

from ddgs import DDGS
from tavily import TavilyClient, UsageLimitExceededError
from tavily.errors import ForbiddenError


def _ddg_search(query: str, max_results: int) -> str:
    results = DDGS().text(query, max_results=max_results)
    lines = [
        f"- {r.get('title', '')} ({r.get('href', '')})\n  {r.get('body', '')}"
        for r in results
    ]
    return "\n".join(lines) if lines else "No results."


def web_search(query: str, api_key: str, max_results: int = 5) -> str:
    # 1. Try DuckDuckGo first
    try:
        ddg_results = _ddg_search(query, max_results)
    except Exception as e:
        ddg_results = f"DDG ERROR: {e}"

    if ddg_results and "No results." not in ddg_results and not ddg_results.startswith("DDG ERROR"):
        return f"[Source: DuckDuckGo]\n{ddg_results}"

    # 2. Fallback to Tavily if DDG failed or returned nothing
    if not api_key:
        return f"[Source: DuckDuckGo (Empty)]\n{ddg_results}\n[Tavily fallback skipped: no key]"
        
    client = TavilyClient(api_key=api_key)
    try:
        res = client.search(query=query, max_results=max_results)
        lines: list[str] = []
        for item in res.get("results", []):
            lines.append(f"- {item['title']} ({item['url']})\n  {item['content']}")
        tavily_results = "\n".join(lines) if lines else "No results."
        return f"[Source: Tavily (DDG fallback: {ddg_results})]\n{tavily_results}"
    except (UsageLimitExceededError, ForbiddenError) as e:
        return f"[Source: DuckDuckGo (Empty)]\n{ddg_results}\n[Tavily fallback failed: limit/forbidden: {e}]"
    except Exception as e:
        return f"[Source: DuckDuckGo (Empty)]\n{ddg_results}\n[Tavily fallback failed: {e}]"
