from __future__ import annotations

from tavily import TavilyClient


def tavily_search(query: str, api_key: str, max_results: int = 5) -> str:
    if not api_key:
        return "ERROR: Tavily API key not configured (set GAIA_TAVILY_API_KEY)."
    client = TavilyClient(api_key=api_key)
    res = client.search(query=query, max_results=max_results)
    lines: list[str] = []
    for item in res.get("results", []):
        lines.append(f"- {item['title']} ({item['url']})\n  {item['content']}")
    return "\n".join(lines) if lines else "No results."
