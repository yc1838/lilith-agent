from __future__ import annotations

from lilith_agent.tools import search as search_mod


def test_web_search_empty_without_tavily_key_includes_recovery_hint(monkeypatch):
    monkeypatch.setattr(search_mod, "_ddg_search", lambda q, n: "No results.")

    out = search_mod.web_search("obscure unlikely query", api_key="")

    lower = out.lower()
    assert "no results" in lower
    assert "rephrase" in lower or "alternative keywords" in lower
    assert "archive.org" in lower or "arxiv" in lower or "crossref" in lower


def test_web_search_prints_query_provider_and_result_preview(monkeypatch, capsys):
    monkeypatch.setattr(
        search_mod,
        "_ddg_search",
        lambda q, n: "- Result Title (https://example.com)\n  Useful snippet",
    )

    search_mod.web_search("moon landing transcript", api_key="", max_results=3)

    printed = capsys.readouterr().out
    assert "[web_search] start provider=duckduckgo query='moon landing transcript' max_results=3" in printed
    assert "[web_search] provider=duckduckgo status=success chars=" in printed
    assert "Result Title" in printed


def test_web_search_empty_with_tavily_failure_includes_recovery_hint(monkeypatch):
    monkeypatch.setattr(search_mod, "_ddg_search", lambda q, n: "No results.")

    class _BoomClient:
        def __init__(self, api_key):
            pass

        def search(self, **_kw):
            raise RuntimeError("tavily offline")

    monkeypatch.setattr(search_mod, "TavilyClient", _BoomClient)

    out = search_mod.web_search("obscure unlikely query", api_key="fake-key")

    lower = out.lower()
    assert "rephrase" in lower or "alternative keywords" in lower
    assert "archive.org" in lower or "arxiv" in lower or "crossref" in lower
