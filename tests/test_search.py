from __future__ import annotations

from lilith_agent.tools import search as search_mod


def test_web_search_empty_without_tavily_key_includes_recovery_hint(monkeypatch):
    monkeypatch.setattr(search_mod, "_ddg_search", lambda q, n: "No results.")

    out = search_mod.web_search("obscure unlikely query", api_key="")

    lower = out.lower()
    assert "no results" in lower
    assert "rephrase" in lower or "alternative keywords" in lower
    assert "archive.org" in lower or "arxiv" in lower or "crossref" in lower


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
