from __future__ import annotations

import pytest

from lilith_agent.tools.web import _is_safe_http_url, fetch_url


@pytest.mark.parametrize("url", [
    "https://example.com",
    "http://example.com/path?q=1",
    "https://en.wikipedia.org/wiki/Python",
    "https://example.com:8080/",
])
def test_public_https_urls_allowed(url):
    ok, _ = _is_safe_http_url(url)
    assert ok is True


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/secret",
    "javascript:alert(1)",
    "data:text/html,<script>",
    "gopher://example.com",
    "",
    "not a url",
])
def test_non_http_schemes_rejected(url):
    ok, reason = _is_safe_http_url(url)
    assert ok is False
    assert reason


@pytest.mark.parametrize("host", [
    "localhost",
    "127.0.0.1",
    "127.1.2.3",
    "0.0.0.0",
    "10.0.0.1",
    "10.255.255.255",
    "172.16.0.1",
    "172.31.255.255",
    "192.168.1.1",
    "169.254.169.254",  # AWS/GCP metadata
    "169.254.0.1",
    "[::1]",
    "[fe80::1]",
    "[fc00::1]",
])
def test_private_or_metadata_hosts_rejected(host):
    url = f"http://{host}/"
    ok, reason = _is_safe_http_url(url)
    assert ok is False, f"expected rejection for {host}"
    assert "private" in reason.lower() or "loopback" in reason.lower() or "metadata" in reason.lower() or "link-local" in reason.lower()


def test_public_ranges_not_rejected():
    # 8.8.8.8, 1.1.1.1 are legitimate public IPs
    ok, _ = _is_safe_http_url("http://8.8.8.8/")
    assert ok is True


def test_fetch_url_rejects_unsafe_scheme_without_network_call():
    out = fetch_url("file:///etc/passwd")
    assert out.startswith("WEB_FETCH_ERROR")
    assert "scheme" in out.lower() or "unsafe" in out.lower() or "blocked" in out.lower()


def test_fetch_url_rejects_private_address_without_network_call():
    out = fetch_url("http://169.254.169.254/latest/meta-data/")
    assert out.startswith("WEB_FETCH_ERROR")


class _FakeResp:
    def __init__(self, status_code=200, text="", headers=None):
        import httpx as _httpx

        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = headers or {"Content-Type": "text/html"}
        self._httpx = _httpx

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise self._httpx.HTTPStatusError(
                f"{self.status_code} error", request=None, response=None
            )


def test_fetch_url_captcha_response_includes_recovery_hints(monkeypatch):
    import httpx as _httpx

    def fake_get(url, *args, **kwargs):
        # Jina path fails (non-200), fallback path returns CAPTCHA page
        if url.startswith("https://r.jina.ai/"):
            return _FakeResp(status_code=403, text="")
        return _FakeResp(
            status_code=200,
            text="<html><body>Please complete the captcha to continue</body></html>",
        )

    monkeypatch.setattr(_httpx, "get", fake_get)

    out = fetch_url("https://example.com/blocked")
    lower = out.lower()
    assert "captcha" in lower or "bot detection" in lower
    assert "third-party" in lower or "citation" in lower
    assert "archive.org" in lower
    assert "snippet" in lower


def test_fetch_url_exception_path_includes_recovery_hints(monkeypatch):
    import httpx as _httpx

    def fake_get(url, *args, **kwargs):
        raise _httpx.HTTPError("403 Forbidden")

    monkeypatch.setattr(_httpx, "get", fake_get)

    out = fetch_url("https://example.com/blocked")
    assert out.startswith("WEB_FETCH_ERROR")
    lower = out.lower()
    assert "third-party" in lower or "citation" in lower
    assert "archive.org" in lower
    assert "snippet" in lower


def test_fetch_url_pdf_failure_includes_recovery_hints(monkeypatch):
    import httpx as _httpx

    def fake_get(url, *args, **kwargs):
        raise _httpx.HTTPError("403 Forbidden")

    monkeypatch.setattr(_httpx, "get", fake_get)

    out = fetch_url("https://example.com/paper.pdf")
    assert out.startswith("WEB_FETCH_ERROR")
    lower = out.lower()
    assert "third-party" in lower or "citation" in lower
    assert "archive.org" in lower
    assert "snippet" in lower


def test_fetch_url_blocks_redirect_to_private_ip(monkeypatch):
    """Post-redirect SSRF re-check: server sends 302 to link-local metadata IP."""
    import httpx as _httpx

    visited: list[str] = []

    def fake_get(url, *args, **kwargs):
        visited.append(url)
        # Redirect every request to the AWS metadata IP.
        return _FakeResp(
            status_code=302,
            headers={"Location": "http://169.254.169.254/latest/meta-data/iam/"},
        )

    monkeypatch.setattr(_httpx, "get", fake_get)

    out = fetch_url("https://example.com/bounce")
    assert out.startswith("WEB_FETCH_ERROR")
    # Agent must never have actually requested the private IP.
    assert not any("169.254.169.254" in u for u in visited)


def test_fetch_url_allows_redirect_to_public_hostname(monkeypatch):
    """A legitimate 302 to a public hostname must still work end-to-end."""
    import httpx as _httpx

    def fake_get(url, *args, **kwargs):
        # Jina path: fail so fallback path exercises the redirect logic.
        if url.startswith("https://r.jina.ai/"):
            return _FakeResp(status_code=403, text="")
        if url == "https://example.com/start":
            return _FakeResp(
                status_code=302,
                headers={"Location": "https://www.example.com/final"},
            )
        if url == "https://www.example.com/final":
            return _FakeResp(
                status_code=200,
                text="<html><body>hello world from final page</body></html>",
            )
        return _FakeResp(status_code=404, text="")

    monkeypatch.setattr(_httpx, "get", fake_get)

    out = fetch_url("https://example.com/start")
    assert not out.startswith("WEB_FETCH_ERROR")
    assert "hello world" in out.lower() or "final page" in out.lower()


def test_fetch_url_blocks_redirect_loop(monkeypatch):
    """Bounded redirect follower stops instead of looping forever."""
    import httpx as _httpx

    calls = {"n": 0}

    def fake_get(url, *args, **kwargs):
        if url.startswith("https://r.jina.ai/"):
            return _FakeResp(status_code=403, text="")
        calls["n"] += 1
        return _FakeResp(
            status_code=302,
            headers={"Location": "https://example.com/next"},
        )

    monkeypatch.setattr(_httpx, "get", fake_get)

    out = fetch_url("https://example.com/start")
    assert out.startswith("WEB_FETCH_ERROR")
    # Must have stopped — not burning arbitrary calls.
    assert calls["n"] <= 10
