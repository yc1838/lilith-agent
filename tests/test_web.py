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
