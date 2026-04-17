import io
import ipaddress
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from pypdf import PdfReader

_FETCH_RECOVERY_HINT = (
    "Try: (1) web_search for third-party citations or mirrors of this content; "
    "(2) fetch the archived version at https://web.archive.org/web/<url>; "
    "(3) use the snippet from your prior web_search result instead of fetching the page."
)

_MAX_REDIRECTS = 5


def _safe_get(url: str, **kwargs) -> httpx.Response:
    """GET with bounded manual redirect following; SSRF-rechecks every hop.

    The initial `_is_safe_http_url` check only validates the first URL; a
    server can still 302 to a private/metadata address. This helper follows
    redirects one hop at a time, re-running the static SSRF check before each
    request, and raises `httpx.HTTPError` if any hop resolves to an unsafe URL
    or the redirect budget is exceeded.
    """
    kwargs.pop("follow_redirects", None)
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        ok, reason = _is_safe_http_url(current)
        if not ok:
            raise httpx.HTTPError(f"redirect to unsafe URL blocked ({reason})")
        resp = httpx.get(current, follow_redirects=False, **kwargs)
        if not resp.is_redirect:
            return resp
        location = resp.headers.get("Location")
        if not location:
            return resp
        current = urljoin(current, location)
    raise httpx.HTTPError(f"too many redirects (> {_MAX_REDIRECTS})")


def _is_safe_http_url(url: str) -> tuple[bool, str]:
    """SSRF guard: allow only http/https to public hosts.

    Returns (ok, reason). Blocks file://, ftp://, javascript:, etc; and blocks
    loopback, RFC1918 private ranges, link-local (incl. cloud metadata
    169.254.169.254), and IPv6 equivalents. Hostnames without a literal IP
    are allowed — DNS rebinding is a residual risk mitigated at fetch time
    by `httpx` following redirects; a second check runs after the request
    lands, but for now a hostname passes the static check.
    """
    if not url or not isinstance(url, str):
        return False, "empty or non-string url"

    try:
        parsed = urlparse(url)
    except Exception as exc:
        return False, f"unparseable url: {exc}"

    if parsed.scheme not in {"http", "https"}:
        return False, f"blocked scheme {parsed.scheme!r}; only http/https allowed"

    host = parsed.hostname
    if not host:
        return False, "missing host"

    host_lower = host.lower()
    if host_lower in {"localhost", "ip6-localhost", "ip6-loopback"}:
        return False, "loopback hostname blocked"

    # If host is a literal IP (v4 or v6), check the range.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is not None:
        if ip.is_loopback:
            return False, "loopback address blocked"
        if ip.is_link_local:
            return False, "link-local address blocked (incl. cloud metadata)"
        if ip.is_private:
            return False, "private address blocked"
        if ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False, "reserved/multicast/unspecified address blocked"

    return True, ""


def fetch_url(url: str, max_chars: int = 8000, timeout: float = 60.0) -> str:
    ok, reason = _is_safe_http_url(url)
    if not ok:
        return f"WEB_FETCH_ERROR: unsafe URL blocked — {reason}"

    # 1. Check for PDF - Local extraction is always more reliable for PDFs
    if url.lower().endswith(".pdf"):
        try:
            resp = _safe_get(url, timeout=timeout)
            resp.raise_for_status()
            reader = PdfReader(io.BytesIO(resp.content))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return text[:max_chars] if len(text) > max_chars else text
        except Exception as e:
            return f"WEB_FETCH_ERROR: PDF fetch failed for {url}: {e}. " + _FETCH_RECOVERY_HINT

    # 2. Try Jina Reader (r.jina.ai) — It's specifically built to bypass anti-bot and return clean markdown
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "X-Return-Format": "markdown"
    }

    try:
        # Use a slightly shorter timeout for Jina to allow for fallback
        jina_timeout = min(timeout, 40.0)
        resp = _safe_get(jina_url, timeout=jina_timeout, headers=headers)
        if resp.status_code == 200 and len(resp.text) > 50:
            text = resp.text
            return text[:max_chars] if len(text) > max_chars else text
    except Exception as e:
        # Silently fail Jina attempt and proceed to fallback
        pass

    # 3. Fallback: Normal httpx + trafilatura
    try:
        resp = _safe_get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        resp.raise_for_status()
        
        # Final PDF check in case headers reveal it later
        content_type = resp.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type:
            reader = PdfReader(io.BytesIO(resp.content))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            extracted = trafilatura.extract(resp.text, include_comments=False, include_tables=True)
            text = extracted or resp.text

        # 4. CAPTCHA / Anti-Bot Detection
        captcha_markers = [
            "verification required", "captcha", "anti-robot", "check connecting",
            "bot detection", "robot check", "friendly captcha"
        ]
        text_lower = text.lower()
        if any(marker in text_lower for marker in captcha_markers):
            return (
                "WEB_FETCH_ERROR: CAPTCHA/Bot detection encountered. This URL is currently inaccessible via automated tools. "
                "DO NOT retry this exact URL. " + _FETCH_RECOVERY_HINT
            )

        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text
    except Exception as e:
        return f"WEB_FETCH_ERROR: all fetch attempts failed for {url}: {e}. " + _FETCH_RECOVERY_HINT
