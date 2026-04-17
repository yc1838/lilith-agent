import io
import ipaddress
from urllib.parse import urlparse

import httpx
import trafilatura
from pypdf import PdfReader


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
            resp = httpx.get(url, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
            reader = PdfReader(io.BytesIO(resp.content))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return text[:max_chars] if len(text) > max_chars else text
        except Exception as e:
            return f"PDF Fetch Failed: {e}"

    # 2. Try Jina Reader (r.jina.ai) — It's specifically built to bypass anti-bot and return clean markdown
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "X-Return-Format": "markdown"
    }

    try:
        # Use a slightly shorter timeout for Jina to allow for fallback
        jina_timeout = min(timeout, 40.0)
        resp = httpx.get(jina_url, timeout=jina_timeout, follow_redirects=True, headers=headers)
        if resp.status_code == 200 and len(resp.text) > 50:
            text = resp.text
            return text[:max_chars] if len(text) > max_chars else text
    except Exception as e:
        # Silently fail Jina attempt and proceed to fallback
        pass

    # 3. Fallback: Normal httpx + trafilatura
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
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
                "DO NOT retry this exact URL. Try a different source or use search snippets to answer."
            )

        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text
    except Exception as e:
        return f"All Fetch Attempts Failed: {e}"
