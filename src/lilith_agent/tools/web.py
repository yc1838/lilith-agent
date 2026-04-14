import io
import httpx
import trafilatura
from pypdf import PdfReader


def fetch_url(url: str, max_chars: int = 8000, timeout: float = 60.0) -> str:
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

        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text
    except Exception as e:
        return f"All Fetch Attempts Failed: {e}"
