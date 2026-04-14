import io
from pathlib import Path

import httpx
from pypdf import PdfReader

from lilith_agent.tools.vision import inspect_visual_content


def inspect_pdf(url_or_path: str, query: str) -> str:
    """Extract text from a PDF (local or remote). If query asks for visual/layout details, consider vision analysis."""
    try:
        if url_or_path.startswith(("http://", "https://")):
            resp = httpx.get(url_or_path, follow_redirects=True, timeout=30.0)
            resp.raise_for_status()
            data = resp.content
        else:
            path = Path(url_or_path)
            if not path.exists():
                return f"File not found: {url_or_path}"
            data = path.read_bytes()

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        
        # Heuristic: if text is very short OR query mentions 'image', 'diagram', 'chart', 'layout'
        # we might want to use vision. But let's start with text.
        if not text.strip() or any(k in query.lower() for k in ["diagram", "chart", "figure", "image", "visual"]):
            # Fallback to vision for the first few pages? 
            # For GAIA, usually text extraction is enough or it's a visual task.
            pass
            
        return text if text.strip() else "PDF contains no extractable text. It might be an image-only PDF."
    except Exception as e:
        return f"Error inspecting PDF: {e}"
