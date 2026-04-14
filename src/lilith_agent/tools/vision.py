import base64
import logging
import os
from pathlib import Path

import httpx
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from lilith_agent.config import Config

log = logging.getLogger(__name__)

_vision_session_failed: bool = False

def reset_vision_state():
    global _vision_session_failed
    _vision_session_failed = False
def inspect_visual_content(file_path_or_url: str, prompt: str, cfg: Config) -> str:
    """Analyze image/video with remapping and prioritized fallback logic."""
    global _vision_session_failed
    if _vision_session_failed:
        return ("Vision services are unavailable for this session. "
                "All providers failed on a previous attempt. "
                "Try alternative approaches (web_search for image descriptions, etc.).")
    
    # 1. Model Remapping Table
    MODEL_MAP = {
        "claude-sonnet-4-6": "claude-3-5-sonnet-20240620",
        "llava": "fal-ai/moondream-next", # Accurate & Fast default
        "llava-1.5": "fal-ai/llava-next",
        "fal-ai/llava/v1.5/7b": "fal-ai/llava-next", # Autocorrect user string
        "gemini-flash": "gemini-3.0-flash",
    }
    
    requested_model = cfg.vision_model
    target_model = MODEL_MAP.get(requested_model, requested_model)
    requested_provider = cfg.vision_provider

    def _call_vision(model_name: str, provider: str, b64_data: str, mime_type: str, content: list) -> str:
        # SMART PROVISION: If model name looks like a FAL model, force FAL provider
        actual_provider = provider
        if model_name.startswith("fal-ai/"):
            actual_provider = "fal"
            
        if actual_provider == "fal":
            try:
                import fal_client
                os.environ["FAL_KEY"] = cfg.fal_vision_api_key
                image_data_url = f"data:{mime_type};base64,{b64_data}"
                log.info(f"[vision] Attempting FAL | Model: {model_name}")
                result = fal_client.subscribe(
                    model_name,
                    arguments={"prompt": prompt, "image_url": image_data_url}
                )
                return result.get("text") or result.get("description") or str(result)
            except Exception as e:
                log.warning(f"[vision] FAL call failed for {model_name}: {e}")
                return f"ERROR:{e}"
        else:
            # Default Google
            try:
                log.info(f"[vision] Attempting Google | Model: {model_name}")
                # Ensure we use a clean model ID for Google
                clean_model = model_name.replace("models/", "")
                model = ChatGoogleGenerativeAI(model=clean_model, google_api_key=cfg.google_api_key)
                response = model.invoke([HumanMessage(content=content)])
                return str(response.content)
            except Exception as e:
                log.warning(f"[vision] Google call failed for {model_name}: {e}")
                return f"ERROR:{e}"

    # Preparation
    content = [{"type": "text", "text": prompt}]
    try:
        if file_path_or_url.startswith(("http://", "https://")):
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            resp = httpx.get(file_path_or_url, headers=headers, follow_redirects=True, timeout=30.0)
            resp.raise_for_status()
            data = resp.content
            mime_type = resp.headers.get("Content-Type", "image/png").split(";")[0].strip()
            if not (mime_type.startswith("image/") or mime_type.startswith("video/")):
                return (f"URL did not return an image/video (got {mime_type}). "
                        f"The URL likely points to an HTML page, not a direct media file. "
                        f"Find a direct image URL (ending in .jpg/.png/.webp) instead.")
        else:
            path = Path(file_path_or_url)
            data = path.read_bytes()
            mime_type = "video/mp4" if path.suffix.lower() == ".mp4" else "image/png"
        b64_data = base64.b64encode(data).decode("utf-8")
        content.append({"type": "media", "mime_type": mime_type, "data": b64_data})
    except Exception as e:
        log.error(f"[vision] File preparation failed: {e}")
        return f"File Preparation Failed: {e}"

    # 2. Attempt 1: Targeted Model on Configured Provider
    res = _call_vision(target_model, requested_provider, b64_data, mime_type, content)
    if not res.startswith("ERROR:"):
        return res

    # 3. Attempt 2: Stable Provider-Specific Fallback
    if requested_provider == "fal":
        # If the primary FAL model failed, try a very stable one on FAL
        fallback_model = "fal-ai/moondream2" 
        res2 = _call_vision(fallback_model, "fal", b64_data, mime_type, content)
    else:
        # If the primary Google model failed, try the stable Flash model
        res2 = _call_vision("gemini-3-flash-preview", "google", b64_data, mime_type, content)
    
    if not res2.startswith("ERROR:"):
        return res2

    # 4. Attempt 3: Cross-Provider Ultimate Last Resort (Gemini Flash)
    if requested_provider == "fal":
        log.warning("[vision] All FAL attempts failed. Triggering ultimate last resort: gemini-3-flash-preview on Google")
        ultimate_res = _call_vision("gemini-3-flash-preview", "google", b64_data, mime_type, content)
        if not ultimate_res.startswith("ERROR:"):
            return ultimate_res
        _vision_session_failed = True
        return f"All Vision Attempts Failed. Final Error: {ultimate_res}"
    
    _vision_session_failed = True
    return f"All Vision Attempts Failed. Final Error: {res2}"
