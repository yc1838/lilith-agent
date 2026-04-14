from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass
class Config:
    cheap_provider: str
    cheap_model: str
    strong_provider: str
    strong_model: str
    extra_strong_provider: str
    extra_strong_model: str
    vision_provider: str
    vision_model: str
    fal_vision_api_key: str
    api_url: str
    checkpoint_dir: str
    whisper_model: str
    anthropic_api_key: str
    google_api_key: str
    huggingface_api_key: str
    tavily_api_key: str
    lmstudio_base_url: str
    max_tokens: int
    max_json_repairs: int = 5
    caveman: bool = False
    caveman_mode: str = "full"
    recursion_limit: int = 50

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            cheap_provider=os.getenv("GAIA_CHEAP_PROVIDER", "google"),
            cheap_model=os.getenv("GAIA_CHEAP_MODEL", "gemini-3-flash-preview"),
            strong_provider=os.getenv("GAIA_STRONG_PROVIDER", "anthropic"),
            strong_model=os.getenv("GAIA_STRONG_MODEL", "claude-sonnet-4-6"),
            extra_strong_provider=os.getenv("GAIA_EXTRA_STRONG_PROVIDER", os.getenv("GAIA_STRONG_PROVIDER", "anthropic")),
            extra_strong_model=os.getenv("GAIA_EXTRA_STRONG_MODEL", os.getenv("GAIA_STRONG_MODEL", "claude-sonnet-4-6")),
            vision_provider=os.getenv("GAIA_VISION_PROVIDER", "fal"),
            vision_model=os.getenv("GAIA_VISION_MODEL", "gemini-3-flash-preview"),
            fal_vision_api_key=os.getenv("GAIA_FAL_VISION_API_KEY", ""),
            api_url=os.getenv("GAIA_API_URL", "https://agents-course-unit4-scoring.hf.space"),
            checkpoint_dir=os.getenv("GAIA_CHECKPOINT_DIR", ".checkpoints"),
            whisper_model=os.getenv("GAIA_WHISPER_MODEL", "base"),
            anthropic_api_key=os.getenv("GAIA_ANTHROPIC_API_KEY", ""),
            google_api_key=os.getenv("GAIA_GOOGLE_API_KEY", ""),
            huggingface_api_key=os.getenv("GAIA_HUGGINGFACE_API_KEY", ""),
            tavily_api_key=os.getenv("GAIA_TAVILY_API_KEY", ""),
            lmstudio_base_url=os.getenv("GAIA_LMSTUDIO_BASE_URL", ""),
            max_tokens=int(os.getenv("GAIA_MAX_TOKENS", "65536")),
            max_json_repairs=int(os.getenv("GAIA_MAX_JSON_REPAIRS", "5")),
            caveman=os.getenv("GAIA_CAVEMAN", "true").lower() == "true",
            caveman_mode=os.getenv("GAIA_CAVEMAN_MODE", "full"),
            recursion_limit=int(os.getenv("GAIA_RECURSION_LIMIT", "50")),
        )
