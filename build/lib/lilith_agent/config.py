from __future__ import annotations
from dataclasses import dataclass
import os

def _get_int_env(key: str, default: str) -> int:
    val = os.getenv(key, default)
    if not val or not val.strip():
        return int(default)
    try:
        return int(val)
    except ValueError:
        return int(default)


def _get_float_env(key: str, default: str) -> float:
    val = os.getenv(key, default)
    if not val or not val.strip():
        return float(default)
    try:
        return float(val)
    except ValueError:
        return float(default)

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
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    agent_model_tier: str = "strong"
    agent_provider: str = ""
    agent_model: str = ""
    caveman: bool = False
    caveman_mode: str = "full"
    recursion_limit: int = 100
    budget_hard_cap: int = 25
    budget_warn_at: int = 15
    semantic_dedup_threshold: float = 0.5
    compact_summarize: bool = True
    llm_formatter_enabled: bool = True
    answer_contract_enabled: bool = True
    give_up_recovery_enabled: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        agent_model_tier = os.getenv("GAIA_AGENT_MODEL_TIER", "strong").strip().lower()
        if agent_model_tier not in {"cheap", "strong"}:
            raise ValueError(
                "GAIA_AGENT_MODEL_TIER must be one of: cheap, strong"
            )

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
            deepseek_api_key=os.getenv("GAIA_DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")),
            deepseek_base_url=os.getenv("GAIA_DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            agent_model_tier=agent_model_tier,
            agent_provider=os.getenv("GAIA_AGENT_PROVIDER", "").strip(),
            agent_model=os.getenv("GAIA_AGENT_MODEL", "").strip(),
            max_tokens=_get_int_env("GAIA_MAX_TOKENS", "65536"),
            caveman=os.getenv("GAIA_CAVEMAN", "true").lower() == "true",
            caveman_mode=os.getenv("GAIA_CAVEMAN_MODE", "full"),
            recursion_limit=_get_int_env("GAIA_RECURSION_LIMIT", "50"),
            budget_hard_cap=_get_int_env("GAIA_BUDGET_HARD_CAP", "25"),
            budget_warn_at=_get_int_env("GAIA_BUDGET_WARN_AT", "15"),
            semantic_dedup_threshold=_get_float_env("GAIA_SEMANTIC_DEDUP_THRESHOLD", "0.5"),
            compact_summarize=os.getenv("GAIA_COMPACT_SUMMARIZE", "true").lower() == "true",
            llm_formatter_enabled=os.getenv("GAIA_LLM_FORMATTER_ENABLED", "true").lower() == "true",
            answer_contract_enabled=os.getenv("GAIA_ANSWER_CONTRACT_ENABLED", "true").lower() == "true",
            give_up_recovery_enabled=os.getenv("GAIA_GIVE_UP_RECOVERY_ENABLED", "true").lower() == "true",
        )
