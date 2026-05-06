from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from tenacity import AsyncRetrying, Retrying, retry_if_exception, stop_after_attempt, wait_exponential
from lilith_agent.config import Config

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = None

try:
    from google.genai.errors import ClientError as GenAIClientError
except ImportError:
    GenAIClientError = None

try:
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    AnthropicRateLimitError = None

try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = None


def is_retryable_rate_limit(exc: BaseException) -> bool:
    if ResourceExhausted is not None and isinstance(exc, ResourceExhausted):
        return True
    if GenAIClientError is not None and isinstance(exc, GenAIClientError):
        return getattr(exc, "code", None) == 429
    if AnthropicRateLimitError is not None and isinstance(exc, AnthropicRateLimitError):
        return True
    if OpenAIRateLimitError is not None and isinstance(exc, OpenAIRateLimitError):
        return True
    return False


def _tenacity_sleep(seconds: float) -> None:
    time.sleep(seconds)


async def _async_tenacity_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _base_retry_params() -> dict[str, Any]:
    return dict(
        retry=retry_if_exception(is_retryable_rate_limit),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: log.warning(
            f"LLM Rate Limit (429) hit. Retrying in {retry_state.next_action.sleep}s... "
            f"(Attempt {retry_state.attempt_number}/5)"
        ),
        reraise=True,
    )


def _sync_retry_params() -> dict[str, Any]:
    return {**_base_retry_params(), "sleep": _tenacity_sleep}


def _async_retry_params() -> dict[str, Any]:
    return {**_base_retry_params(), "sleep": _async_tenacity_sleep}

try:
    from langchain_community.cache import SQLiteCache
except ImportError:
    try:
        from langchain.cache import SQLiteCache
    except ImportError:
        SQLiteCache = None

try:
    from langchain_core.globals import set_llm_cache
except ImportError:
    try:
        from langchain.globals import set_llm_cache
    except ImportError:
        set_llm_cache = None

# Enable LLM cache for development efficiency
if SQLiteCache:
    try:
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    except Exception as e:
        # Fail gracefully if sqlite is missing or DB file locked
        pass

log = logging.getLogger(__name__)

LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1"
_NO_THINK = "/no_think"
_GEMINI_COOLDOWN_MODELS = {"gemini-3-flash-preview", "gemini-3.1-pro"}
_COOLDOWN_LADDER_SECONDS = (60, 120, 300)
_QUESTION_STREAK_LIMIT = 50
_cooldown_until: dict[tuple[str, str], float] = {}
_rate_limit_exhaustions: dict[tuple[str, str], int] = {}
_question_rate_limit_streak: ContextVar[int | None] = ContextVar("question_rate_limit_streak", default=None)


@dataclass
class RateLimitCooldownError(Exception):
    provider: str
    model: str
    cooldown_seconds: int
    original_error: str

    def __str__(self) -> str:
        return (
            f"rate limited provider={self.provider} model={self.model} "
            f"cooldown={self.cooldown_seconds}s original={self.original_error}"
        )


@dataclass
class BatchAbortRateLimitError(Exception):
    reason: str
    original_error: str

    def __str__(self) -> str:
        return f"batch abort rate limit: {self.reason}; original={self.original_error}"


@dataclass
class QuestionRateLimitStreakError(Exception):
    count: int

    def __str__(self) -> str:
        return f"question hit {self.count} consecutive rate-limit events"


def _reset_rate_limit_state_for_tests() -> None:
    _cooldown_until.clear()
    _rate_limit_exhaustions.clear()


@contextmanager
def rate_limit_question_scope():
    token = _question_rate_limit_streak.set(0)
    try:
        yield
    finally:
        _question_rate_limit_streak.reset(token)


def record_rate_limit_observation(exc: BaseException) -> None:
    current = _question_rate_limit_streak.get()
    if current is None or not is_retryable_rate_limit(exc):
        return
    current += 1
    _question_rate_limit_streak.set(current)
    if current >= _QUESTION_STREAK_LIMIT:
        raise QuestionRateLimitStreakError(count=current) from exc


def record_rate_limit_success() -> None:
    if _question_rate_limit_streak.get() is not None:
        _question_rate_limit_streak.set(0)


def _gemini_lane(provider: str | None, model_name: str | None) -> tuple[str, str] | None:
    if provider == "google" and model_name in _GEMINI_COOLDOWN_MODELS:
        return (provider, model_name)
    return None


def _cooldown_seconds_for_exhaustion(count: int) -> int:
    idx = min(max(count, 1), len(_COOLDOWN_LADDER_SECONDS)) - 1
    return _COOLDOWN_LADDER_SECONDS[idx]


def _sleep_active_cooldown(lane: tuple[str, str] | None) -> None:
    if lane is None:
        return
    remaining = _cooldown_until.get(lane, 0.0) - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def _record_success(lane: tuple[str, str] | None) -> None:
    if lane is None:
        return
    _rate_limit_exhaustions[lane] = 0
    _cooldown_until.pop(lane, None)


def _record_exhausted_rate_limit(lane: tuple[str, str], exc: BaseException) -> RateLimitCooldownError:
    count = _rate_limit_exhaustions.get(lane, 0) + 1
    _rate_limit_exhaustions[lane] = count
    cooldown = _cooldown_seconds_for_exhaustion(count)
    _cooldown_until[lane] = time.monotonic() + cooldown
    return RateLimitCooldownError(
        provider=lane[0],
        model=lane[1],
        cooldown_seconds=cooldown,
        original_error=str(exc),
    )


def _iter_error_details(exc: BaseException) -> list[dict[str, Any]]:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        nested = details.get("error", {}).get("details", details.get("details", []))
        return nested if isinstance(nested, list) else []
    return details if isinstance(details, list) else []


def _parse_retry_delay_seconds(value: Any) -> float | None:
    if isinstance(value, str) and value.endswith("s"):
        try:
            return float(value[:-1])
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _batch_abort_reason(exc: BaseException) -> str | None:
    for detail in _iter_error_details(exc):
        retry_delay = _parse_retry_delay_seconds(detail.get("retryDelay"))
        if retry_delay is not None and retry_delay > 600:
            return f"retry delay {retry_delay:g}s exceeds batch threshold"
        violations = detail.get("violations", [])
        if isinstance(violations, list):
            for violation in violations:
                if isinstance(violation, dict) and "PerDay" in str(violation.get("quotaId", "")):
                    return f"daily quota exhausted: {violation.get('quotaId')}"
        if "PerDay" in str(detail.get("quotaId", "")):
            return f"daily quota exhausted: {detail.get('quotaId')}"
    return None


class _NoThinkWrapper(BaseChatModel):
    """Wraps a ChatOpenAI model to append /no_think to every HumanMessage.

    Qwen3 Instruct in LM Studio enters thinking (chain-of-thought) mode by
    default, producing a huge reasoning_content blob and empty content.
    Appending /no_think disables it per Qwen3's chat template spec.
    """

    inner: ChatOpenAI
    model_name: str

    @property
    def _llm_type(self) -> str:
        return "lmstudio-no-think"

    def _inject(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        if "qwen" not in self.model_name.lower():
            return list(messages)
            
        out = []
        for msg in messages:
            if isinstance(msg, HumanMessage) and not str(msg.content).endswith(_NO_THINK):
                content = str(msg.content) + f" {_NO_THINK}"
                out.append(HumanMessage(content=content))
            else:
                out.append(msg)
        return out

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self.inner._generate(self._inject(messages), stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await self.inner._agenerate(self._inject(messages), stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any):
        # Delegate tool binding to inner model; return a wrapper that injects /no_think
        bound = self.inner.bind_tools(tools, **kwargs)
        wrapper = _BoundNoThinkWrapper(bound=bound, inject=self._inject)
        return wrapper


class _BoundNoThinkWrapper(Runnable):
    """Thin Runnable wrapper around a tool-bound model that injects /no_think."""

    def __init__(self, bound, inject):
        self._bound = bound
        self._inject = inject

    def invoke(self, input, config=None, **kwargs):
        return self._bound.invoke(self._inject(input), config=config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        return await self._bound.ainvoke(self._inject(input), config=config, **kwargs)

    def __getattr__(self, name):
        return getattr(self._bound, name)


class _RetryWrapper(BaseChatModel):
    """Wraps any BaseChatModel to apply exponential backoff on 429 errors."""

    inner: BaseChatModel
    provider: str | None = None
    model_name: str | None = None

    @property
    def _llm_type(self) -> str:
        return f"retry-{self.inner._llm_type}"

    def _generate(self, *args, **kwargs):
        lane = _gemini_lane(self.provider, self.model_name)
        _sleep_active_cooldown(lane)
        try:
            for attempt in Retrying(**_sync_retry_params()):
                with attempt:
                    try:
                        result = self.inner._generate(*args, **kwargs)
                    except Exception as observed:
                        record_rate_limit_observation(observed)
                        raise
                    record_rate_limit_success()
                    _record_success(lane)
                    return result
        except Exception as exc:
            if lane is not None and is_retryable_rate_limit(exc):
                reason = _batch_abort_reason(exc)
                if reason is not None:
                    raise BatchAbortRateLimitError(reason=reason, original_error=str(exc)) from exc
                raise _record_exhausted_rate_limit(lane, exc) from exc
            raise

    async def _agenerate(self, *args, **kwargs):
        async for attempt in AsyncRetrying(**_async_retry_params()):
            with attempt:
                return await self.inner._agenerate(*args, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any):
        bound = self.inner.bind_tools(tools, **kwargs)
        # We need to wrap the resulting bound tool caller's invoke method too
        return _BoundRetryWrapper(bound=bound)


class _BoundRetryWrapper(Runnable):
    """Wraps a tool-bound Runnable to apply retry logic to .invoke()."""

    def __init__(self, bound):
        self._bound = bound

    def invoke(self, input, config=None, **kwargs):
        for attempt in Retrying(**_sync_retry_params()):
            with attempt:
                try:
                    result = self._bound.invoke(input, config=config, **kwargs)
                except Exception as observed:
                    record_rate_limit_observation(observed)
                    raise
                record_rate_limit_success()
                return result

    async def ainvoke(self, input, config=None, **kwargs):
        async for attempt in AsyncRetrying(**_async_retry_params()):
            with attempt:
                return await self._bound.ainvoke(input, config=config, **kwargs)

    def __getattr__(self, name):
        return getattr(self._bound, name)


def _build(provider: str, model: str, cfg: Config) -> BaseChatModel:
    log.info("Building model for provider=%r, model=%r", provider, model)
    max_tokens = cfg.max_tokens
    
    # helper to wrap final model
    def _wrap(m):
        return _RetryWrapper(inner=m, provider=provider, model_name=model)

    if provider == "ollama":
        return _wrap(ChatOllama(model=model, num_predict=max_tokens))
    if provider == "anthropic":
        return _wrap(ChatAnthropic(model=model, api_key=cfg.anthropic_api_key, max_tokens=max_tokens))
    if provider == "google":
        from langchain_google_genai import HarmBlockThreshold, HarmCategory
        
        # Suppress all safety filters to avoid silent empty responses on academic tasks
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        return _wrap(ChatGoogleGenerativeAI(
            model=model, 
            google_api_key=cfg.google_api_key, 
            max_output_tokens=max_tokens,
            safety_settings=safety_settings
        ))
    if provider == "huggingface":
        endpoint = HuggingFaceEndpoint(
            repo_id=model, huggingfacehub_api_token=cfg.huggingface_api_key, max_new_tokens=max_tokens
        )
        return _wrap(ChatHuggingFace(llm=endpoint))
    if provider == "lmstudio":
        # LM Studio exposes an OpenAI-compatible API.
        inner = ChatOpenAI(
            model=model,
            base_url=cfg.lmstudio_base_url or LMSTUDIO_DEFAULT_BASE_URL,
            api_key="lm-studio",
            temperature=0,
            max_tokens=max_tokens,
        )
        # Wrap with /no_think injection to disable Qwen3 chain-of-thought mode
        # Then wrap the result with retry logic
        wrapped = _NoThinkWrapper(inner=inner, model_name=model)
        return _wrap(wrapped)
    raise ValueError(f"Unknown provider: {provider}")


def get_cheap_model(cfg: Config) -> BaseChatModel:
    return _build(cfg.cheap_provider, cfg.cheap_model, cfg)


def get_strong_model(cfg: Config) -> BaseChatModel:
    return _build(cfg.strong_provider, cfg.strong_model, cfg)


def get_extra_strong_model(cfg: Config) -> BaseChatModel:
    return _build(cfg.extra_strong_provider, cfg.extra_strong_model, cfg)
