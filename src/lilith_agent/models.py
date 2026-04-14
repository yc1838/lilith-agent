from __future__ import annotations

from typing import Any, Sequence

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, Retrying, AsyncRetrying
from lilith_agent.config import Config

# Identify retryable exceptions (429s) across providers
RETRY_EXCEPTIONS = []
try:
    from google.api_core.exceptions import ResourceExhausted
    RETRY_EXCEPTIONS.append(ResourceExhausted)
except ImportError:
    pass

try:
    from anthropic import RateLimitError as AnthropicRateLimitError
    RETRY_EXCEPTIONS.append(AnthropicRateLimitError)
except ImportError:
    pass

try:
    from openai import RateLimitError as OpenAIRateLimitError
    RETRY_EXCEPTIONS.append(OpenAIRateLimitError)
except ImportError:
    pass

RETRY_EXCEPTIONS = tuple(RETRY_EXCEPTIONS)

# Shared retry configuration for context managers
_RETRY_PARAMS = dict(
    retry=retry_if_exception_type(RETRY_EXCEPTIONS) if RETRY_EXCEPTIONS else lambda e: False,
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: log.warning(
        f"LLM Rate Limit (429) hit. Retrying in {retry_state.next_action.sleep}s... "
        f"(Attempt {retry_state.attempt_number}/5)"
    ),
    reraise=True
)

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

    @property
    def _llm_type(self) -> str:
        return f"retry-{self.inner._llm_type}"

    def _generate(self, *args, **kwargs):
        for attempt in Retrying(**_RETRY_PARAMS):
            with attempt:
                return self.inner._generate(*args, **kwargs)

    async def _agenerate(self, *args, **kwargs):
        async for attempt in AsyncRetrying(**_RETRY_PARAMS):
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
        for attempt in Retrying(**_RETRY_PARAMS):
            with attempt:
                return self._bound.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        async for attempt in AsyncRetrying(**_RETRY_PARAMS):
            with attempt:
                return await self._bound.ainvoke(input, config=config, **kwargs)

    def __getattr__(self, name):
        return getattr(self._bound, name)


def _build(provider: str, model: str, cfg: Config) -> BaseChatModel:
    log.info("Building model for provider=%r, model=%r", provider, model)
    max_tokens = cfg.max_tokens
    
    # helper to wrap final model
    def _wrap(m):
        return _RetryWrapper(inner=m)

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
