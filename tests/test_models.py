import pytest
from unittest.mock import AsyncMock

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from lilith_agent.models import (
    _RetryWrapper,
    _NoThinkWrapper,
    _BoundRetryWrapper,
    _BoundNoThinkWrapper,
    BatchAbortRateLimitError,
    QuestionRateLimitStreakError,
    RateLimitCooldownError,
    _reset_rate_limit_state_for_tests,
    batch_rate_limit_pause_seconds,
    clear_batch_rate_limit_window,
    is_retryable_rate_limit,
    rate_limit_question_scope,
)


class _FakeChatModel:
    """Minimal stand-in for a BaseChatModel exposing bind_tools."""

    _llm_type = "fake"

    def __init__(self):
        self.bound_with = None

    def bind_tools(self, tools, **kwargs):
        self.bound_with = tools
        return RunnableLambda(lambda msgs: AIMessage(content="ok"))


@pytest.fixture(autouse=True)
def _disable_retry_sleeps(monkeypatch):
    async def _no_async_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("lilith_agent.models._tenacity_sleep", lambda _seconds: None, raising=False)
    monkeypatch.setattr("lilith_agent.models._async_tenacity_sleep", _no_async_sleep, raising=False)


class _FailingGenerateModel:
    _llm_type = "failing"

    def __init__(self, exc: BaseException):
        self.exc = exc
        self.calls = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        raise self.exc

    async def _agenerate(self, *args, **kwargs):
        self.calls += 1
        raise self.exc

    def bind_tools(self, tools, **kwargs):
        def _raise(_msgs):
            raise self.exc

        return RunnableLambda(_raise)


class _SuccessfulGenerateModel:
    _llm_type = "success"

    def __init__(self):
        self.calls = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    async def _agenerate(self, *args, **kwargs):
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def bind_tools(self, tools, **kwargs):
        return RunnableLambda(lambda msgs: AIMessage(content="ok"))


def test_retry_wrapper_bind_tools_returns_runnable():
    inner = _FakeChatModel()
    wrapper = _RetryWrapper.model_construct(inner=inner)

    bound = wrapper.bind_tools([])

    assert isinstance(bound, _BoundRetryWrapper)
    assert isinstance(bound, Runnable), (
        "bind_tools() must return a Runnable so create_react_agent accepts it"
    )


def test_retry_wrapper_bound_invoke_passes_through():
    inner = _FakeChatModel()
    wrapper = _RetryWrapper.model_construct(inner=inner)

    bound = wrapper.bind_tools([])
    result = bound.invoke([("user", "hi")])

    assert isinstance(result, AIMessage)
    assert result.content == "ok"


def test_no_think_wrapper_bind_tools_returns_runnable():
    inner = _FakeChatModel()
    wrapper = _NoThinkWrapper.model_construct(inner=inner, model_name="qwen-test")

    bound = wrapper.bind_tools([])

    assert isinstance(bound, _BoundNoThinkWrapper)
    assert isinstance(bound, Runnable)


def _make_genai_client_error(code: int):
    pytest.importorskip("google.genai.errors")
    from google.genai.errors import ClientError

    return ClientError(
        code,
        {
            "error": {
                "code": code,
                "status": "RESOURCE_EXHAUSTED" if code == 429 else "INVALID_ARGUMENT",
                "message": "test error",
            }
        },
    )


def test_genai_client_error_429_is_retryable():
    exc = _make_genai_client_error(429)

    assert is_retryable_rate_limit(exc) is True


def test_genai_client_error_400_is_not_retryable():
    exc = _make_genai_client_error(400)

    assert is_retryable_rate_limit(exc) is False


def test_retry_wrapper_records_first_gemini_cooldown(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    sleeps = []
    monkeypatch.setattr("lilith_agent.models.time.sleep", sleeps.append)

    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(RateLimitCooldownError) as raised:
        wrapper._generate([])

    assert raised.value.provider == "google"
    assert raised.value.model == "gemini-3.1-pro"
    assert raised.value.cooldown_seconds == 60
    assert sleeps == []


def test_retry_wrapper_escalates_gemini_cooldowns(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(RateLimitCooldownError) as first:
        wrapper._generate([])
    with pytest.raises(RateLimitCooldownError) as second:
        wrapper._generate([])
    with pytest.raises(RateLimitCooldownError) as third:
        wrapper._generate([])

    assert first.value.cooldown_seconds == 60
    assert second.value.cooldown_seconds == 120
    assert third.value.cooldown_seconds == 300


def test_success_resets_lane_failure_counter(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    failing = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )
    success = _RetryWrapper.model_construct(
        inner=_SuccessfulGenerateModel(), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(RateLimitCooldownError):
        failing._generate([])
    success._generate([])
    with pytest.raises(RateLimitCooldownError) as raised:
        failing._generate([])

    assert raised.value.cooldown_seconds == 60


def test_same_gemini_lane_shares_cooldown(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    now = [1000.0]
    sleeps = []
    monkeypatch.setattr("lilith_agent.models.time.monotonic", lambda: now[0])
    monkeypatch.setattr("lilith_agent.models.time.sleep", sleeps.append)
    first = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )
    second = _RetryWrapper.model_construct(
        inner=_SuccessfulGenerateModel(), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(RateLimitCooldownError):
        first._generate([])
    now[0] = 1005.0
    second._generate([])

    assert sleeps == [55.0]


def test_other_gemini_lane_does_not_sleep_for_pro_cooldown(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    now = [1000.0]
    sleeps = []
    monkeypatch.setattr("lilith_agent.models.time.monotonic", lambda: now[0])
    monkeypatch.setattr("lilith_agent.models.time.sleep", sleeps.append)
    pro = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )
    flash = _RetryWrapper.model_construct(
        inner=_SuccessfulGenerateModel(), provider="google", model_name="gemini-3-flash-preview"
    )

    with pytest.raises(RateLimitCooldownError):
        pro._generate([])
    now[0] = 1005.0
    flash._generate([])

    assert sleeps == []


def test_unknown_google_model_does_not_get_gemini_cooldown(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-unknown"
    )

    with pytest.raises(type(exc)):
        wrapper._generate([])


def _make_genai_quota_error(details: list[dict]):
    pytest.importorskip("google.genai.errors")
    from google.genai.errors import ClientError

    return ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "quota exceeded",
                "details": details,
            }
        },
    )


def test_daily_quota_metadata_raises_batch_abort(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_quota_error(
        [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}],
            }
        ]
    )
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(BatchAbortRateLimitError) as raised:
        wrapper._generate([])

    assert "daily" in raised.value.reason.lower()


def test_long_retry_delay_raises_batch_abort(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_quota_error(
        [
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "900s",
            }
        ]
    )
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(BatchAbortRateLimitError) as raised:
        wrapper._generate([])

    assert "retry" in raised.value.reason.lower()


def test_question_rate_limit_scope_raises_after_50_events():
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)

    with pytest.raises(QuestionRateLimitStreakError) as raised:
        with rate_limit_question_scope():
            for _ in range(50):
                try:
                    raise exc
                except BaseException as caught:
                    from lilith_agent.models import record_rate_limit_observation

                    record_rate_limit_observation(caught)

    assert raised.value.count == 50


def test_question_rate_limit_scope_resets_after_success():
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)

    with rate_limit_question_scope():
        from lilith_agent.models import record_rate_limit_observation, record_rate_limit_success

        for _ in range(49):
            record_rate_limit_observation(exc)
        record_rate_limit_success()
        for _ in range(49):
            record_rate_limit_observation(exc)


def test_batch_window_triggers_pause_after_70_rate_limits_in_100_outcomes():
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    from lilith_agent.models import record_rate_limit_observation, record_rate_limit_success

    for _ in range(70):
        record_rate_limit_observation(exc)
    for _ in range(30):
        record_rate_limit_success()

    assert batch_rate_limit_pause_seconds() == 300
    clear_batch_rate_limit_window()
    assert batch_rate_limit_pause_seconds() is None


def test_batch_window_does_not_trigger_below_threshold():
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    from lilith_agent.models import record_rate_limit_observation, record_rate_limit_success

    for _ in range(69):
        record_rate_limit_observation(exc)
    for _ in range(31):
        record_rate_limit_success()

    assert batch_rate_limit_pause_seconds() is None


def test_bound_retry_wrapper_raises_cooldown_for_gemini_lane(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.time.sleep", lambda _: None)
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    bound = wrapper.bind_tools([])

    with pytest.raises(RateLimitCooldownError) as raised:
        bound.invoke([("user", "hi")])

    assert raised.value.model == "gemini-3.1-pro"


@pytest.mark.asyncio
async def test_async_retry_wrapper_raises_cooldown_for_gemini_lane(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.asyncio.sleep", AsyncMock())
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    with pytest.raises(RateLimitCooldownError) as raised:
        await wrapper._agenerate([])

    assert raised.value.model == "gemini-3.1-pro"


@pytest.mark.asyncio
async def test_async_bound_retry_wrapper_raises_cooldown_for_gemini_lane(monkeypatch):
    _reset_rate_limit_state_for_tests()
    exc = _make_genai_client_error(429)
    monkeypatch.setattr("lilith_agent.models.asyncio.sleep", AsyncMock())
    wrapper = _RetryWrapper.model_construct(
        inner=_FailingGenerateModel(exc), provider="google", model_name="gemini-3.1-pro"
    )

    bound = wrapper.bind_tools([])

    with pytest.raises(RateLimitCooldownError) as raised:
        await bound.ainvoke([("user", "hi")])

    assert raised.value.model == "gemini-3.1-pro"


def test_deepseek_config_defaults_and_env(monkeypatch):
    from lilith_agent.config import Config

    monkeypatch.delenv("GAIA_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GAIA_DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-env-key")

    cfg = Config.from_env()

    assert cfg.deepseek_api_key == "ds-env-key"
    assert cfg.deepseek_base_url == "https://api.deepseek.com"

    monkeypatch.setenv("GAIA_DEEPSEEK_API_KEY", "gaia-ds-key")
    monkeypatch.setenv("GAIA_DEEPSEEK_BASE_URL", "https://deepseek.internal")

    cfg = Config.from_env()

    assert cfg.deepseek_api_key == "gaia-ds-key"
    assert cfg.deepseek_base_url == "https://deepseek.internal"


def test_agent_model_tier_selects_configured_tier(monkeypatch):
    from lilith_agent.config import Config
    from lilith_agent.models import _resolve_agent_model_choice

    monkeypatch.setenv("GAIA_CHEAP_PROVIDER", "deepseek")
    monkeypatch.setenv("GAIA_CHEAP_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("GAIA_STRONG_PROVIDER", "deepseek")
    monkeypatch.setenv("GAIA_STRONG_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("GAIA_AGENT_MODEL_TIER", "strong")

    assert _resolve_agent_model_choice(Config.from_env()) == ("deepseek", "deepseek-v4-pro")


def test_agent_model_direct_override_wins_over_tier(monkeypatch):
    from lilith_agent.config import Config
    from lilith_agent.models import _resolve_agent_model_choice

    monkeypatch.setenv("GAIA_AGENT_MODEL_TIER", "cheap")
    monkeypatch.setenv("GAIA_AGENT_PROVIDER", "deepseek")
    monkeypatch.setenv("GAIA_AGENT_MODEL", "deepseek-v4-pro")

    assert _resolve_agent_model_choice(Config.from_env()) == ("deepseek", "deepseek-v4-pro")


def test_invalid_agent_model_tier_is_rejected(monkeypatch):
    from lilith_agent.config import Config

    monkeypatch.setenv("GAIA_AGENT_MODEL_TIER", "medium")

    with pytest.raises(ValueError, match="GAIA_AGENT_MODEL_TIER"):
        Config.from_env()


def test_build_deepseek_uses_openai_compatible_endpoint(monkeypatch):
    from dataclasses import replace

    from lilith_agent.config import Config
    from lilith_agent.models import _build

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("lilith_agent.models.ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        "lilith_agent.models._RetryWrapper",
        lambda inner, provider, model_name: inner,
    )

    cfg = replace(
        Config.from_env(),
        deepseek_api_key="ds-key",
        deepseek_base_url="https://deepseek.internal",
        max_tokens=123,
    )

    model = _build("deepseek", "deepseek-v4-flash", cfg)

    assert model.kwargs == {
        "model": "deepseek-v4-flash",
        "api_key": "ds-key",
        "base_url": "https://deepseek.internal",
        "max_tokens": 123,
    }


def _fake_deepseek_cfg(monkeypatch):
    from dataclasses import replace

    from lilith_agent.config import Config

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("lilith_agent.models.ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        "lilith_agent.models._RetryWrapper",
        lambda inner, provider, model_name: inner,
    )
    return replace(
        Config.from_env(),
        deepseek_api_key="ds-key",
        deepseek_base_url="https://deepseek.internal",
        max_tokens=123,
    )


def test_build_deepseek_thinking_disabled_sets_extra_body(monkeypatch):
    from lilith_agent.models import _build

    cfg = _fake_deepseek_cfg(monkeypatch)

    model = _build("deepseek", "deepseek-v4-flash", cfg, thinking=False)

    assert model.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_build_deepseek_thinking_enabled_omits_extra_body(monkeypatch):
    from lilith_agent.models import _build

    cfg = _fake_deepseek_cfg(monkeypatch)

    model = _build("deepseek", "deepseek-v4-flash", cfg)

    assert "extra_body" not in model.kwargs
