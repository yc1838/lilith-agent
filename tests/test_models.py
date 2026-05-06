import pytest

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from lilith_agent.models import (
    _RetryWrapper,
    _NoThinkWrapper,
    _BoundRetryWrapper,
    _BoundNoThinkWrapper,
    RateLimitCooldownError,
    _reset_rate_limit_state_for_tests,
    is_retryable_rate_limit,
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
