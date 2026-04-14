from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.messages import AIMessage

from lilith_agent.models import (
    _RetryWrapper,
    _NoThinkWrapper,
    _BoundRetryWrapper,
    _BoundNoThinkWrapper,
)


class _FakeChatModel:
    """Minimal stand-in for a BaseChatModel exposing bind_tools."""

    _llm_type = "fake"

    def __init__(self):
        self.bound_with = None

    def bind_tools(self, tools, **kwargs):
        self.bound_with = tools
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
