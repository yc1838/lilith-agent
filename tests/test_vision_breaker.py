from __future__ import annotations

import threading

from lilith_agent.tools.vision import (
    _is_vision_breaker_tripped,
    _trip_vision_breaker,
    reset_vision_state,
)


def test_reset_clears_breaker():
    _trip_vision_breaker()
    assert _is_vision_breaker_tripped() is True
    reset_vision_state()
    assert _is_vision_breaker_tripped() is False


def test_breaker_is_per_thread_not_process_global():
    """A failure in one thread must not taint other threads."""
    reset_vision_state()
    other_tripped = {"value": None}

    def other_thread():
        # Other thread starts untripped regardless of main-thread state.
        other_tripped["value"] = _is_vision_breaker_tripped()

    _trip_vision_breaker()
    assert _is_vision_breaker_tripped() is True

    t = threading.Thread(target=other_thread)
    t.start()
    t.join()

    assert other_tripped["value"] is False
    # Main thread still tripped
    assert _is_vision_breaker_tripped() is True
    reset_vision_state()
