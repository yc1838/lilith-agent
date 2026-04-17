from __future__ import annotations

from lilith_agent.app import _strip_response_metadata_noise


def test_preserves_token_usage():
    meta = {
        "input_tokens": 100,
        "output_tokens": 50,
        "safety_ratings": [{"category": "X", "probability": "LOW"}],
    }
    out = _strip_response_metadata_noise(meta)
    assert out["input_tokens"] == 100
    assert out["output_tokens"] == 50
    assert "safety_ratings" not in out


def test_preserves_usage_subdict():
    meta = {
        "usage": {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 5},
        "logprobs": {"big_noise": "x" * 1000},
    }
    out = _strip_response_metadata_noise(meta)
    assert out["usage"]["input_tokens"] == 10
    assert out["usage"]["cache_read_input_tokens"] == 5
    assert "logprobs" not in out


def test_preserves_model_and_stop_reason():
    meta = {
        "model_name": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "safety_settings": [1, 2, 3],
    }
    out = _strip_response_metadata_noise(meta)
    assert out["model_name"] == "claude-sonnet-4-6"
    assert out["stop_reason"] == "end_turn"
    assert "safety_settings" not in out


def test_empty_input_gives_empty_output():
    assert _strip_response_metadata_noise({}) == {}
    assert _strip_response_metadata_noise(None) == {}
