from __future__ import annotations


def test_apply_safe_thread_env_overrides_existing_values(monkeypatch):
    from lilith_agent.runtime_env import SAFE_THREAD_ENV, apply_safe_thread_env

    for key in SAFE_THREAD_ENV:
        monkeypatch.setenv(key, "99")

    apply_safe_thread_env()

    for key, value in SAFE_THREAD_ENV.items():
        assert __import__("os").environ[key] == value
