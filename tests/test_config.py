from __future__ import annotations

from dataclasses import fields

from lilith_agent.config import Config


def test_config_has_no_dead_max_json_repairs_field():
    """Guard: max_json_repairs was unused anywhere in src/ and was removed."""
    field_names = {f.name for f in fields(Config)}
    assert "max_json_repairs" not in field_names


def test_config_from_env_loads_without_errors():
    cfg = Config.from_env()
    assert cfg.recursion_limit > 0
    assert cfg.cheap_provider
    assert cfg.strong_provider


def test_budget_knobs_have_sensible_defaults():
    cfg = Config.from_env()
    assert cfg.budget_hard_cap >= cfg.budget_warn_at > 0
    assert 0.0 < cfg.semantic_dedup_threshold <= 1.0


def test_budget_knobs_are_env_overridable(monkeypatch):
    monkeypatch.setenv("GAIA_BUDGET_HARD_CAP", "7")
    monkeypatch.setenv("GAIA_BUDGET_WARN_AT", "3")
    monkeypatch.setenv("GAIA_SEMANTIC_DEDUP_THRESHOLD", "0.8")
    cfg = Config.from_env()
    assert cfg.budget_hard_cap == 7
    assert cfg.budget_warn_at == 3
    assert abs(cfg.semantic_dedup_threshold - 0.8) < 1e-9
