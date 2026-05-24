from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _project_dependencies() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    return {dep.split("[", 1)[0].split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].strip() for dep in deps}


def test_direct_runtime_imports_are_declared_dependencies():
    deps = _project_dependencies()

    assert {"datasets", "httpx", "pandas", "requests"}.issubset(deps)


def test_ci_installs_test_extra():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert 'pip install -e ".[test]"' in workflow
