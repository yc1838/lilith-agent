"""Behavior tests for the sandboxed Python executor.

The process-level fallback must enforce: scrubbed env (no secrets), scratch
cwd (no repo-root writes from relative paths), wall-clock timeout, and bounded
output. The Docker backend is tested separately via argv-mocking; actual
docker runs require docker on PATH and are skipped otherwise.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from lilith_agent.tools.python_exec import run_python


@pytest.fixture(autouse=True)
def _force_process_backend(monkeypatch):
    """Default every test to the process backend. Docker-selection tests
    override this by re-setting ``LILITH_SANDBOX`` inside the test body.
    The docker integration test near the bottom of this file explicitly opts in.
    """
    monkeypatch.setenv("LILITH_SANDBOX", "process")


def test_simple_expression_returns_value():
    result = run_python("2 + 2")
    assert "4" in result


def test_print_output_captured():
    result = run_python("print('hello'); print('world')")
    assert "hello" in result
    assert "world" in result


def test_exception_returns_traceback_with_numbered_code():
    result = run_python("x = 1\ny = 2\nraise ValueError('boom')")
    assert "ValueError" in result
    assert "boom" in result
    # Numbered code assists debugging by pinpointing the failing line.
    assert "  3: raise ValueError" in result


def test_api_key_env_vars_are_scrubbed(monkeypatch):
    """Secrets in the parent env must not leak into subprocess-exec'd code."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-should-not-leak")
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-should-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-should-not-leak")
    monkeypatch.setenv("GAIA_HUGGINGFACE_API_KEY", "hf-should-not-leak")

    result = run_python(
        "import os\n"
        "print('A=' + str(os.environ.get('ANTHROPIC_API_KEY')))\n"
        "print('G=' + str(os.environ.get('GOOGLE_API_KEY')))\n"
        "print('O=' + str(os.environ.get('OPENAI_API_KEY')))\n"
        "print('W=' + str(os.environ.get('AWS_SECRET_ACCESS_KEY')))\n"
        "print('H=' + str(os.environ.get('GAIA_HUGGINGFACE_API_KEY')))\n"
    )
    assert "should-not-leak" not in result
    assert "A=None" in result
    assert "G=None" in result
    assert "O=None" in result
    assert "W=None" in result
    assert "H=None" in result


def test_allowlisted_env_passes_through():
    result = run_python(
        "import os\nprint('HAS_PATH=' + str(bool(os.environ.get('PATH'))))"
    )
    assert "HAS_PATH=True" in result


def test_cwd_is_not_repo_root():
    """Subprocess runs in a scratch dir so relative-path writes cannot hit the repo."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    result = run_python("import os\nprint('CWD=' + os.getcwd())")
    cwd_lines = [ln for ln in result.splitlines() if ln.startswith("CWD=")]
    assert cwd_lines, f"no CWD= line in output: {result!r}"
    cwd = cwd_lines[0][len("CWD="):].strip()
    assert cwd != repo_root
    assert not cwd.startswith(repo_root + "/")


def test_relative_write_goes_to_scratch_not_repo():
    """A relative-path write from inside run_python must not land in the repo."""
    canary = "canary-relative-write-" + os.urandom(4).hex()
    # Attempt a relative write
    run_python(f"open('leak.txt', 'w').write({canary!r})")
    repo_root = Path(__file__).resolve().parent.parent
    assert not (repo_root / "leak.txt").exists(), (
        "relative-path write escaped the sandbox into the repo root"
    )


def test_timeout_terminates_infinite_loop():
    result = run_python("while True:\n    pass", timeout=2)
    assert "timed out" in result.lower()


def test_output_is_capped():
    """A runaway print loop must not blow the caller's context budget."""
    # 2MB of output; cap should hold it well below that.
    result = run_python("print('x' * (2 * 1024 * 1024))")
    assert len(result) < 512 * 1024, f"output not capped: {len(result)} chars"


# --- Docker backend tests (mocked subprocess, no docker required) ---


def test_docker_backend_argv_contains_isolation_flags(monkeypatch):
    """The docker invocation must include every isolation flag from the design."""
    from lilith_agent.tools import python_exec as pe

    captured: dict = {}

    class FakeProc:
        def __init__(self) -> None:
            self.stdout = "ok\n"
            self.stderr = ""
            self.returncode = 0

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(pe.subprocess, "run", fake_run)

    pe._run_docker_sandbox("print('hi')", timeout=30, image="lilith-pysandbox:latest")

    argv = captured["argv"]
    assert argv[0] == "docker"
    assert argv[1] == "run"
    assert "--rm" in argv
    assert "--network=bridge" in argv
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    # tmpfs mount for /scratch
    tmpfs_idx = argv.index("--tmpfs")
    assert argv[tmpfs_idx + 1].startswith("/scratch")
    # memory + cpu caps present
    assert any(a.startswith("--memory=") for a in argv)
    assert any(a.startswith("--pids-limit=") for a in argv)
    # image name is the last positional before args to the entrypoint
    assert "lilith-pysandbox:latest" in argv


def test_docker_backend_does_not_pass_secrets_via_env(monkeypatch):
    """We must not forward the host's env into the container."""
    from lilith_agent.tools import python_exec as pe

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")

    captured: dict = {}

    class FakeProc:
        stdout = "ok\n"
        stderr = ""
        returncode = 0

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(pe.subprocess, "run", fake_run)
    pe._run_docker_sandbox("print('hi')", timeout=30, image="lilith-pysandbox:latest")

    argv = captured["argv"]
    # No --env / -e flag injecting the secret.
    for i, a in enumerate(argv):
        assert "sk-should-not-leak" not in a
        if a in ("-e", "--env"):
            assert "ANTHROPIC_API_KEY" not in argv[i + 1]


def test_auto_select_falls_back_to_process_when_docker_unavailable(monkeypatch):
    """`auto` mode must pick process backend if docker is not on PATH."""
    from lilith_agent.tools import python_exec as pe

    monkeypatch.setattr(pe, "_docker_available", lambda: False)
    # Run with auto — should succeed via process fallback.
    monkeypatch.setenv("LILITH_SANDBOX", "auto")
    result = run_python("print('via-process')")
    assert "via-process" in result


def test_force_docker_errors_clearly_when_unavailable(monkeypatch):
    """Explicit docker mode must not silently fall back — fail loudly."""
    from lilith_agent.tools import python_exec as pe

    monkeypatch.setattr(pe, "_docker_available", lambda: False)
    monkeypatch.setenv("LILITH_SANDBOX", "docker")
    result = run_python("print('should-not-run')")
    assert "ERROR" in result
    assert "docker" in result.lower()


def test_force_process_runs_locally_even_if_docker_present(monkeypatch):
    """Explicit process mode must not invoke docker."""
    from lilith_agent.tools import python_exec as pe

    called = {"docker": False}

    def fake_run_docker(*a, **kw):
        called["docker"] = True
        return "unused"

    monkeypatch.setattr(pe, "_docker_available", lambda: True)
    monkeypatch.setattr(pe, "_run_docker_sandbox", fake_run_docker)
    monkeypatch.setenv("LILITH_SANDBOX", "process")

    result = run_python("print('local')")
    assert "local" in result
    assert called["docker"] is False


# --- Integration test: actually runs docker if it's available ---


@pytest.mark.skipif(
    shutil.which("docker") is None or os.getenv("LILITH_SKIP_DOCKER_INTEGRATION") == "1",
    reason="docker not installed or integration tests disabled",
)
def test_docker_integration_if_available(monkeypatch):
    """Smoke test: if docker is on PATH and the image exists, run a simple program.

    This test is permissive: it skips if the image isn't built (common in CI).
    """
    import subprocess as sp
    from lilith_agent.tools import python_exec as pe

    image = "lilith-pysandbox:latest"
    inspect = sp.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
    )
    if inspect.returncode != 0:
        pytest.skip(f"docker image {image} not built; run `docker build -t {image} sandbox/`")

    monkeypatch.setenv("LILITH_SANDBOX", "docker")
    result = run_python("print(2 + 2)")
    assert "4" in result
