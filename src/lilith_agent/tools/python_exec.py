"""Sandboxed Python executor for the agent.

LLM-generated code is untrusted. Two backends:

- ``process`` (default, always works): subprocess with env scrubbed to an
  allowlist, cwd pinned to a per-call scratch tempdir, wall-clock timeout,
  output bounded, resource limits via rlimit on Unix.
- ``docker``: container with bridge network, read-only rootfs + tmpfs /sandbox,
  ``--cap-drop=ALL``, ``--security-opt=no-new-privileges``, memory and pid
  caps. Metadata-IP block lives in the image's ``sitecustomize.py``.

Selection: ``LILITH_SANDBOX`` env var — ``auto`` (default), ``process``,
``docker``. ``auto`` uses docker when available, falls back to process.

The process backend does not prevent the subprocess from reading the host
filesystem or making outbound network calls. Env scrubbing defeats the most
common exfil path (API keys in env). For hard filesystem/network isolation
run with ``LILITH_SANDBOX=docker`` and a built ``lilith-pysandbox`` image.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

_OUTPUT_CAP_CHARS = 200_000
_DOCKER_IMAGE_DEFAULT = "lilith-pysandbox:latest"
_DOCKER_STARTUP_HEADROOM_S = 5

# Env vars that are safe (or necessary) to forward. Everything else is dropped.
_ENV_ALLOWLIST = frozenset({
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "USER",
    "LOGNAME",
    "SHELL",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONIOENCODING",
    # Proxy config is routinely needed for scraping; not a secret.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
})


# Runner is injected into the subprocess via `python -c`. Reads user code from
# stdin so nothing sensitive lands in argv (visible via `ps`).
_RUNNER_SCRIPT = textwrap.dedent(
    """
    import ast, sys, io, contextlib, traceback

    try:
        import resource
        # CPU 60s, address space 1GiB, single-file write 128MiB.
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024 * 1024, 128 * 1024 * 1024))
    except Exception:
        pass  # non-Unix / sandbox already capping us — fine.

    code = sys.stdin.read()
    buf = io.StringIO()
    ns: dict = {}
    try:
        tree = ast.parse(code, mode='exec')
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(body=tree.body[-1].value)
            tree.body = tree.body[:-1]
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(tree, '<agent>', 'exec'), ns)
            if last_expr is not None:
                val = eval(compile(last_expr, '<agent>', 'eval'), ns)
                if val is not None:
                    print(repr(val), file=buf)
        sys.stdout.write(buf.getvalue())
    except Exception:
        numbered = '\\n'.join(f'{i+1:3d}: {line}' for i, line in enumerate(code.splitlines()))
        sys.stdout.write(
            buf.getvalue()
            + '\\nERROR in agent-generated code:\\n\\n'
            + numbered
            + '\\n\\n'
            + traceback.format_exc()
        )
    """
).strip()


def _scrubbed_env() -> dict[str, str]:
    """Return a fresh env dict containing only allowlisted keys from os.environ."""
    return {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}


def _truncate_output(text: str) -> str:
    if len(text) <= _OUTPUT_CAP_CHARS:
        return text
    dropped = len(text) - _OUTPUT_CAP_CHARS
    return text[:_OUTPUT_CAP_CHARS] + f"\n...[output truncated: +{dropped} chars]"


def _run_process_sandbox(code: str, timeout: int) -> str:
    """Run ``code`` in a subprocess with scrubbed env + scratch cwd + rlimits."""
    with tempfile.TemporaryDirectory(prefix="lilith-pysbx-") as scratch:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER_SCRIPT],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_scrubbed_env(),
                cwd=scratch,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: execution timed out after {timeout}s"
        out = proc.stdout or ""
        if proc.stderr and not out:
            out = proc.stderr
        return _truncate_output(out)


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _run_docker_sandbox(
    code: str,
    timeout: int,
    image: str = _DOCKER_IMAGE_DEFAULT,
) -> str:
    """Run ``code`` inside a hardened container. Image must already be built."""
    argv = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network=bridge",
        "--read-only",
        "--tmpfs",
        # mode=1777 so the non-root container user can write to the tmpfs;
        # without it the default owner is root:root 755 and writes fail EACCES.
        "/sandbox:rw,size=128m,mode=1777,noexec,nosuid,nodev",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1.0",
        "--pids-limit=128",
        "-w",
        "/sandbox",
        image,
    ]

    try:
        proc = subprocess.run(
            argv,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout + _DOCKER_STARTUP_HEADROOM_S,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: execution timed out after {timeout}s (docker backend)"
    except FileNotFoundError:
        return "ERROR: docker binary not found on PATH"

    out = proc.stdout or ""
    if proc.returncode != 0 and not out:
        out = (proc.stderr or "").strip() or f"docker exited with code {proc.returncode}"
    return _truncate_output(out)


def _select_backend() -> str:
    raw = (os.getenv("LILITH_SANDBOX") or "auto").strip().lower()
    if raw not in {"auto", "process", "docker"}:
        raw = "auto"
    if raw == "auto":
        return "docker" if _docker_available() else "process"
    return raw


def run_python(code: str, timeout: int = 60) -> str:
    backend = _select_backend()
    if backend == "docker":
        if not _docker_available():
            return (
                "ERROR: LILITH_SANDBOX=docker was requested but `docker` is not on PATH. "
                "Install Docker or switch to LILITH_SANDBOX=process."
            )
        return _run_docker_sandbox(code, timeout=timeout)
    return _run_process_sandbox(code, timeout=timeout)
