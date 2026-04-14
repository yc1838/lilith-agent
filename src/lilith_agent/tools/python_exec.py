"""Sandboxed Python executor for the agent.

LLM-generated code is untrusted. The safety boundary is a spawn-based
subprocess with a hard wall-clock timeout. Output is captured text only.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import multiprocessing as mp
import traceback

_PY_RUN = getattr(builtins, "exec")
_PY_EVAL = getattr(builtins, "eval")


def _worker(code: str, q: "mp.Queue") -> None:
    buf = io.StringIO()
    ns: dict = {}
    try:
        tree = ast.parse(code, mode="exec")
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(body=tree.body[-1].value)
            tree.body = tree.body[:-1]
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            _PY_RUN(compile(tree, "<agent>", "exec"), ns)
            if last_expr is not None:
                val = _PY_EVAL(compile(last_expr, "<agent>", "eval"), ns)
                if val is not None:
                    print(repr(val), file=buf)
        q.put(buf.getvalue())
    except Exception:
        # Prepend line numbers to the code for better debugging in the traceback
        lines = code.splitlines()
        numbered_code = "\n".join(f"{i+1:3d}: {line}" for i, line in enumerate(lines))
        error_msg = f"ERROR in agent-generated code:\n\n{numbered_code}\n\n{traceback.format_exc()}"
        q.put(buf.getvalue() + "\n" + error_msg)


def run_python(code: str, timeout: int = 60) -> str:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return f"ERROR: execution timed out after {timeout}s"
    if not q.empty():
        return q.get()
    return ""
