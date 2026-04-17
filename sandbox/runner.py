"""In-container entrypoint.

Reads the agent's code from stdin, runs it, prints output to stdout. Same
REPL-style last-expression handling as the process-level fallback in
``src/lilith_agent/tools/python_exec.py`` — keep the behavior identical so
switching backends doesn't change what the agent sees.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import sys
import traceback

_PY_RUN = builtins.exec
_PY_EVAL = builtins.eval


def main() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024 * 1024, 128 * 1024 * 1024))
    except Exception:
        pass

    code = sys.stdin.read()
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
        sys.stdout.write(buf.getvalue())
    except Exception:
        numbered = "\n".join(
            f"{i+1:3d}: {line}" for i, line in enumerate(code.splitlines())
        )
        sys.stdout.write(
            buf.getvalue()
            + "\nERROR in agent-generated code:\n\n"
            + numbered
            + "\n\n"
            + traceback.format_exc()
        )


if __name__ == "__main__":
    main()
