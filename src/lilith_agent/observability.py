from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

log = logging.getLogger(__name__)


def setup_arize(project_name: str = "lilith") -> bool:
    """Enable Arize AX tracing if ARIZE_SPACE_ID and ARIZE_API_KEY are set.

    Uses OpenInference's LangChain instrumentor, which auto-captures every
    LangChain / LangGraph run (LLM calls, tool calls, chains). Returns True
    when tracing is active, False otherwise.
    """
    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")
    if not (space_id and api_key):
        log.info("Arize tracing skipped: ARIZE_SPACE_ID or ARIZE_API_KEY not set")
        return False

    try:
        from arize.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError:
        log.warning(
            "Arize packages missing. Install: "
            "pip install arize-otel openinference-instrumentation-langchain"
        )
        return False

    # Cap OTLP export attempts so a bad key / network can't block shutdown.
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TIMEOUT", "2")
    os.environ.setdefault("OTEL_BSP_EXPORT_TIMEOUT", "2000")

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name=project_name,
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    import atexit
    def _shutdown():
        try:
            tracer_provider.force_flush(timeout_millis=2000)
        except Exception:
            pass
        try:
            tracer_provider.shutdown()
        except Exception:
            pass
    atexit.register(_shutdown)

    log.info("Arize tracing enabled for project=%s", project_name)
    return True


def setup_logging(log_dir: str | Path = ".lilith") -> Path:
    """Configure root logger to write INFO+ to a session log file and WARNING+ to stderr."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"session-{stamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))

    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    # Avoid duplicate handlers on re-entry
    root.handlers = [h for h in root.handlers if not getattr(h, "_lilith", False)]
    fh._lilith = True
    sh._lilith = True
    root.addHandler(fh)
    root.addHandler(sh)
    return log_path


# Keys stripped from every trace record — high-volume, low-signal noise
# from provider responses (Gemini reasoning traces, safety metadata, etc).
_NOISE_KEYS = frozenset({
    "__gemini_function_call_thought_signatures__",
    "reasoning",
    "signature",
    "safety_ratings",
    "safety_settings",
    "thought_signature",
    "thought_signatures",
})


def _sanitize(obj: Any, depth: int = 0) -> Any:
    if depth > 10:
        return obj
    if hasattr(obj, "content") and hasattr(obj, "additional_kwargs") and hasattr(obj, "type"):
        return _msg_to_dict(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v, depth + 1) for k, v in obj.items() if k not in _NOISE_KEYS}
    if isinstance(obj, list):
        return [_sanitize(v, depth + 1) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize(v, depth + 1) for v in obj)
    return obj

def _coerce(obj: Any) -> Any:
    try:
        out = _sanitize(obj)
        # Attempt to see if it is JSON-serializable after sanitization
        json.dumps(out)
        return out
    except Exception:
        # Fallback to repr if there's still un-serializable objects
        try:
            return repr(out)
        except UnboundLocalError:
            return repr(obj)

def _msg_to_dict(msg: Any) -> dict:
    """Convert any BaseMessage-like object to a JSON-safe dict with all payload fields."""
    try:
        mtype = getattr(msg, "type", None) or msg.__class__.__name__
        out = {
            "type": mtype,
            "content": _sanitize(getattr(msg, "content", None), 1),
        }
        for attr in ("name", "tool_call_id", "tool_calls", "additional_kwargs", "response_metadata", "usage_metadata"):
            val = getattr(msg, attr, None)
            if val:
                out[attr] = _sanitize(val, 1)
        return out
    except Exception:
        return {"type": "unknown", "repr": repr(msg)}

def _generations_to_list(response: Any) -> Any:
    gens = getattr(response, "generations", None)
    if not gens:
        return _coerce(response)
    out = []
    for batch in gens:
        batch_out = []
        for gen in batch:
            msg = getattr(gen, "message", None)
            if msg is not None:
                batch_out.append(_msg_to_dict(msg))
            else:
                info = getattr(gen, "generation_info", None)
                batch_out.append({"text": getattr(gen, "text", ""), "info": _coerce(info)})
        out.append(batch_out)
    usage = getattr(response, "llm_output", None)
    return {"generations": out, "llm_output": _coerce(usage)}


class JsonlTraceCallback(BaseCallbackHandler):
    """Append every LLM / tool / chain event as a JSONL record AND mirror to the Python logger.

    Captures full payloads — no truncation — so you can replay a session from disk.
    Writes are flushed after every line for real-time tailing.
    """

    def __init__(self, path: str | Path, logger_name: str = "lilith_agent.trace"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._starts: dict[Any, float] = {}
        self._logger = logging.getLogger(logger_name)
        # Keep a persistent handle + flush each write for real-time tailing.
        self._fh = self.path.open("a", buffering=1)

    def _emit(self, record: dict) -> None:
        record.setdefault("ts", datetime.now(timezone.utc).isoformat())
        record.pop("run_id", None)
        line = json.dumps(record, default=repr)
        self._fh.write(line + "\n")
        self._fh.flush()
        # Mirror at DEBUG: LangGraph emits nested chain_start/chain_end for every
        # subgraph, which floods the console at INFO. Full payloads still live in
        # the .jsonl file; errors stay visible because we log them at WARNING.
        summary = f"{record['event']}"
        for k in ("name", "model", "elapsed_s"):
            if k in record:
                summary += f" {k}={record[k]}"
        if "error" in record:
            self._logger.warning("%s error=%s", summary, record["error"])
        else:
            self._logger.debug(summary)

    # --- chain (LangGraph node) boundaries ---
    def on_chain_start(self, serialized, inputs, *, run_id=None, **kwargs):
        self._starts[run_id] = time.monotonic()
        self._emit({
            "event": "chain_start",
            "run_id": str(run_id),
            "name": (serialized or {}).get("name"),
            "inputs": _coerce(inputs),
            "tags": _coerce(kwargs.get("tags")),
        })

    def on_chain_end(self, outputs, *, run_id=None, **kwargs):
        elapsed = time.monotonic() - self._starts.pop(run_id, time.monotonic())
        self._emit({
            "event": "chain_end",
            "run_id": str(run_id),
            "elapsed_s": round(elapsed, 3),
            "outputs": _coerce(outputs),
        })

    def on_chain_error(self, error, *, run_id=None, **kwargs):
        elapsed = time.monotonic() - self._starts.pop(run_id, time.monotonic())
        self._emit({
            "event": "chain_error",
            "run_id": str(run_id),
            "elapsed_s": round(elapsed, 3),
            "error": f"{type(error).__name__}: {error}",
        })

    # --- tool boundaries ---
    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs):
        self._starts[run_id] = time.monotonic()
        self._emit({
            "event": "tool_start",
            "run_id": str(run_id),
            "name": (serialized or {}).get("name"),
            "input": _coerce(input_str),
        })

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        elapsed = time.monotonic() - self._starts.pop(run_id, time.monotonic())
        self._emit({
            "event": "tool_end",
            "run_id": str(run_id),
            "elapsed_s": round(elapsed, 3),
            "output": _coerce(output),
        })

    def on_tool_error(self, error, *, run_id=None, **kwargs):
        elapsed = time.monotonic() - self._starts.pop(run_id, time.monotonic())
        self._emit({
            "event": "tool_error",
            "run_id": str(run_id),
            "elapsed_s": round(elapsed, 3),
            "error": f"{type(error).__name__}: {error}",
        })

    # --- LLM / chat-model boundaries (full payload, both directions) ---
    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self._starts[run_id] = time.monotonic()
        self._emit({
            "event": "chat_model_start",
            "run_id": str(run_id),
            "model": (serialized or {}).get("name"),
            "invocation_params": _coerce(kwargs.get("invocation_params")),
            "messages": [[_msg_to_dict(m) for m in batch] for batch in (messages or [])],
        })

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._starts[run_id] = time.monotonic()
        self._emit({
            "event": "llm_start",
            "run_id": str(run_id),
            "model": (serialized or {}).get("name"),
            "prompts": [_coerce(p) for p in (prompts or [])],
        })

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        elapsed = time.monotonic() - self._starts.pop(run_id, time.monotonic())
        self._emit({
            "event": "llm_end",
            "run_id": str(run_id),
            "elapsed_s": round(elapsed, 3),
            "response": _generations_to_list(response),
        })

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        elapsed = time.monotonic() - self._starts.pop(run_id, time.monotonic())
        self._emit({
            "event": "llm_error",
            "run_id": str(run_id),
            "elapsed_s": round(elapsed, 3),
            "error": f"{type(error).__name__}: {error}",
        })

    # --- agent actions (ReAct reasoning steps) ---
    def on_agent_action(self, action, *, run_id=None, **kwargs):
        self._emit({
            "event": "agent_action",
            "run_id": str(run_id),
            "tool": getattr(action, "tool", None),
            "tool_input": _coerce(getattr(action, "tool_input", None)),
            "log": getattr(action, "log", None),
        })

    def on_agent_finish(self, finish, *, run_id=None, **kwargs):
        self._emit({
            "event": "agent_finish",
            "run_id": str(run_id),
            "return_values": _coerce(getattr(finish, "return_values", None)),
            "log": getattr(finish, "log", None),
        })

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
