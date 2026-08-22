"""LangChain callback handler that feeds the telemetry log.

The orchestrator's own stream tells us *what* the agent did, but not how long
each part took: by the time a tool result appears in the message stream, the
work is already over. Callbacks fire at the real boundaries, so this is where
accurate per-call durations come from — and where the token counts that explain
a slow turn are available.

It does two jobs at once:

1. **Logs** every model round-trip and every tool call to `telemetry`.
2. **Feeds the chat UI**, by queueing small events the orchestrator drains and
   forwards over SSE. That is what puts "· 1.4s" next to each step in the live
   trace instead of a bare bullet.

Nothing here may raise. A telemetry bug that killed a turn would be much worse
than a missing log line.
"""
from __future__ import annotations

import threading
import time

from langchain_core.callbacks import BaseCallbackHandler

import telemetry


def _tool_name(serialized: dict | None, kwargs: dict) -> str:
    """Tool name, whichever way this LangChain version reports it."""
    if serialized and serialized.get("name"):
        return serialized["name"]
    return kwargs.get("name") or "tool"


class TelemetryCallbackHandler(BaseCallbackHandler):
    """Times model and tool runs, logs them, and queues UI events."""

    # Long tool results are logged in full to the buffer's own limit; the UI
    # preview is deliberately short so the live feed stays readable.
    PREVIEW = 200

    def __init__(self, thread_id: str | None = None):
        self.thread_id = thread_id
        self._lock = threading.Lock()
        self._starts: dict[str, float] = {}   # run_id -> perf_counter at start
        self._names: dict[str, str] = {}      # run_id -> tool name
        self._pending: list[dict] = []        # UI events waiting to be streamed
        self._durations: dict[str, list[float]] = {}  # tool name -> finished durations
        self.llm_turns = 0
        self.total_llm_ms = 0.0
        self.total_tool_ms = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        # Prompt-cache reads. If this stays 0 across the turns of one question,
        # the cached prefix is being invalidated — see _cacheable_system_prompt.
        self.tokens_cached = 0

    # --- what the orchestrator consumes -----------------------------------
    def drain(self) -> list[dict]:
        """Take the queued UI events (called between agent stream chunks)."""
        with self._lock:
            events, self._pending = self._pending, []
        return events

    def take_duration(self, tool_name: str) -> float | None:
        """Pop the oldest recorded duration for a tool, to attach to its result."""
        with self._lock:
            queue = self._durations.get(tool_name)
            return queue.pop(0) if queue else None

    def summary(self) -> dict:
        return {
            "llm_turns": self.llm_turns,
            "llm_ms": round(self.total_llm_ms, 1),
            "tool_ms": round(self.total_tool_ms, 1),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_cached": self.tokens_cached,
        }

    def _queue(self, event: dict) -> None:
        with self._lock:
            self._pending.append(event)

    # --- model ------------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        try:
            self._starts[str(run_id)] = time.perf_counter()
            count = sum(len(m) for m in (messages or []))
            telemetry.record("llm", "llm.start", "Model call started", level="debug",
                             thread_id=self.thread_id, data={"messages": count})
        except Exception:
            pass

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        try:
            self._starts[str(run_id)] = time.perf_counter()
        except Exception:
            pass

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        try:
            started = self._starts.pop(str(run_id), None)
            elapsed = (time.perf_counter() - started) * 1000 if started else None
            self.llm_turns += 1
            if elapsed:
                self.total_llm_ms += elapsed

            usage = self._usage(response)
            self.tokens_in += usage.get("input_tokens") or 0
            self.tokens_out += usage.get("output_tokens") or 0
            self.tokens_cached += usage.get("cached_tokens") or 0

            telemetry.record(
                "llm", "llm.end", f"Model turn {self.llm_turns}",
                thread_id=self.thread_id, duration_ms=elapsed,
                data={"turn": self.llm_turns, **usage},
            )
            self._queue({"type": "llm_turn", "turn": self.llm_turns,
                         "duration_ms": round(elapsed, 1) if elapsed else None, **usage})
        except Exception:
            pass

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        try:
            started = self._starts.pop(str(run_id), None)
            telemetry.record(
                "llm", "llm.error", f"Model call failed: {type(error).__name__}",
                level="error", thread_id=self.thread_id,
                duration_ms=(time.perf_counter() - started) * 1000 if started else None,
                data={"error": str(error)},
            )
        except Exception:
            pass

    @staticmethod
    def _usage(response) -> dict:
        """Pull token counts out of whichever field this provider populated."""
        try:
            output = getattr(response, "llm_output", None) or {}
            usage = output.get("token_usage") or output.get("usage") or {}
            if not usage:
                for generations in getattr(response, "generations", []) or []:
                    for gen in generations:
                        meta = getattr(getattr(gen, "message", None),
                                       "usage_metadata", None)
                        if meta:
                            usage = meta
                            break
            # Prompt-cache reads live in a nested details block on the OpenAI-shaped
            # response, and under input_token_details on LangChain's own usage
            # metadata. Check both so the number is right whichever path answered.
            details = (usage.get("prompt_tokens_details")
                       or usage.get("input_token_details") or {})
            return {
                "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
                "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
                "cached_tokens": details.get("cached_tokens") or details.get("cache_read"),
            }
        except Exception:
            return {}

    # --- tools ------------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs):
        try:
            name = _tool_name(serialized, kwargs)
            key = str(run_id)
            self._starts[key] = time.perf_counter()
            self._names[key] = name
            telemetry.record("tool", "tool.start", name, level="debug",
                             thread_id=self.thread_id,
                             data={"tool": name, "input": str(input_str)})
        except Exception:
            pass

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        try:
            key = str(run_id)
            started = self._starts.pop(key, None)
            name = self._names.pop(key, _tool_name(None, kwargs))
            elapsed = (time.perf_counter() - started) * 1000 if started else None
            if elapsed:
                self.total_tool_ms += elapsed
                with self._lock:
                    self._durations.setdefault(name, []).append(round(elapsed, 1))

            text = str(getattr(output, "content", output))
            telemetry.record("tool", "tool.end", name, thread_id=self.thread_id,
                             duration_ms=elapsed,
                             data={"tool": name, "result": text[:self.PREVIEW]})
        except Exception:
            pass

    def on_tool_error(self, error, *, run_id=None, **kwargs):
        try:
            key = str(run_id)
            started = self._starts.pop(key, None)
            name = self._names.pop(key, _tool_name(None, kwargs))
            telemetry.record(
                "tool", "tool.error", f"{name} failed: {type(error).__name__}",
                level="error", thread_id=self.thread_id,
                duration_ms=(time.perf_counter() - started) * 1000 if started else None,
                data={"tool": name, "error": str(error)},
            )
        except Exception:
            pass
