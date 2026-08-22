"""Structured event log for the whole system — what happened, when, how long.

A single agent answer can take 30+ seconds, and until now there was no way to see
*where* that time went: the UI showed a spinner and the terminal showed nothing.
This module is the shared recorder. Everything interesting emits an event here —
HTTP requests, agent turns, individual tool calls, LLM round-trips, cache hits,
memory recall, safety screens — and the Logs tab reads them back.

Design notes
------------
* **In-memory ring buffer, no disk.** Keeping the last `MAX_EVENTS` in a deque
  means logging can never fill a disk, never blocks on I/O in the middle of a
  tool call, and needs nothing gitignored. The trade is that a uvicorn restart
  clears the log; the Logs tab has an Export button for anything worth keeping.
* **Monotonic sequence numbers.** Clients poll `since=<seq>` to tail the log.
  Wall-clock timestamps are for humans and are not safe to page on (two events
  can share a millisecond).
* **Never let logging break the app.** Every public function swallows its own
  errors — an observability bug must not take down an answer.
* **Backend only.** The browser keeps its own buffer in the same event shape
  (`web/src/logbus.js`) rather than posting every click over the wire mid-stream;
  the Logs tab shows the two side by side and merges them for the combined
  timeline. `source` is carried on every event so the split is explicit either
  way.
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone

# How many events to keep. ~2000 covers several full agent turns plus the
# dashboard traffic around them, at a few hundred KB of memory.
MAX_EVENTS = 2000

# Event groups, in the order the UI shows them. Kept as a constant so the
# frontend filter list and the backend emitters cannot drift apart.
BACKEND_GROUPS = (
    "http",      # inbound API requests
    "agent",     # one agent turn, start to finish
    "router",    # deterministic turn labelling: intent, complexity, risk
    "llm",       # model round-trips
    "tool",      # agent tool calls
    "external",  # outbound third-party APIs (weather, geocode, EIA, ...)
    "cache",     # exact + semantic answer cache, tool-result cache
    "memory",    # episodic recall / record
    "rag",       # knowledge-base searches
    "research",  # web research: provider searches, ranking, dropped passages
    "safety",    # guardrail screens
    "system",    # startup, admin actions, log control
)
FRONTEND_GROUPS = (
    "ui",        # user actions (send, clear, tab switch)
    "api",       # fetch calls from the browser
    "stream",    # SSE events received
    "render",    # React lifecycle / error boundaries
    "error",     # window.onerror + unhandled rejections
)

LEVELS = ("debug", "info", "warn", "error")

_LOCK = threading.Lock()
_EVENTS: deque[dict] = deque(maxlen=MAX_EVENTS)
_SEQ = itertools.count(1)
_START = time.monotonic()
_DROPPED = 0  # events pushed out of the ring buffer, for an honest UI count


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def record(
    group: str,
    event: str,
    message: str = "",
    *,
    source: str = "backend",
    level: str = "info",
    duration_ms: float | None = None,
    thread_id: str | None = None,
    data: dict | None = None,
    ts: str | None = None,
) -> dict | None:
    """Append one event and return it (or None if logging failed).

    `event` is a dotted name like ``tool.end`` — stable enough to filter on,
    while `message` is the human sentence shown in the log row.
    """
    try:
        entry = {
            "seq": next(_SEQ),
            "ts": ts or _now_iso(),
            "t_ms": round((time.monotonic() - _START) * 1000, 1),
            "source": source,
            "group": group,
            "event": event,
            "level": level if level in LEVELS else "info",
            "message": message,
        }
        if duration_ms is not None:
            entry["duration_ms"] = round(float(duration_ms), 1)
        if thread_id:
            entry["thread_id"] = thread_id
        if data:
            entry["data"] = _shrink(data)

        global _DROPPED
        with _LOCK:
            if len(_EVENTS) == _EVENTS.maxlen:
                _DROPPED += 1
            _EVENTS.append(entry)
        return entry
    except Exception:
        return None  # observability must never break the thing it observes


def _shrink(data: dict, limit: int = 600) -> dict:
    """Truncate long values so one fat tool result cannot dominate the buffer."""
    out = {}
    for key, value in list(data.items())[:24]:
        if isinstance(value, str) and len(value) > limit:
            out[key] = value[:limit] + f"… (+{len(value) - limit} chars)"
        elif isinstance(value, (list, tuple)) and len(value) > 20:
            out[key] = list(value[:20]) + [f"… (+{len(value) - 20} more)"]
        else:
            out[key] = value
    return out


@contextmanager
def span(
    group: str,
    name: str,
    message: str = "",
    *,
    thread_id: str | None = None,
    data: dict | None = None,
    level: str = "info",
):
    """Time a block and emit `<name>.start` / `<name>.end` (or `.error`).

    Yields a mutable dict — anything put in it is merged into the end event, so a
    caller can report what it found without knowing it up front::

        with span("tool", "tool", "geocode_address") as s:
            result = geocode_address(addr)
            s["ok"] = result["ok"]
    """
    extra: dict = {}
    started = time.perf_counter()
    record(group, f"{name}.start", message or name, thread_id=thread_id,
           data=data, level="debug")
    try:
        yield extra
    except Exception as exc:
        record(group, f"{name}.error", f"{message or name} failed: {exc}",
               level="error", thread_id=thread_id,
               duration_ms=(time.perf_counter() - started) * 1000,
               data={**(data or {}), **extra, "error": f"{type(exc).__name__}: {exc}"})
        raise
    else:
        record(group, f"{name}.end", message or name, level=level, thread_id=thread_id,
               duration_ms=(time.perf_counter() - started) * 1000,
               data={**(data or {}), **extra} or None)


def snapshot(
    since: int = 0,
    source: str | None = None,
    group: str | None = None,
    level: str | None = None,
    limit: int = 500,
) -> dict:
    """Read events newer than `since`, oldest first, with optional filters."""
    try:
        with _LOCK:
            events = list(_EVENTS)
        selected = [
            e for e in events
            if e["seq"] > since
            and (source is None or e["source"] == source)
            and (group is None or e["group"] == group)
            and (level is None or e["level"] == level)
        ]
        truncated = max(0, len(selected) - limit)
        return {
            "events": selected[-limit:] if limit else selected,
            "latest_seq": events[-1]["seq"] if events else since,
            "total": len(events),
            "dropped": _DROPPED,
            "truncated": truncated,
            "capacity": MAX_EVENTS,
        }
    except Exception:
        return {"events": [], "latest_seq": since, "total": 0, "dropped": 0,
                "truncated": 0, "capacity": MAX_EVENTS}


def clear() -> int:
    """Drop every stored event. Returns how many were removed."""
    global _DROPPED
    with _LOCK:
        removed = len(_EVENTS)
        _EVENTS.clear()
        _DROPPED = 0
    record("system", "logs.cleared", f"Log buffer cleared ({removed} events)")
    return removed


def stats() -> dict:
    """Counts by source and group — powers the Logs tab summary strip."""
    with _LOCK:
        events = list(_EVENTS)
    by_source: dict[str, int] = {}
    by_group: dict[str, int] = {}
    errors = 0
    for e in events:
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1
        by_group[e["group"]] = by_group.get(e["group"], 0) + 1
        if e["level"] == "error":
            errors += 1
    return {
        "total": len(events),
        "capacity": MAX_EVENTS,
        "dropped": _DROPPED,
        "errors": errors,
        "by_source": by_source,
        "by_group": by_group,
        "backend_groups": list(BACKEND_GROUPS),
        "frontend_groups": list(FRONTEND_GROUPS),
    }
