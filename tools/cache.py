"""A small TTL cache for tool results and agent answers.

Answers are slow for a structural reason: every tool call is a real network
request, and the Tree-of-Thought advisor alone makes three LLM calls. Most of
that work is repeated — elevation never changes, a forecast is good for minutes,
and asking the same question twice in a demo should not cost the same again.

TTLs are chosen per data type rather than globally, because "how long is this
still true" differs enormously: elevation is effectively permanent, an active
weather alert is not.
"""
from __future__ import annotations

import functools
import threading
import time

import telemetry

_LOCK = threading.Lock()
_STORE: dict[str, tuple[float, object]] = {}

# How long each kind of data stays fresh (seconds).
TTL_ELEVATION = 30 * 24 * 3600   # ground level does not move
TTL_GEOCODE = 24 * 3600          # an address maps to the same point
TTL_FORECAST = 10 * 60           # forecasts update every few minutes
TTL_ALERTS = 5 * 60              # advisories are time-critical; keep short
TTL_AIR_QUALITY = 15 * 60
TTL_ENERGY = 6 * 3600            # EIA publishes monthly
TTL_RAG = 60 * 60                # the corpus only changes on re-ingest
TTL_ANSWER = 15 * 60             # a full agent answer
TTL_RESEARCH = 24 * 3600         # web evidence: how-to guidance ages in years,
                                 # and the free providers meter every call


def _key(prefix: str, args: tuple, kwargs: dict) -> str:
    return f"{prefix}|{args!r}|{sorted(kwargs.items())!r}"


def _group_for(name: str) -> str:
    """Which log group a cached call belongs to.

    Everything decorated here is either a knowledge-base search or a third-party
    HTTP call, and those read very differently in the Logs tab.
    """
    return "rag" if name.startswith("memory.rag_store") else "external"


def cached(ttl: float, prefix: str | None = None):
    """Memoize a function's return value for `ttl` seconds.

    Only caches truthy results, and never caches a dict carrying ok=False — a
    failed lookup should be retried, not remembered.

    Every call is logged: a hit (microseconds) or a miss followed by the real
    call and how long it took. Since every decorated function is an outbound API
    request or a KB search, this is where most of a slow answer's time shows up.
    """
    def decorator(fn):
        name = prefix or f"{fn.__module__}.{fn.__qualname__}"
        short = name.rsplit(".", 1)[-1]
        group = _group_for(name)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = _key(name, args, kwargs)
            now = time.time()
            with _LOCK:
                hit = _STORE.get(key)
                if hit and hit[0] > now:
                    telemetry.record(
                        "cache", "cache.hit", f"{short} served from cache",
                        level="debug",
                        data={"function": name, "ttl_left_s": round(hit[0] - now, 1)},
                    )
                    return hit[1]

            telemetry.record("cache", "cache.miss", f"{short} not cached — calling out",
                             level="debug", data={"function": name})
            started = time.perf_counter()
            try:
                value = fn(*args, **kwargs)
            except Exception as exc:
                telemetry.record(
                    group, f"{group}.error", f"{short} raised {type(exc).__name__}",
                    level="error", duration_ms=(time.perf_counter() - started) * 1000,
                    data={"function": name, "error": str(exc)},
                )
                raise

            elapsed_ms = (time.perf_counter() - started) * 1000
            ok = not (isinstance(value, dict) and value.get("ok") is False)
            telemetry.record(
                group, f"{group}.call", short,
                level="info" if ok else "warn", duration_ms=elapsed_ms,
                data={"function": name, "ok": ok,
                      "error": (value or {}).get("error") if isinstance(value, dict) else None},
            )

            cacheable = bool(value) and ok
            if cacheable:
                with _LOCK:
                    _STORE[key] = (now + ttl, value)
                telemetry.record("cache", "cache.store", f"{short} cached for {ttl:.0f}s",
                                 level="debug", data={"function": name, "ttl_s": ttl})
            return value

        wrapper.cache_clear = lambda: clear(name)  # type: ignore[attr-defined]
        return wrapper
    return decorator


def get(key: str):
    """Read a manually-managed entry (used for whole agent answers)."""
    with _LOCK:
        hit = _STORE.get(key)
        if hit and hit[0] > time.time():
            return hit[1]
    return None


def put(key: str, value, ttl: float) -> None:
    with _LOCK:
        _STORE[key] = (time.time() + ttl, value)


def clear(prefix: str | None = None) -> int:
    """Drop everything, or just the entries under one prefix."""
    with _LOCK:
        keys = [k for k in _STORE if prefix is None or k.startswith(prefix)]
        for k in keys:
            _STORE.pop(k, None)
    telemetry.record("cache", "cache.clear",
                     f"Cleared {len(keys)} cache entr{'y' if len(keys) == 1 else 'ies'}"
                     + (f" under {prefix}" if prefix else ""),
                     data={"prefix": prefix, "removed": len(keys)})
    return len(keys)


def stats() -> dict:
    now = time.time()
    with _LOCK:
        live = sum(1 for expiry, _ in _STORE.values() if expiry > now)
        return {"entries": len(_STORE), "live": live}
