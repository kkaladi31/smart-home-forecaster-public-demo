"""One pooled HTTP session shared by every outbound tool call.

`requests.get(...)` builds a brand-new connection each time, so every tool call
paid a fresh TCP handshake *and* a fresh TLS handshake. That is wasted time on
any network, and noticeably worse on machines where HTTPS is inspected (see
docs/ and .env.example) because the interception adds its own handshake.

A single `Session` keeps connections alive per host, so the second call to
api.weather.gov — and there are always at least two — reuses the first one's
socket. Retries are for transport-level failures only: the tools already
implement their own fallbacks for *bad answers* (NWS -> Open-Meteo, Google ->
Census), and this must not interfere with that logic or slow it down.

Import and use `SESSION.get(...)` exactly as you would `requests.get(...)`; the
exception types are unchanged (`requests.RequestException` still applies), so
existing error handling keeps working.
"""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 v2 and v1 expose Retry from different places
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover - very old urllib3
    Retry = None  # type: ignore[assignment]

# Enough connections for the parallel fan-out in tools/hazard_check.py without
# holding sockets open against a dozen hosts we only talk to occasionally.
_POOL_CONNECTIONS = 10
_POOL_MAXSIZE = 20


def _build_session() -> requests.Session:
    session = requests.Session()
    if Retry is not None:
        # Only retry things that are safe and clearly transient. Deliberately NOT
        # retrying 404/400: a "no data for this point" answer is a real answer and
        # the caller's own fallback should handle it immediately, not 3 seconds later.
        # One retry, not two. Retrying multiplies against the per-call timeout,
        # and these tools are chained: geocoding tries Google, then Census, then
        # Open-Meteo in sequence. At the old budget (2 retries x 20s) a single
        # unresponsive host cost ~60s and the full chain could exceed 180s —
        # which is exactly the stall this module's docstring promises not to
        # cause. Failing fast is what lets the *fallback* provide the resilience.
        retry = Retry(
            total=1,
            connect=1,
            read=0,
            status=0,
            backoff_factor=0.2,
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            pool_connections=_POOL_CONNECTIONS,
            pool_maxsize=_POOL_MAXSIZE,
            max_retries=retry,
        )
    else:  # pragma: no cover
        adapter = HTTPAdapter(pool_connections=_POOL_CONNECTIONS, pool_maxsize=_POOL_MAXSIZE)

    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _build_session()


def close() -> None:
    """Release pooled sockets (used by tests; the app keeps the session open)."""
    SESSION.close()
