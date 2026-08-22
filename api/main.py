"""FastAPI backend for the Smart-Home Forecaster web UI.

Run it from the project root:
    python -m uvicorn api.main:app --reload --port 8000

Design notes
------------
* **Chat is streamed** (Server-Sent Events). A single answer can take 30-100s
  because every tool call is a real API request, so the UI shows each tool firing
  live instead of a blank spinner. `/api/chat/stream` is the interesting endpoint.
* **Auth is a demo login only.** It exists to show an auth flow; it stores no real
  accounts and no personal data, which is deliberate — the capstone forbids
  putting private information in the project (see docs/safety.md).
* The dashboard reads the *same* tools the agent uses, so the two can never
  disagree about the weather.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import config
import telemetry
from api.dashboard import build_dashboard
from tools.alerts import get_weather_alerts
from tools.geocode import geocode_address, reverse_geocode, suggest_addresses
from tools.pros.core import find_pros
from tools.homes import UnknownHome, list_homes, load_home, resolve_home_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLOORPLAN_DIR = config.floorplans_root()

# --- demo auth ---------------------------------------------------------------
# Credentials come from the environment so nothing real is committed. These are
# demo-only; no user records, emails, or locations are persisted anywhere.
DEMO_USER = os.getenv("DEMO_USER", "demo")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "forecaster")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() not in ("0", "false", "no")
_ACTIVE_TOKENS: set[str] = set()

def _warm_up() -> None:
    """Pay the one-time startup costs before a user can be waiting on them.

    Two things load lazily and are slow exactly once: building the agent compiles
    the graph and opens SQLite, and the first RAG or semantic-cache query loads
    Chroma's MiniLM ONNX model. Left alone, both land on whoever asks the first
    question — which, during a demo or a recording, is the worst possible moment.

    Best-effort by design: a warm-up failure must never stop the server booting,
    because every one of these paths still works (just slower) on first use. A
    missing API key is the normal case here, not an error.
    """
    try:
        with telemetry.span("system", "warmup.agent", "Pre-building the agent graph"):
            get_agent()
    except Exception as exc:
        telemetry.record("system", "warmup.skipped",
                         f"Agent not pre-built: {type(exc).__name__}: {exc}",
                         level="debug")

    try:
        # One real query walks the entire retrieval stack: the MiniLM embedder,
        # the BM25 index build, and the reranker (which downloads ~23 MB of ONNX
        # weights the very first time the project is ever run). Doing it here means
        # a cold clone pays that once at boot instead of on the first question.
        with telemetry.span("system", "warmup.retrieval", "Warming the retrieval stack"):
            from memory.rag_store import search_policies

            search_policies("warm up the retrieval stack", k=1)
    except Exception as exc:
        telemetry.record("system", "warmup.skipped",
                         f"Retrieval not pre-loaded: {type(exc).__name__}: {exc}",
                         level="debug")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Off the event loop: both warm-up steps are blocking and would otherwise
    # stall startup for every other request the server is trying to accept.
    await asyncio.to_thread(_warm_up)
    yield


app = FastAPI(title="Smart-Home Forecaster API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # The browser now calls this API directly rather than through Vite's proxy
    # (see web/src/api.js), so both the dev server and the preview server need to
    # be allowed origins.
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent = None  # built lazily; reused across requests


@app.exception_handler(UnknownHome)
async def unknown_home_handler(request: Request, exc: UnknownHome):
    """A request naming a home that does not exist is a client error, not a crash.

    `resolve_home_id` raises rather than silently substituting the primary home,
    because answering a Dallas question out of Minneapolis's covenants
    produces a confident, well-cited, entirely wrong answer that nothing
    downstream can detect. The boundary's job is to turn that into a clear 404
    naming the ids that do exist, so the caller can fix the request.
    """
    telemetry.record("http", "http.unknown_home",
                     f"Rejected request for unknown home {exc.home_id!r}",
                     level="warn", data={"requested": exc.home_id, "known": exc.known})
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "requested": exc.home_id, "known_homes": exc.known},
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Time every request into the shared event log.

    The chat stream is excluded from the duration reading that matters — it is
    long-lived by design, so its own `agent` events describe the work. Logging
    the request itself is still useful for spotting a stream that never closed.
    """
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        telemetry.record(
            "http", "http.error", f"{request.method} {request.url.path} raised",
            level="error", duration_ms=(time.perf_counter() - started) * 1000,
            data={"error": f"{type(exc).__name__}: {exc}"},
        )
        raise

    elapsed = (time.perf_counter() - started) * 1000
    status = response.status_code
    # Two paths are skipped:
    #  * /api/logs — polling it would otherwise log itself once a second forever.
    #  * the chat stream — middleware returns as soon as the response *starts*,
    #    so its duration here is meaningless and reads as a fast request for an
    #    answer that took 30s. `stream.open`/`stream.close` time it properly.
    noisy = request.url.path.startswith("/api/logs") or request.url.path == "/api/chat/stream"
    if not noisy:
        telemetry.record(
            "http", "http.request", f"{request.method} {request.url.path} → {status}",
            level="error" if status >= 500 else "warn" if status >= 400 else "info",
            duration_ms=elapsed,
            data={"method": request.method, "path": request.url.path,
                  "status": status, "query": str(request.url.query or "")},
        )
    return response


def get_agent():
    """Build the agent once per process (it compiles a graph and opens SQLite)."""
    global _agent
    if _agent is None:
        with telemetry.span("system", "agent.build", "Building the agent graph"):
            from agents.orchestrator import build_agent

            _agent = build_agent()
    return _agent


def require_token(request: Request) -> None:
    """Bearer-token check for protected endpoints (no-op when auth is disabled)."""
    if not REQUIRE_AUTH:
        return
    header = request.headers.get("Authorization", "")
    token = header[7:] if header.startswith("Bearer ") else ""
    if token not in _ACTIVE_TOKENS:
        raise HTTPException(status_code=401, detail="Not authenticated")


# --- models ------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "web"
    persona: str = "owner"
    # The location currently shown on the dashboard. Answers follow what the user
    # is looking at, and it is part of the answer-cache identity so the same
    # question about a different place is not served from cache.
    location: str | None = None
    # Which saved home the question is about. The browser sends whatever the home
    # switcher has selected; an absent or unknown value resolves to the primary
    # home rather than erroring.
    home_id: str | None = None


class ModeRequest(BaseModel):
    # True = demo (free/open stack only), False = full.
    demo: bool


# --- auth --------------------------------------------------------------------
@app.post("/api/auth/login")
def login(body: LoginRequest) -> dict:
    if body.username != DEMO_USER or body.password != DEMO_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid demo credentials")
    token = secrets.token_urlsafe(24)
    _ACTIVE_TOKENS.add(token)
    return {"token": token, "username": body.username}


@app.post("/api/auth/logout")
def logout(request: Request) -> dict:
    header = request.headers.get("Authorization", "")
    _ACTIVE_TOKENS.discard(header[7:] if header.startswith("Bearer ") else "")
    return {"ok": True}


# --- status / profile --------------------------------------------------------
@app.get("/api/status")
def status() -> dict:
    """Public status used by the login screen and the sidebar."""
    try:
        from memory.rag_store import count as kb_count

        knowledge = kb_count()
    except Exception:
        knowledge = 0
    try:
        from memory.episodic import count as ep_count

        remembered = ep_count()
    except Exception:
        remembered = 0
    return {
        "model": config.active_model(),
        "mode": "demo" if config.demo_mode() else "full",
        "demo": config.demo_mode(),
        "has_llm_key": bool(config.OPENROUTER_API_KEY),
        "has_eia_key": bool(config.EIA_API_KEY),
        "knowledge_passages": knowledge,
        "remembered_interactions": remembered,
        "requires_auth": REQUIRE_AUTH,
    }


@app.get("/api/client-config", dependencies=[Depends(require_token)])
def client_config() -> dict:
    """Settings the browser needs at runtime.

    The Maps JavaScript API key has to reach the browser to work at all — that is
    how Google's client library is designed, and the real protection is an HTTP
    referrer restriction on the key, not secrecy. Serving it from here (rather
    than hardcoding it in the front-end) keeps it out of the committed source and
    behind the login. When it is absent the UI falls back to Leaflet/OpenStreetMap.

    Demo mode withholds the key entirely rather than merely asking the front-end
    to prefer Leaflet. A key the browser never receives cannot be billed by a
    stray component, so the free stack is enforced here rather than trusted.
    """
    enabled = config.google_enabled()
    return {
        "google_maps_key": config.GOOGLE_MAPS_API_KEY if enabled else None,
        "maps_provider": "google" if enabled else "leaflet",
    }


@app.get("/api/mode", dependencies=[Depends(require_token)])
def get_mode() -> dict:
    """Which stack is live, and the provider behind every capability."""
    return config.mode_report()


@app.post("/api/mode", dependencies=[Depends(require_token)])
def set_mode(body: ModeRequest) -> dict:
    """Switch between the full and free/open stacks without a restart.

    The cached agent is dropped so the next question is answered by the newly
    selected model — the agent captures its model at build time, so leaving the
    old instance in place would silently keep billing the paid one.

    A demo build refuses outright rather than silently returning success. It has
    no alternative stack to switch to: the keyed providers are gated off, the
    model is pinned to the free slug, and the real data is not on disk. Returning
    200 with `demo: true` — which is what it used to do — reports success for
    something that did not happen.
    """
    if config.is_demo_build():
        raise HTTPException(
            status_code=409,
            detail="This build runs one service stack. There is nothing to switch to.")

    previous = config.demo_mode()
    config.set_demo_mode(body.demo)
    if previous != config.demo_mode():
        global _agent
        _agent = None
        telemetry.record(
            "system", "mode.switch",
            f"Switched to {'demo (free/open)' if body.demo else 'full'} mode",
            data={"demo": body.demo, "model": config.active_model()},
        )
    return config.mode_report()


@app.get("/api/homes", dependencies=[Depends(require_token)])
def homes() -> dict:
    """Every saved home, primary first — the source for the UI's home switcher."""
    return {"homes": list_homes()}


@app.get("/api/profile", dependencies=[Depends(require_token)])
def profile(home_id: str | None = None) -> dict:
    """One home's profile. Defaults to the primary home when `home_id` is absent.

    `floorplans` lists only THIS home's plans. It used to glob a single shared
    directory, so every home was offered every plan and the sidebar's
    "first plan in the list" fallback could render the wrong house's layout —
    the same wrong-home hazard `docs/safety.md` R13 covers for retrieval. Plans
    now live in `floorplans/<home_id>/` and are returned as `<home_id>/<name>`.
    """
    home = load_home(home_id)
    resolved = home["home_id"]
    home_dir = FLOORPLAN_DIR / resolved
    plans = sorted(
        f"{resolved}/{p.name}"
        for p in home_dir.glob("*")
        if p.suffix in {".svg", ".png"}
    ) if home_dir.is_dir() else []
    return {"home": home, "floorplans": plans}


@app.get("/api/floorplan/{home_id}/{name}")
def floorplan(home_id: str, name: str):
    """Serve one home's floor-plan image.

    Deliberately NOT behind the auth gate: a browser `<img src=...>` cannot send
    an Authorization header, so requiring a token here just produces a broken
    image. These files are synthetic or public-domain and contain nothing
    sensitive.

    The path is still validated, and now against the HOME's directory rather than
    the floorplan root, so a traversal cannot walk sideways into another home.
    """
    from fastapi.responses import FileResponse

    home_dir = (FLOORPLAN_DIR / home_id).resolve()
    path = (home_dir / name).resolve()
    if not path.is_file() or path.parent != home_dir or FLOORPLAN_DIR.resolve() != home_dir.parent:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


# --- location ----------------------------------------------------------------
@app.get("/api/geocode/suggest", dependencies=[Depends(require_token)])
def geocode_suggest(
    q: str = Query(..., min_length=2),
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    """Autocomplete. `lat`/`lon` bias results toward the current view, so nearby
    matches rank first the way they do in a maps app."""
    return {"results": suggest_addresses(q, lat=lat, lon=lon)}


@app.get("/api/geocode", dependencies=[Depends(require_token)])
def geocode(address: str) -> dict:
    return geocode_address(address)


@app.get("/api/reverse-geocode", dependencies=[Depends(require_token)])
def reverse(lat: float, lon: float) -> dict:
    return reverse_geocode(lat, lon)


# --- dashboard ---------------------------------------------------------------
@app.get("/api/dashboard", dependencies=[Depends(require_token)])
def dashboard(
    lat: float | None = None,
    lon: float | None = None,
    address: str | None = None,
    hours: int = 48,
    home_id: str | None = None,
) -> dict:
    """Live weather metrics for a location, by coordinates or address."""
    label = None
    if lat is None or lon is None:
        if not address:
            # Default to the selected home so the dashboard is never empty.
            address = load_home(home_id)["address"]
        geo = _geocode_bounded(address)
        if not geo.get("ok"):
            raise HTTPException(status_code=400, detail=geo.get("error", "Could not geocode"))
        lat, lon, label = geo["latitude"], geo["longitude"], geo["matched_address"]
    else:
        label = reverse_geocode(lat, lon).get("label")

    data = build_dashboard(lat, lon, label=label, horizon_hours=hours)
    if not data.get("ok"):
        raise HTTPException(status_code=502, detail=data.get("error", "forecast unavailable"))
    return data


# --- provider fan-out budget --------------------------------------------------
# One slow upstream must never hold a dashboard request open. tools/http.py
# retries transport failures twice against a 20s HTTP_TIMEOUT, so a single stuck
# provider can cost ~60s on its own — which is how a mode switch ended up
# appearing to hang for minutes. These caps bound what the browser waits for.
DETAIL_TIMEOUT = 20.0   # the forecast itself — without it there is no dashboard
EXTRA_TIMEOUT = 6.0     # elevation / advisories / label — enrichments only
GEOCODE_TIMEOUT = 12.0  # the whole Google -> Census -> Open-Meteo chain


def _geocode_bounded(address: str) -> dict:
    """Geocode with a hard ceiling on the whole provider chain.

    geocode_address() walks up to three providers in sequence, so an unhealthy
    one used to be able to hold a dashboard request open for minutes. The cap is
    on the chain as a whole rather than per provider, because what a person is
    waiting on is the total.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        result = _settle(pool.submit(geocode_address, address), "geocoding", GEOCODE_TIMEOUT)
        return result or {"ok": False, "error": f"geocoding did not respond within {GEOCODE_TIMEOUT:.0f}s"}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _settle(future, what: str, timeout: float):
    """Wait for one provider, giving up instead of stalling the whole response.

    Returns None when the provider timed out or failed, leaving the caller to
    decide whether that piece was essential. Failures are recorded rather than
    raised so the Logs tab still shows which provider was the slow one.
    """
    if future is None:
        return None
    try:
        return future.result(timeout=timeout)
    except FutureTimeout:
        telemetry.record(
            "api", "provider.timeout",
            f"{what} did not respond within {timeout:.0f}s — continuing without it",
            level="warn",
        )
        return None
    except Exception as exc:
        telemetry.record(
            "api", "provider.error", f"{what} failed: {type(exc).__name__}: {exc}",
            level="warn",
        )
        return None


@app.get("/api/weather", dependencies=[Depends(require_token)])
def weather(
    lat: float | None = None,
    lon: float | None = None,
    address: str | None = None,
    home_id: str | None = None,
) -> dict:
    """Rich weather detail: current, hourly, and 8 days of daily summaries.

    Returns everything the UI needs for the 24h / 48h / 7-day views and the
    per-day detail panel (sunrise/sunset, UV, AQI, pollen, dew point, pressure,
    visibility, moon phase, yesterday comparison, running conditions).
    """
    from tools.weather_detail import get_weather_detail

    label = None
    if lat is None or lon is None:
        # Coordinates are needed before anything else can start, so this one
        # lookup is unavoidably sequential.
        if not address:
            address = load_home(home_id)["address"]
        geo = _geocode_bounded(address)
        if not geo.get("ok"):
            raise HTTPException(status_code=400, detail=geo.get("error", "Could not geocode"))
        lat, lon, label = geo["latitude"], geo["longitude"], geo["matched_address"]

    # Hazard ratings come from the same deterministic assessors the agent uses,
    # over the next 48h, so the dashboard and the agent always agree.
    from tools.elevation import get_elevation
    from tools.freeze_risk import assess_freeze_risk
    from tools.heat_risk import assess_heat_risk

    # Everything below only needs the coordinates, so it all runs at once instead
    # of four sequential round trips. The reverse geocode joins the batch only
    # when the caller gave coordinates without a label.
    needs_label = label is None
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        f_detail = pool.submit(get_weather_detail, lat, lon)
        f_elev = pool.submit(get_elevation, lat, lon)
        f_alerts = pool.submit(get_weather_alerts, lat, lon)
        f_label = pool.submit(reverse_geocode, lat, lon) if needs_label else None

        data = _settle(f_detail, "forecast", DETAIL_TIMEOUT) or {
            "ok": False, "error": f"forecast provider did not respond within {DETAIL_TIMEOUT:.0f}s",
        }
        # The three below are enrichments. A dashboard without an elevation
        # figure or an advisories list is still useful; one that never arrives
        # is not, so a slow provider is dropped rather than waited on.
        elev = _settle(f_elev, "elevation", EXTRA_TIMEOUT) or {"ok": False}
        alerts = _settle(f_alerts, "alerts", EXTRA_TIMEOUT) or {"ok": False}
        if f_label is not None:
            label = (_settle(f_label, "reverse geocode", EXTRA_TIMEOUT) or {}).get("label")
    finally:
        # wait=False is the point of the timeouts above. The default shutdown
        # blocks until every worker finishes, so a provider that hangs would
        # still hold the whole request open — the timeouts would buy nothing.
        pool.shutdown(wait=False, cancel_futures=True)

    if not data.get("ok"):
        raise HTTPException(status_code=502, detail=data.get("error", "forecast unavailable"))

    elevation_ft = elev.get("elevation_ft") if elev.get("ok") else None

    upcoming = [h for h in data["hourly"] if not h["is_past"]][:48]
    if upcoming:
        coldest = min(upcoming, key=lambda h: h["temp_f"])
        hottest = max(upcoming, key=lambda h: h["temp_f"])
        data["freeze"] = assess_freeze_risk(
            coldest["temp_f"], min_wind_mph=coldest.get("wind_mph"),
            elevation_ft=elevation_ft,
        )
        data["heat"] = assess_heat_risk(hottest["temp_f"], humidity_pct=hottest.get("humidity"))
        data["range"] = {
            "min_temp_f": coldest["temp_f"], "min_temp_time": coldest["time"],
            "max_temp_f": hottest["temp_f"], "max_temp_time": hottest["time"],
        }

    data["location"] = {"label": label, "latitude": lat, "longitude": lon,
                        "elevation_ft": elevation_ft}
    data["alerts"] = alerts.get("alerts", []) if alerts.get("ok") else []
    data["alerts_available"] = bool(alerts.get("ok"))
    return data


@app.get("/api/contractors", dependencies=[Depends(require_token)])
def contractors(trade: str | None = None, limit: int = 6,
                home_id: str | None = None) -> dict:
    """Licensed professionals serving one home's area — the primary home's by default.

    Returns `withheld` alongside `results`, and that is the point of the endpoint
    rather than a detail of it. The gate in `tools/pros` refuses any contractor
    whose registration is not current, and `find_contractors` deliberately hides
    those from the Advisor — a professional the system must not recommend has no
    business in a model's prompt. But a *person* reading the panel is owed the
    opposite: "four roofers nearby, all with lapsed registrations" is far more
    useful than an empty list, and it is the honest answer.

    So the two consumers get different views on purpose. `trade` accepts either a
    trade name or a whole natural-language need.
    """
    results = find_pros(trade, home_id=resolve_home_id(home_id), limit=limit)
    return {
        "trade": results.trade,
        "trade_label": results.trade_label,
        "source": results.source,
        "area": results.jurisdiction,
        "notes": results.notes,
        "results": [_pro_payload(p) for p in results.eligible],
        "withheld": [_pro_payload(p) for p in results.withheld],
    }


def _pro_payload(pro) -> dict:
    """One professional, flattened for the UI.

    `specialty` is stripped for display but the underlying value is not — the
    registry publishes one specialty with a trailing space, and a citation should
    quote what the registry actually says.
    """
    return {
        "name": pro.name,
        "match": pro.match,
        "license_status": pro.license_status,
        "license_number": pro.license_number,
        "license_expires": pro.license_expires,
        "license_type": pro.license_type,
        "specialty": (pro.specialty or "").strip() or None,
        "city": pro.city,
        "phone": pro.phone,
        "rating": pro.rating,
        "reviews": pro.review_count,
        "bond_usd": pro.bond_amount,
        "eligible": pro.eligible,
        "withheld_reason": pro.withheld_reason,
        "hourly_rate_usd": pro.extra.get("hourly_rate_usd"),
        "availability": pro.extra.get("availability"),
    }


# --- chat (streamed) ---------------------------------------------------------
@app.post("/api/chat/stream", dependencies=[Depends(require_token)])
def chat_stream(body: ChatRequest) -> StreamingResponse:
    """Stream one agent turn as Server-Sent Events.

    Each event is `data: <json>` — see agents.orchestrator.stream_answer for the
    event shapes (guardrail / memory / tool_call / tool_result / answer / done).
    """
    from agents.orchestrator import stream_answer

    def event_source():
        started = time.perf_counter()
        telemetry.record("http", "stream.open", "Chat stream opened",
                         thread_id=body.thread_id,
                         data={"persona": body.persona, "location": body.location,
                               "home_id": body.home_id})
        try:
            agent = get_agent()
        except RuntimeError as exc:  # missing API key, etc.
            telemetry.record("http", "stream.error", f"Could not build the agent: {exc}",
                             level="error", thread_id=body.thread_id)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'blocked': False})}\n\n"
            return

        count = 0
        try:
            for event in stream_answer(
                body.message, persona=body.persona, agent=agent,
                thread_id=body.thread_id, location=body.location,
                home_id=body.home_id,
            ):
                count += 1
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            # `finally` so a client that navigates away mid-answer is recorded as
            # a closed stream rather than vanishing from the log.
            telemetry.record("http", "stream.close", f"Chat stream closed ({count} events)",
                             thread_id=body.thread_id,
                             duration_ms=(time.perf_counter() - started) * 1000,
                             data={"events": count})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/conversations", dependencies=[Depends(require_token)])
def conversations(limit: int = 30) -> dict:
    """Past conversations, newest first — sourced from episodic memory."""
    from memory.episodic import list_threads

    return {"conversations": list_threads(limit=limit)}


@app.get("/api/conversations/{thread_id}", dependencies=[Depends(require_token)])
def conversation(thread_id: str) -> dict:
    """Replay one conversation. Resuming this thread_id also restores the
    agent's own memory of it, so follow-ups still work."""
    from memory.episodic import thread_messages

    return {"thread_id": thread_id, "messages": thread_messages(thread_id)}


# --- logs --------------------------------------------------------------------
@app.get("/api/logs", dependencies=[Depends(require_token)])
def logs(
    since: int = 0,
    group: str | None = None,
    level: str | None = None,
    limit: int = 500,
) -> dict:
    """Tail the backend event log.

    Clients page with `since=<latest_seq>` rather than by timestamp: sequence
    numbers are unique and monotonic, so a poll can never skip or repeat an event
    the way a millisecond-resolution clock allows.
    """
    return telemetry.snapshot(since=since, source="backend", group=group,
                              level=level, limit=limit)


@app.get("/api/logs/stats", dependencies=[Depends(require_token)])
def log_stats() -> dict:
    return telemetry.stats()


@app.delete("/api/logs", dependencies=[Depends(require_token)])
def clear_logs() -> dict:
    return {"ok": True, "removed": telemetry.clear()}


# --- maintenance -------------------------------------------------------------
class ClearChatRequest(BaseModel):
    thread_id: str


@app.get("/api/admin/stats", dependencies=[Depends(require_token)])
def admin_stats() -> dict:
    """What the destructive buttons would actually delete, so the UI can say so."""
    from memory import episodic, semantic_cache
    from tools import cache

    try:
        conversations = len(episodic.list_threads(limit=1000))
    except Exception:
        conversations = 0
    try:
        interactions = episodic.count()
    except Exception:
        interactions = 0
    return {
        "conversations": conversations,
        "remembered_interactions": interactions,
        "tool_cache": cache.stats(),
        "semantic_cache_entries": semantic_cache.count(),
        "log_events": telemetry.stats()["total"],
    }


@app.post("/api/admin/clear-chat", dependencies=[Depends(require_token)])
def clear_chat(body: ClearChatRequest) -> dict:
    """Forget one conversation — its checkpoint and its episodic rows.

    Scoped to a single thread on purpose. Wiping everything the assistant knows
    is a separate, clearly-labelled button; the two should never be one click
    apart in behaviour.
    """
    from agents.orchestrator import forget_thread

    return {"ok": True, **forget_thread(body.thread_id, agent=_agent)}


@app.post("/api/admin/clear-memory", dependencies=[Depends(require_token)])
def clear_memory() -> dict:
    """Erase all episodic memory: every past interaction and its semantic index.

    The RAG knowledge base is deliberately NOT touched — that is ingested
    reference material, not something the user said, and rebuilding it needs a
    separate `python ingest.py` run.
    """
    from memory import episodic

    before = episodic.count()
    episodic.clear_all()
    telemetry.record("system", "memory.cleared",
                     f"Episodic memory cleared ({before} interactions)",
                     level="warn", data={"deleted": before})
    return {"ok": True, "deleted": before}


@app.post("/api/admin/clear-cache", dependencies=[Depends(require_token)])
def clear_cache() -> dict:
    """Drop both answer caches and every cached tool result.

    Useful when a demo needs to show the real latency, or when a forecast is
    being served from a cache entry that has not expired yet.
    """
    from memory import semantic_cache
    from tools import cache

    removed = cache.clear()
    # Reported from what `clear()` actually removed, and confirmed by re-counting.
    # This used to read the count BEFORE clearing and return that, so the button
    # reported success even on the runs where the semantic cache did not clear at
    # all — the one failure mode a "make it slow again" control must not have.
    semantic_entries = semantic_cache.clear()
    still_there = semantic_cache.count()

    level = "warn" if not still_there else "error"
    message = (f"Caches cleared ({removed} tool/answer entries, "
               f"{semantic_entries} semantic)")
    if still_there:
        message += f" — {still_there} semantic entries SURVIVED the clear"
    telemetry.record("system", "cache.cleared", message, level=level,
                     data={"entries": removed, "semantic_entries": semantic_entries,
                           "semantic_remaining": still_there})
    return {"ok": not still_there, "entries": removed,
            "semantic_entries": semantic_entries, "semantic_remaining": still_there}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
