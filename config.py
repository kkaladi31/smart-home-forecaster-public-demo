"""Central configuration for the Smart-Home Forecaster agent.

All secrets come from a local .env file (never committed). See .env.example.
The only thing you *must* set to run the full agent is OPENROUTER_API_KEY.
The raw data tools (geocode / elevation / weather) work with NO key at all.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

# --- LLM access (via OpenRouter; free-tier models are fine) --------------------
# OpenRouter is OpenAI-API-compatible, so we point an OpenAI client at its URL.
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Default model. Claude Sonnet 5 is dramatically better at multi-step tool use
# than the free tier and is cheaper than Sonnet 4.5 ($2/$10 per M vs $3/$15).
#
# Cost-free alternative: set LLM_MODEL to whatever FREE_LLM_MODEL currently names
# (below) — the demo build runs on it and everything works without spending
# anything. Free slugs rotate and DO get withdrawn without notice; list current
# ones at https://openrouter.ai/models?max_price=0 filtering for `tools` support,
# or run `python scripts/check_free_model.py`.
LLM_MODEL: str = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-5")

# --- Run mode: "full" vs "demo" ------------------------------------------------
# Demo mode pins the app to services that cost nothing. It is a *presentation*
# switch, not a feature flag: no capability is removed, only the provider behind
# it changes.
#
# Free government APIs stay ON in both modes — NWS forecasts and alerts, EIA
# energy prices, Census geographies. They cost nothing, and turning them off
# would weaken the demo for no benefit. What demo mode drops is the metered
# stack: the paid LLM and the billable Google Maps Platform services.
#
# NOTE: in a demo BUILD (SHF_PROFILE=demo) this switch is forced on and cannot be
# turned off — see demo_mode(). The published artifact's requirement is free
# providers only, and a requirement that depends on remembering to flip a toggle
# is not a requirement. In a full build it behaves as described above.
# The free model the demo build runs on.
#
# CHANGED 2026-08-21, under duress: OpenRouter withdrew `openai/gpt-oss-20b:free`
# from the free tier and the slug now 404s with "This model is unavailable for
# free". Every LLM-dependent evaluation case failed at once and the demo could
# not answer a single question. `.env.example` had warned that free slugs rotate;
# this is what that looks like when it happens.
#
# The replacement was chosen by measurement, not reputation — see
# FREE_LLM_FALLBACKS. On the two questions that define the demo it is roughly an
# order of magnitude faster than what it replaces:
#
#     high-risk referral   12.8 s / 8.3 s   (was 182-202 s)
#     flagship freeze       6.5 s / 3.5 s
#
# Both called the correct tools and produced grounded, cited answers.
FREE_LLM_MODEL: str = os.getenv("FREE_LLM_MODEL",
                                "nvidia/nemotron-3-super-120b-a12b:free")

# Verified working alternatives, best first, for when the one above is withdrawn
# or rate-limited. NOT an automatic failover — the demo pins one model on purpose
# so a recorded run is reproducible, and silently swapping models mid-demo would
# make the trace unexplainable. This is the list to pick the next pin from, and
# `python scripts/check_free_model.py` is what tells you the pin has gone bad
# before an audience does.
#
# Measured 2026-08-21 on "Am I allowed to replace my backyard grass with stones?"
# (needs a tool call, retrieval grounding and a citation):
#     nvidia/nemotron-3-super-120b-a12b:free    6 s   correct tool, cited
#     nvidia/nemotron-3-nano-30b-a3b:free      12 s   correct tool, cited
#     nvidia/nemotron-3.5-lightning:free       15 s   correct tool, cited
# google/gemma-4-*:free and z-ai/glm-5.2:free returned 429 at the time and were
# neither confirmed working nor ruled out.
FREE_LLM_FALLBACKS: tuple[str, ...] = (
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "z-ai/glm-5.2:free",
)

# Startup default. The UI can flip this at runtime via set_demo_mode(); the env
# var is what CLI runs (main.py) and the eval suite honour, since they have no UI.
DEMO_MODE: bool = os.getenv("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

_demo_mode: bool = DEMO_MODE


def demo_mode() -> bool:
    """True when the app is pinned to the free/open stack.

    A demo BUILD is always in demo mode. The published artifact's requirement is
    free providers only, and that must not depend on a toggle someone can flip in
    the sidebar — or forget to flip. `SHF_PROFILE=demo` therefore implies the
    free stack, and the runtime switch can only ever turn it *on* in a full build.

    Deliberately a function, not a module-level constant. Callers that did
    `from config import SOME_FLAG` would capture the value at import time and
    never see a runtime flip, which is exactly the bug this avoids.
    """
    return _demo_mode or is_demo_build()


def set_demo_mode(enabled: bool) -> bool:
    """Switch modes at runtime. Returns the state actually in effect.

    Turning demo mode OFF in a demo build is refused rather than obeyed: there is
    no paid stack behind it to switch to, and a half-applied switch is worse than
    a disabled one. Callers should compare the returned value to what they asked
    for; the API surfaces the refusal rather than reporting a silent success.
    """
    global _demo_mode
    if is_demo_build() and not enabled:
        return True  # refused — a demo build has no full stack
    _demo_mode = bool(enabled)
    return demo_mode()


# A process-wide model override, for DIAGNOSTICS ONLY.
#
# Set by `eval/run_eval.py --model`, and by nothing else. It exists to separate
# two questions the suite otherwise conflates: *is the code correct?* and *does
# the graded artifact work on a free model?* A failure on the free model may be a
# product bug or a weak-model limit, and the only way to tell them apart is to
# run the same cases on a strong one.
#
# Deliberately NOT reachable from the API or the UI. A demo build exists to prove
# the artifact runs on free tokens; a runtime switch that let it quietly answer on
# a paid model would defeat the one constraint the build profile is there to
# enforce. `set_demo_mode` refuses for the same reason.
#
# It also overrides the FULL profile's model, so a run is never a mixture: sub-
# agents resolve through `active_model()` too, and an override that only reached
# the orchestrator would leave the Advisor and Critic on a different model than
# the turn that called them.
_model_override: str | None = None


def set_model_override(slug: str | None) -> str | None:
    """Force one model for this process. Returns what is now in effect."""
    global _model_override
    _model_override = (slug or "").strip() or None
    return _model_override


def model_override() -> str | None:
    return _model_override


def active_model() -> str:
    """The model to use right now — free slug in demo mode, else the full one."""
    if _model_override:
        return _model_override
    return FREE_LLM_MODEL if demo_mode() else LLM_MODEL


# Serve web research from a frozen snapshot instead of the live web.
#
# OFF by default, because live evidence is the Researcher's entire point. It
# exists for the two situations where the live web is a liability rather than a
# feature:
#
#   * a GRADED run, which must be reproducible. A research case otherwise
#     asserts on what the web returned that day, so a grader re-running it months
#     later gets different sources and the case passes or fails on the internet's
#     mood rather than on this code.
#   * a RECORDING, where a scrape that rate-limits mid-take ruins it.
#
# DuckDuckGo is not an API — `ddgs` scrapes HTML endpoints, which is why
# `providers._UNAVAILABLE` exists at all. Depending on it for a deliverable is
# depending on a scraper staying unblocked at the moment someone presses record.
_research_fixtures: bool = os.getenv(
    "SHF_RESEARCH_FIXTURES", "").strip().lower() in {"1", "true", "yes", "on"}


def research_fixtures_enabled() -> bool:
    return _research_fixtures


def set_research_fixtures(enabled: bool) -> bool:
    global _research_fixtures
    _research_fixtures = bool(enabled)
    return _research_fixtures


def graded_model() -> str:
    """The model the graded artifact runs on, ignoring any diagnostic override.

    The ledger compares against this rather than `active_model()`: a pass earned
    under `--model` is evidence about the code, not about the artifact, and
    counting the two together would let a green Sonnet run masquerade as proof
    that the free-tier deliverable works.
    """
    return FREE_LLM_MODEL if demo_mode() else LLM_MODEL


# --- Build profile: which DATA PLANE this build is wired to --------------------
# This is NOT the same switch as demo_mode() above, and conflating them is a bug.
#
#   demo_mode()      is a PRESENTATION switch. Runtime-flippable from the sidebar
#                    (POST /api/mode). Decides which provider answers and which
#                    model slug is used. Removes no capability.
#
#   build_profile()  is a DATA-PLANE switch. Read ONCE at import from the
#                    environment and never flippable at runtime. Decides which
#                    data root and which vector/SQLite state this process reads.
#
# Why they must stay separate: if the data root keyed off demo_mode(), a stray
# sidebar toggle would repoint the app at a data tree that isn't there, or — far
# worse — swap corpora mid-conversation while the vector store still held the
# other profile's embeddings. Answering a Minneapolis question out of a Bonney
# Lake index is a wrong answer that looks exactly like a right one.
#
# The "demo" profile is the DEFAULT because it is the one that cannot leak. A
# fresh clone of the public repo therefore just works; the private full build
# opts in explicitly with SHF_PROFILE=full in .env.
PROFILES = ("demo", "full")

_profile_raw = os.getenv("SHF_PROFILE", "demo").strip().lower()
if _profile_raw not in PROFILES:
    raise RuntimeError(
        f"SHF_PROFILE={_profile_raw!r} is not valid. Use one of: {', '.join(PROFILES)}.\n"
        "  demo = 100% synthetic data, free providers only (the public build)\n"
        "  full = real data, keyed providers allowed (private build only)"
    )
BUILD_PROFILE: str = _profile_raw


def build_profile() -> str:
    """Which data plane this process is wired to: "demo" or "full"."""
    return BUILD_PROFILE


def is_demo_build() -> bool:
    """True when this process must never touch real-world data."""
    return BUILD_PROFILE == "demo"


class RealDataDisabled(RuntimeError):
    """Raised when a demo build reaches for a real-world data provider."""


def require_real_data(feature: str) -> None:
    """Gate every real-world provider. Call this on entry, before any network I/O.

    The filesystem separation is the primary control — a demo build has no real
    document on disk to read. This is the second one, for the case where code
    reaches for a live API rather than a file.
    """
    if is_demo_build():
        raise RealDataDisabled(
            f"{feature!r} needs real-world data, but this is a demo build "
            f"(SHF_PROFILE={BUILD_PROFILE}). Demo builds are 100% synthetic by design."
        )


# --- Data + state roots ---------------------------------------------------------
# Both trees are keyed directly off the profile name, so the mapping is one line
# and there is no lookup table to get out of step:
#
#   data/demo/  data/full/      corpus, home profiles, floor plans, fixtures
#   state/demo/ state/full/     Chroma, episodic SQLite, conversation checkpoints
#
# This separation IS the primary control on the "demo carries no real data" rule.
# A demo build does not filter real documents out — it has none on disk to read,
# and `data/full/` is never copied into the published tree at all. The
# provider_allowed()/require_real_data() gates below are the second line, for the
# case where code reaches for a live API instead of a file.
# Overrides exist so a test can point the whole tree at a tmpdir without
# mutating global state. Read once, alongside the profile, for the same reason.
_DATA_ROOT_OVERRIDE = os.getenv("SHF_DATA_ROOT", "").strip() or None
_STATE_ROOT_OVERRIDE = os.getenv("SHF_STATE_ROOT", "").strip() or None


def data_root() -> Path:
    """Root of the corpus/home/fixture tree for this build profile."""
    if _DATA_ROOT_OVERRIDE:
        return Path(_DATA_ROOT_OVERRIDE).resolve()
    return PROJECT_ROOT / "data" / BUILD_PROFILE


def homes_root() -> Path:
    """Directory of one-JSON-per-home profiles."""
    return data_root() / "homes"


def corpus_root() -> Path:
    """Directory of per-scope RAG corpora (one subdirectory per home_scope)."""
    return data_root() / "corpus"


def floorplans_root() -> Path:
    """Root of floor-plan images. One subdirectory per home_id — plans are NOT
    pooled in a shared directory, because a shared directory means a home with no
    plan of its own silently renders someone else's."""
    return data_root() / "floorplans"


def fixtures_root() -> Path:
    """Recorded provider responses used when a live provider is unavailable."""
    return data_root() / "fixtures"


def state_root() -> Path:
    """Root of generated state: vector store, episodic memory, checkpoints.

    Separate per profile so a demo build physically cannot read embeddings
    generated from real documents — the index itself is a different file tree.
    """
    if _STATE_ROOT_OVERRIDE:
        return Path(_STATE_ROOT_OVERRIDE).resolve()
    return PROJECT_ROOT / "state" / BUILD_PROFILE


def chroma_dir() -> Path:
    """Chroma PersistentClient path. Shared by three collections — the RAG store,
    the semantic answer cache, and episodic memory — so it moves as one unit."""
    return state_root() / "chroma"


def episodic_db() -> Path:
    """SQLite file behind episodic memory."""
    return state_root() / "home.db"


def conversations_db() -> Path:
    """SQLite file behind the LangGraph conversation checkpointer."""
    return state_root() / "conversations.db"


# --- Optional keys (used in later phases) --------------------------------------
# .strip() guards against a trailing space/newline sneaking in when pasting a key.
def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value else None


EIA_API_KEY: str | None = _env("EIA_API_KEY")          # Phase 4: energy prices
TAVILY_API_KEY: str | None = _env("TAVILY_API_KEY")    # optional web search

# Google Maps Platform. Optional: when absent the app falls back to the
# no-key stack (Open-Meteo + Leaflet/OpenStreetMap + RainViewer), so the project
# stays runnable by anyone who clones the repo without a billing account.
GOOGLE_MAPS_API_KEY: str | None = _env("GOOGLE_MAPS_API_KEY")


# Which providers exist, what they need, and whether they touch real-world data.
#   key       — env var that must be present, or None if the provider needs none
#   real_data — True when using it means reading real-world property/business data
#               (as opposed to real *weather*, which is fine in a demo build)
_PROVIDERS: dict[str, dict] = {
    "google_maps": {"key": "GOOGLE_MAPS_API_KEY", "real_data": False},
    "google_places": {"key": "GOOGLE_MAPS_API_KEY", "real_data": True},
    "youtube": {"key": "YOUTUBE_API_KEY", "real_data": True},
    "tavily": {"key": "TAVILY_API_KEY", "real_data": False},
    "eia": {"key": "EIA_API_KEY", "real_data": False},
    "duckduckgo": {"key": None, "real_data": False},
    "lni_wa": {"key": None, "real_data": True},
    "parcel_pierce": {"key": None, "real_data": True},
    "doc_acquisition": {"key": None, "real_data": True},
}


def provider_allowed(name: str) -> bool:
    """Whether `name` may be used right now. AND-ed, never OR-ed.

    Three independent conditions, all of which must hold:
      1. a demo BUILD may never use a real-world-data provider, even with a key;
      2. demo MODE drops metered providers (today's cost switch);
      3. the provider's key, if it needs one, is actually configured.
    """
    spec = _PROVIDERS.get(name)
    if spec is None:
        raise KeyError(f"unknown provider {name!r}; known: {', '.join(sorted(_PROVIDERS))}")
    if spec["real_data"] and is_demo_build():
        return False
    key_name = spec["key"]
    if key_name is None:
        return True
    if demo_mode():
        # Every keyed provider here is either metered or an account we don't want
        # a cost-free demo leaning on. Free government APIs need no key and so
        # never reach this branch.
        return False
    return bool(_env(key_name))


def google_enabled() -> bool:
    """True when the billable Google Maps Platform stack should be used.

    Single gate for geocoding, weather detail, air quality, and map tiles. In
    demo mode this is False even when a key is configured, so the app exercises
    its no-key fallbacks (Nominatim, Open-Meteo, Leaflet/OpenStreetMap) — the
    same path anyone cloning the repo without a billing account would hit.
    """
    return provider_allowed("google_maps")


def service_matrix() -> list[dict]:
    """Which provider is behind each capability right now, for the UI panel.

    `tier` is the cost story: "gov" = free public agency API, "free" = free
    third-party or local, "billed" = metered account. Demo mode should show no
    "billed" rows at all — that is the claim the panel lets a viewer verify.
    """
    google = google_enabled()
    return [
        {"service": "Language model", "provider": active_model(),
         "tier": "free" if demo_mode() else "billed"},
        {"service": "Forecast + alerts", "provider": "NWS (weather.gov)", "tier": "gov"},
        {"service": "Energy prices",
         "provider": "EIA open data" if EIA_API_KEY else "published averages",
         "tier": "gov" if EIA_API_KEY else "free"},
        # Named honestly: there is no FCC call anywhere in this project. The row
        # said "Census + FCC" for months while only the Census geocoder existed,
        # and this list is rendered to the user as a statement of what is running.
        # Add FCC back here when tools/jurisdiction.py actually calls it.
        {"service": "Jurisdiction", "provider": "US Census geographies", "tier": "gov"},
        # The free chain is the federal Census geocoder first (authoritative for
        # US street addresses), falling back to Open-Meteo for place names.
        # Nominatim is deliberately not used — see the note in tools/geocode.py.
        {"service": "Geocoding",
         "provider": "Google Geocoding" if google else "US Census + Open-Meteo",
         "tier": "billed" if google else "gov"},
        {"service": "Weather detail",
         "provider": "Google Weather + Air Quality" if google else "Open-Meteo",
         "tier": "billed" if google else "free"},
        {"service": "Map tiles",
         "provider": "Google Maps" if google else "Leaflet / OpenStreetMap",
         "tier": "billed" if google else "free"},
        {"service": "Web search", "provider": "DuckDuckGo", "tier": "free"},
        {"service": "Embeddings", "provider": "MiniLM (local)", "tier": "free"},
    ]


def mode_report() -> dict:
    """Everything the UI needs to describe the current mode.

    **A DEMO BUILD DOES NOT DESCRIBE A FULL MODE — IT DOES NOT MENTION ONE.**

    The earlier version always returned `build_profile`, `mode_locked`,
    `google_enabled` and `has_google_key`, so the published demo rendered a
    "Full — best quality, uses metered services" button that did nothing when
    clicked. That is worse than either alternative: it advertises a paid stack
    the artifact does not have, and it is a visibly broken control in the thing
    a reviewer is looking at.

    Omission rather than a disabled flag is the point. A `mode_locked: true`
    field still tells the reader a locked door exists. The demo payload simply
    has no concept of another mode, which is also the truth — a demo build has
    no full stack behind it, `set_demo_mode(False)` refuses, and the real data
    is not on disk.

    A full build returns `switchable: true` and the fields the toggle needs, so
    it can present both. That asymmetry is deliberate and is the reconciliation:
    demo shows demo; full shows full *and* demo.
    """
    report = {
        "demo": demo_mode(),
        "mode": "demo" if demo_mode() else "full",
        "model": active_model(),
        "services": service_matrix(),
    }
    if not is_demo_build():
        report.update({
            "switchable": True,
            "build_profile": build_profile(),
            "google_enabled": google_enabled(),
            "has_google_key": bool(GOOGLE_MAPS_API_KEY),
        })
    return report


# --- HTTP etiquette ------------------------------------------------------------
# The US National Weather Service asks every client to send a descriptive
# User-Agent with contact info. Change the email to your own.
CONTACT_EMAIL: str = os.getenv("CONTACT_EMAIL", "capstone-student@example.com")
USER_AGENT: str = f"smart-home-forecaster/0.1 ({CONTACT_EMAIL})"

# Default request timeout (seconds) for all outbound API calls.
# 8s, not 20s. Every upstream here answers in well under a second when healthy,
# so a long timeout only decides how long a *sick* one blocks a person staring at
# a loading dashboard. The tools all have fallbacks; reaching them quickly matters
# more than giving any single provider a generous window.
HTTP_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", "8"))


def require_llm_key() -> str:
    """Return the OpenRouter key or raise a clear, beginner-friendly error."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set.\n"
            "1) Get a free key at https://openrouter.ai/keys\n"
            "2) Copy .env.example to .env\n"
            "3) Paste your key after OPENROUTER_API_KEY=\n"
            "Note: the data tools (geocode/elevation/weather) run without any key."
        )
    return OPENROUTER_API_KEY
