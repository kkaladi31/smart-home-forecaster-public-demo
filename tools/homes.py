"""The home registry: every saved home, and which one the current turn is about.

The project started with a single `home_profile.json`. Supporting a second home
turns one question into two — *which* home, and *whose rules* — and the second
one is a safety question. Telling a Minneapolis owner that their Texas HOA
requires 40% living ground cover is not a small error; it is a confidently
delivered rule that does not apply to them, which is the exact failure mode
`docs/safety.md` lists as the headline retrieval hazard.

So the home is resolved the same way the persona is (see
`tools.agent_tools._current_persona`): from the run config the orchestrator sets,
never from a model-chosen argument. A model that picks its own jurisdiction
filter can pick the wrong one, and a wrong-jurisdiction answer looks exactly like
a right one. The UI knows which home the user is looking at, so it passes that
down and the model cannot get it wrong.

Profiles live one-per-file in `<data_root>/homes/`. There is deliberately no
index file: the primary is whichever profile carries `"is_primary": true`, so a
profile and its registry entry cannot drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import config

HOMES_DIR = config.homes_root()

# Corpus documents that are not tied to any one home (freeze physics, appliance
# care). Kept here rather than in rag_store so the registry owns the vocabulary.
COMMON_SCOPE = "common"


def _load_all() -> list[dict]:
    """Every profile on disk, primary first, then alphabetically by label."""
    homes = []
    for path in sorted(HOMES_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            home = json.load(f)
        home["_file"] = path.name
        homes.append(home)
    if not homes:
        raise RuntimeError(f"No home profiles found in {HOMES_DIR}")
    homes.sort(key=lambda h: (not h.get("is_primary", False), h.get("label", "")))
    return homes


def list_homes() -> list[dict]:
    """Short summaries for the UI's home switcher, primary first."""
    return [
        {
            "home_id": h["home_id"],
            "label": h.get("label") or h["home_id"],
            "address": h.get("address"),
            "is_primary": bool(h.get("is_primary")),
            "city": (h.get("jurisdiction") or {}).get("city"),
            "state": (h.get("jurisdiction") or {}).get("state"),
            "hoa": (h.get("hoa") or {}).get("association_name"),
        }
        for h in _load_all()
    ]


def primary_home_id() -> str:
    return _load_all()[0]["home_id"]


class UnknownHome(ValueError):
    """Raised when a specific home was asked for and does not exist."""

    def __init__(self, home_id: str, known: list[str]):
        self.home_id = home_id
        self.known = known
        super().__init__(
            f"No saved home with id {home_id!r}. Known homes: {', '.join(known)}.")


def resolve_home_id(home_id: str | None) -> str:
    """Map a requested id to a real one. `None` means "the primary home".

    An UNKNOWN id raises. This used to fall back to the primary home on the
    reasoning that a typo should not 500 the dashboard — but the fallback is the
    more dangerous behaviour, and it is the exact shape of failure the rest of
    this module exists to prevent.

    Asking about `demo-001` and silently receiving the primary home's covenants
    produces a confident, well-cited, completely wrong answer, and every
    downstream identity agrees on the wrong home so nothing detects it. The
    resolved id is echoed back in the payload, but a caller that got the id wrong
    is not the caller who will notice it was corrected.

    `None` is different and still means the primary home: that is an absence of
    preference, not a mistake. The failure mode being closed here is a *specific*
    request for a home that does not exist.
    """
    if not home_id:
        return primary_home_id()
    known = [h["home_id"] for h in _load_all()]
    if home_id not in known:
        raise UnknownHome(home_id, known)
    return home_id


def load_home(home_id: str | None = None) -> dict:
    """Return one full home profile — the active/primary home when `home_id` is None.

    The returned profile carries an `other_homes` list so a caller (or the model)
    can see that another home exists without a second lookup. That matters for
    "what about my other house?", which is otherwise unanswerable without the
    model guessing.
    """
    homes = _load_all()
    wanted = resolve_home_id(home_id)
    home = next(h for h in homes if h["home_id"] == wanted)
    home = {k: v for k, v in home.items() if k != "_file"}
    home["other_homes"] = [
        {"home_id": h["home_id"], "label": h.get("label"), "address": h.get("address")}
        for h in homes
        if h["home_id"] != wanted
    ]
    return home


def current_home_id() -> str:
    """The home this turn is about, read from the run config (not from the model).

    Outside a graph run — direct calls, the eval harness, `python -m` smoke tests
    — there is no config, so this is the primary home.
    """
    try:
        from langgraph.config import get_config

        configured = (get_config().get("configurable") or {}).get("home_id")
    except Exception:
        return primary_home_id()
    try:
        return resolve_home_id(configured)
    except UnknownHome:
        # Validation belongs at the boundary, not here. Both orchestrator entry
        # points resolve the id before it ever reaches the run config, so an
        # unknown value at this depth means a caller bypassed that check. Raising
        # from inside a tool mid-turn would surface as an opaque failure three
        # frames from the cause, so fall back loudly instead: the turn still
        # answers, and the log says which id was rejected.
        import telemetry

        telemetry.record(
            "safety", "home.unknown_in_config",
            f"Run config carried unknown home_id {configured!r}; using the primary home",
            level="warn", data={"requested": configured})
        return primary_home_id()


def current_home() -> dict:
    return load_home(current_home_id())


def home_scope_ids(home_id: str | None = None) -> list[str]:
    """The corpus scopes readable for a home: its own documents plus the shared ones.

    `COMMON_SCOPE` is always included, so narrowing to one home can never hide a
    rule that applies everywhere — the same conservative rule the audience filter
    follows in `memory.rag_store._build_where`.
    """
    return [resolve_home_id(home_id), COMMON_SCOPE]


if __name__ == "__main__":
    for summary in list_homes():
        flag = "PRIMARY" if summary["is_primary"] else "       "
        print(f"{flag}  {summary['home_id']:<12}  {summary['label']:<16}  {summary['address']}")
    print("\nActive home profile (no config -> primary):")
    active = load_home()
    print(json.dumps({k: active[k] for k in
                      ("home_id", "address", "climate_zone", "hoa", "other_homes")}, indent=2))
