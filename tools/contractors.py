"""Contractor lookup — now a thin adapter over the licence-gated Pro Finder.

**This module used to be the bug.** It mapped natural language to trades with a
dict of substrings and `if key in query`, so the entry `"ac": "hvac"` matched
"repl**ac**e my lawn" and "surf**ac**e"; and because the Advisor passes the
user's *entire question* as the `trade` argument, a lawn question could return an
HVAC company. It then sorted by star rating and returned the top row — with no
concept of a licence at all, which meant the project's claim that it never
recommends an unlicensed contractor was simply not true of the code.

Both faults are fixed by delegation rather than by patching:

* trade identification moved to `tools/pros/trades.py`, which matches on token
  boundaries via `tools/phrases.py` and returns **every** plausible trade instead
  of the first substring to hit;
* the licence gate lives in `tools/pros/core.py`, and this function returns
  **eligible professionals only**. A contractor whose registration is not current
  can no longer reach the Advisor's prompt at all.

The return shape is unchanged — `agents/advisor.py` reads `name`, `trade`,
`rating` and `hourly_rate_usd` from these dicts — so the fix reaches the Advisor
without touching it. Callers wanting the withheld list and its reasons should use
`tools.pros.core.find_pros` directly; this adapter deliberately hides them,
because the Advisor's prompt should never contain a professional it must not
recommend.
"""
from __future__ import annotations

from tools.pros.core import find_pros


def _to_legacy_row(pro) -> dict:
    """One `Pro` in the dict shape this module has always returned."""
    row = {
        "name": pro.name,
        "trade": pro.extra.get("trade") or (pro.specialty or "").strip().lower(),
        "rating": pro.rating,
        "reviews": pro.review_count,
        "hourly_rate_usd": pro.extra.get("hourly_rate_usd"),
        "min_job_usd": pro.extra.get("min_job_usd"),
        "availability": pro.extra.get("availability"),
        "service_area": pro.extra.get("service_area") or pro.city,
        "phone": pro.phone,
        # New, and the reason this module exists in its current form.
        "license_status": pro.license_status,
        "license_number": pro.license_number,
        "licensed": "yes" if pro.eligible else "no",
    }
    return row


def find_contractors(
    trade: str | None = None,
    max_hourly_rate: float | None = None,
    min_rating: float | None = None,
    limit: int = 5,
    home_id: str | None = None,
) -> list[dict]:
    """Licensed professionals near one home, best match first.

    Args:
        trade: a trade or a whole natural-language question. Both work — the
            question form is what the Advisor actually passes, and handling it
            correctly is the point of `trades.identify`.
        max_hourly_rate: only professionals at or below this hourly rate. Applies
            to the demo directory, which publishes rates; a state registry does
            not, so the filter is a no-op against live data rather than an empty
            result.
        min_rating: only professionals at or above this star rating, where a
            rating exists. A missing rating is NOT treated as a low one — it
            means the source does not publish ratings, and dropping a properly
            licensed contractor for that would be the wrong reading.
        limit: max results.
        home_id: which home's service area to search; defaults to the active home.

    Ordering comes from `pros.core.rank_key`: registered specialists first, then
    bond, then rating. Rating is no longer the primary key — a licence is a fact
    about a business and a star average is an opinion about it.
    """
    results = find_pros(trade, home_id=home_id, limit=max(limit * 3, 15))
    rows = [_to_legacy_row(p) for p in results.eligible]

    if max_hourly_rate is not None:
        rows = [r for r in rows
                if r["hourly_rate_usd"] is None or r["hourly_rate_usd"] <= max_hourly_rate]
    if min_rating is not None:
        rows = [r for r in rows if r["rating"] is None or r["rating"] >= min_rating]
    return rows[:limit]


def load_contractors(home_id: str | None = None) -> list[dict]:
    """Every eligible professional serving one home."""
    return [_to_legacy_row(p)
            for p in find_pros(None, home_id=home_id, limit=100).eligible]


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # The original bug, as a demonstration: this question must not return an
    # HVAC company.
    print("=== 'replace my lawn' (the \"ac\" -> hvac substring bug)")
    for row in find_contractors("replace my lawn", limit=3):
        print(f"   {row['name'][:34]:<34} {row['trade']}")

    for home in ("demo-002", "demo-001"):
        print(f"\n=== {home}: 'moss on the roof'")
        print(json.dumps(find_contractors("moss on the roof", limit=2, home_id=home),
                         indent=2))
