"""The demo profile's professional directory — 100% invented, no network.

Reads `<data_root>/contractors.csv`, which under the demo profile contains
fictional businesses at fictional phone numbers with fictional registration
numbers. Nothing here touches a real registry, and there is no code path that
could: the file is the whole source.

The fixture deliberately carries the **same registration vocabulary as the live
WA L&I data** — real status values, both licence axes, the trailing-space HVAC
spelling — so the demo exercises the identical gate rather than a simplified one.
A demo that showed a gate the full build does not run would be a demo of nothing.

Five rows exist specifically to be refused, because a gate with nothing to stop
is untestable:

    c013 Skyline Roofworks         SUSPENDED
    c115 Frostline Roofing         EXPIRED
    c117 Boulder Ridge Exteriors   OUT OF BUSINESS
    c005 Maple Grove Handyworks    EXPIRED
    c116 Kettle River Plumbing     ACTIVE, but expired 2026-06-30

The last is the interesting one, and it mirrors 28 real rows in the live dataset
that report ACTIVE alongside a past expiry date. It is also the highest-rated
plumber for the Minneapolis home — 4.9 stars, 388 reviews — so the code this
replaced would have ranked it **first**. That single row is the difference
between sorting by rating and gating on registration.
"""
from __future__ import annotations

import csv
from datetime import date

import config
from tools.homes import current_home_id, resolve_home_id
from tools.pros import trades as trade_lib
from tools.pros.core import Pro, ProResults

# The provenance is in the PARENTHETICAL, deliberately, so `splitQualifier` in
# the UI moves it to a hover title instead of printing it on screen. The demo
# must be honest about its data without shouting "synthetic" at every reader.
SOURCE = "State contractor registry (synthetic)"


def registry_today() -> date:
    """The date this registry keeps.

    The invented registry has no timezone of its own, so the server's date is the
    only sensible answer. The function exists because `core.find_pros` asks every
    provider for one — which is what keeps the live provider from silently
    falling back to a UTC container clock.
    """
    return date.today()


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _rows(home_id: str | None) -> list[dict]:
    wanted = resolve_home_id(home_id) if home_id else current_home_id()
    path = config.data_root() / "contractors.csv"
    with open(path, encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("home_id") == wanted]


def _to_pro(row: dict) -> Pro:
    return Pro(
        name=row.get("name", "").strip(),
        license_number=row.get("license_number") or None,
        license_status=row.get("license_status") or None,
        license_expires=row.get("license_expires") or None,
        license_type=row.get("license_type") or None,
        # Raw, including the trailing space the live registry publishes on the
        # long HVAC value — normalising here would hide the exact data-quality
        # problem the fixture is meant to reproduce.
        specialty=row.get("specialty"),
        city=(row.get("service_area") or "").split("/")[0].strip() or None,
        phone=row.get("phone") or None,
        ubi=row.get("ubi") or None,
        bond_amount=_float(row.get("bond_amount")),
        rating=_float(row.get("rating")),
        review_count=_int(row.get("reviews")),
        source=SOURCE,
        extra={
            "contractor_id": row.get("contractor_id"),
            "trade": row.get("trade"),
            "hourly_rate_usd": _float(row.get("hourly_rate_usd")),
            "min_job_usd": _float(row.get("min_job_usd")),
            "availability": row.get("availability"),
            "service_area": row.get("service_area"),
        },
    )


def search(trade: trade_lib.Trade | None, *, home_id: str | None = None,
           limit: int = 20) -> ProResults:
    """Every fixture row for the home that plausibly serves this trade.

    Filtering mirrors the live provider: match on whichever axis the trade uses,
    and fall back to general contractors only for unrestricted work. The gate has
    not run at this point — `core.find_pros` applies it — so ineligible rows are
    returned here on purpose.
    """
    pros = [_to_pro(r) for r in _rows(home_id)]
    notes: list[str] = []

    if trade:
        matched = [p for p in pros
                   if trade_lib.match_quality(trade, p.license_type, p.specialty)
                   == trade_lib.SPECIALIST]
        if not matched and not trade.restricted:
            matched = [p for p in pros
                       if trade_lib.match_quality(trade, p.license_type, p.specialty)
                       == trade_lib.GENERAL]
            if matched:
                notes.append(
                    f"No contractor in this directory is registered specifically for "
                    f"{trade.label.lower()}, so general contractors are shown instead.")
        pros = matched

    jurisdiction = None
    rows = _rows(home_id)
    if rows:
        jurisdiction = rows[0].get("service_area")

    return ProResults(eligible=pros[:limit], source=SOURCE,
                      jurisdiction=jurisdiction, notes=notes)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from tools.pros.core import find_pros

    for home, question in (("demo-002", "there is moss all over the roof"),
                           ("demo-002", "my water heater is leaking"),
                           ("demo-002", "the breaker keeps tripping"),
                           ("demo-001", "I need someone to hang a heavy mirror"),
                           ("demo-002", "my driveway is cracked")):
        res = find_pros(question, home_id=home, limit=4)
        print(f"\n=== [{home}] {question}   trade={res.trade}")
        for pro in res.eligible:
            print(f"   RECOMMEND  {pro.name[:34]:<34} {pro.match:<10} "
                  f"{pro.license_status:<8} exp {pro.license_expires}  "
                  f"{pro.rating}*")
        for pro in res.withheld:
            print(f"   WITHHELD   {pro.name[:34]:<34} {pro.rating}*  "
                  f"-> {pro.withheld_reason}")
        for note in res.notes:
            print(f"   note: {note}")
