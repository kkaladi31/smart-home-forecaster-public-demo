"""The licence gate, the ranking, and the shapes both providers return.

Kept separate from the providers so the gate is one piece of code rather than one
per source. A second provider that forgot to apply it would be a silent hole in
the only safety claim this package makes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import config
import telemetry
from tools.pros import trades as trade_lib

# ---------------------------------------------------------------------------
# The gate
#
# An ALLOWLIST, and that is load-bearing. The live status vocabulary has ten
# values, measured 2026-08-16 across 161,093 rows:
#
#   ACTIVE 75,477 · EXPIRED 61,430 · SUSPENDED 9,808 · RE-LICENSED 9,415
#   OUT OF BUSINESS 4,714 · INACTIVE 133 · SUPERCEDED 94 · PASSED AWAY 15
#   RESTORED FROM ARCHIVED 6 · REVOKED DUE DEPT ERR 1
#
# A denylist of the bad ones fails open the moment L&I adds an eleventh, and the
# failure would be invisible: an unrecognised status would sail through as
# recommendable. RE-LICENSED is excluded deliberately — it plausibly means a
# lapsed registration since renewed, but "plausibly" is not the standard for
# telling someone a contractor is licensed.
# ---------------------------------------------------------------------------
ELIGIBLE_STATUSES = frozenset({"ACTIVE"})

# Status is cross-checked against the expiry date. Measured 2026-08-16, the live
# data is internally consistent — zero ACTIVE rows expired before this month —
# so this is defence in depth against a future publishing lag, not a workaround
# for an observed defect. Saying otherwise in a comment would be inventing
# evidence for a decision that stands on its own.
#
# THE DATE MUST BE THE REGISTRY'S LOCAL DATE, NOT THE SERVER'S. This was very
# nearly a real bug, and it surfaced only because a spike using a UTC date
# disagreed with a check using a local one: 28 contractors whose registration
# runs through today read as expired under UTC, because Washington publishes
# local dates and UTC was already tomorrow. On a container — which is where this
# is headed — `date.today()` IS UTC, so the product would have told 28 named,
# properly registered businesses' prospective customers that their registration
# had lapsed. A false factual claim about a real business is a worse error than
# either direction of a borderline call, so the provider supplies the date in
# its own jurisdiction rather than the process happening to be in the right zone.
CHECK_EXPIRY = True


@dataclass
class Pro:
    """One professional, as this system is willing to describe them."""
    name: str
    license_number: str | None = None
    license_status: str | None = None
    license_expires: str | None = None
    license_type: str | None = None
    specialty: str | None = None          # raw, as published — for display/citation
    city: str | None = None
    zip_code: str | None = None
    phone: str | None = None
    ubi: str | None = None
    principal: str | None = None
    bond_amount: float | None = None
    insurance_amount: float | None = None
    rating: float | None = None
    review_count: int | None = None
    # Filled by the gate/ranker rather than the provider.
    match: str = trade_lib.UNKNOWN
    eligible: bool = False
    withheld_reason: str | None = None
    source: str = "unknown"
    # Provider-specific fields with no equivalent in the other provider. The
    # demo directory carries hourly rates and availability, which a state
    # registry does not publish and never will; hoisting those into the dataclass
    # would put demo-only concepts into the shape the live registry fills.
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ProResults:
    """Eligible pros, plus what was withheld and why.

    Withheld entries are returned rather than dropped on purpose. "No licensed
    roofers found" and "four roofers found, all with expired registrations" are
    very different situations for a user, and a list that silently omits the
    second is hiding the most useful thing it learned.
    """
    trade: str | None = None
    trade_label: str | None = None
    eligible: list[Pro] = field(default_factory=list)
    withheld: list[Pro] = field(default_factory=list)
    source: str = "unknown"
    jurisdiction: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "trade": self.trade,
            "trade_label": self.trade_label,
            "eligible": [p.as_dict() for p in self.eligible],
            "withheld": [p.as_dict() for p in self.withheld],
            "source": self.source,
            "jurisdiction": self.jurisdiction,
            "notes": list(self.notes),
        }


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "")).date()
    except ValueError:
        return None


def apply_gate(pro: Pro, *, today: date | None = None) -> Pro:
    """Decide eligibility and, when withheld, say why in the user's language.

    Mutates and returns the same object. The reason is written to be shown, not
    logged: "registration expired on 2024-11-02" is something a user can act on,
    where "status != ACTIVE" is not.
    """
    today = today or date.today()
    status = (pro.license_status or "").strip().upper()

    if not status:
        pro.eligible = False
        pro.withheld_reason = "no registration record could be found for this business"
        return pro

    if status not in ELIGIBLE_STATUSES:
        readable = {
            "EXPIRED": "their contractor registration has expired",
            "SUSPENDED": "their contractor registration is suspended",
            "OUT OF BUSINESS": "they are recorded as out of business",
            "INACTIVE": "their contractor registration is inactive",
            "REVOKED DUE DEPT ERR": "their contractor registration was revoked",
            "PASSED AWAY": "the registration holder is recorded as deceased",
            "RE-LICENSED": ("their registration is recorded as re-licensed, which "
                            "this system does not treat as confirmation that it is "
                            "currently valid"),
        }.get(status, f"their registration status is {status.title()}")
        pro.eligible = False
        pro.withheld_reason = readable
        return pro

    expires = _parse_date(pro.license_expires)
    # A licence expiring on date D is valid THROUGH D, so the comparison is
    # strictly-before rather than on-or-before. `today` is the registry's local
    # date — see CHECK_EXPIRY.
    if CHECK_EXPIRY and expires and expires < today:
        # The record says ACTIVE and the date says otherwise. Trust the stricter
        # of the two: a stale status field is a data-freshness artefact, and
        # resolving it in the contractor's favour is resolving it against the user.
        pro.eligible = False
        pro.withheld_reason = (
            f"their registration is recorded as active but expired on {expires.isoformat()}")
        return pro

    pro.eligible = True
    pro.withheld_reason = None
    return pro


# Ranking among ELIGIBLE pros only. Nothing here can promote an ineligible one,
# because ineligible ones are not in this list at all.
_MATCH_RANK = {trade_lib.SPECIALIST: 0, trade_lib.GENERAL: 1,
               trade_lib.UNASSESSED: 2, trade_lib.UNKNOWN: 3}


def rank_key(pro: Pro) -> tuple:
    """Deterministic ordering: specialisation, then bond, then rating, then name.

    Specialisation first because it is the thing the licence data actually
    establishes. Rating is third and not first — it comes from a different source
    with a different failure mode (see the join rules in `places.py`), and a
    directory rating is an opinion where a registration is a fact. Name last so
    two otherwise identical firms order reproducibly rather than by dict order.
    """
    return (
        _MATCH_RANK.get(pro.match, 3),
        -(pro.bond_amount or 0.0),
        -(pro.rating or 0.0),
        -(pro.review_count or 0),
        pro.name.lower(),
    )


def find_pros(
    query: str | None = None,
    *,
    home_id: str | None = None,
    limit: int = 5,
    trade_key: str | None = None,
) -> ProResults:
    """Find licensed professionals for a request, gated on registration status.

    `query` is natural language — "my water heater is leaking" — and is mapped to
    a trade by `trades.identify`, which returns every plausible trade rather than
    guessing one.

    The provider is chosen by build profile, not by a runtime flag: a demo build
    has no real contractor data on disk and must not acquire any over the network.
    """
    matched = trade_lib.identify(query)
    if trade_key:
        picked = trade_lib.TRADES_BY_KEY.get(trade_key)
        matched = [picked] if picked else matched
    trade = matched[0] if matched else None

    # Imported INSIDE the branch, not above it.
    #
    # This used to be `from tools.pros import fixtures, lni_wa` followed by a
    # ternary, so a demo build loaded the real Washington L&I client and then
    # declined to use it. That inverts the project's own rule: the filesystem is
    # the primary control and the runtime gate is the backup, not the reverse. A
    # real provider that is merely unused is one refactor away from being used.
    #
    # It also had a consequence outside the process. `lni_wa.py` had to ship in
    # the public demo build purely to satisfy this import, and shipping it told
    # any reader that the full build queries a *Washington* registry — while the
    # demo's homes are Dallas and Minneapolis. The synthetic-data rule protects
    # the address; it should not be undone by an import naming the state.
    #
    # Now a demo build never imports it, and the public build does not carry it.
    if config.is_demo_build():
        from tools.pros import fixtures as provider
    else:
        from tools.pros import lni_wa as provider
    # Each provider supplies the date its own registry keeps, so the gate never
    # compares a Washington expiry against a UTC clock. See CHECK_EXPIRY.
    today = provider.registry_today()

    with telemetry.span("tool", "pros.find", "Finding licensed professionals") as s:
        results = provider.search(trade, home_id=home_id, limit=max(limit * 4, 20))
        results.trade = trade.key if trade else None
        results.trade_label = trade.label if trade else None

        eligible, withheld = [], []
        for pro in results.eligible + results.withheld:
            apply_gate(pro, today=today)
            # Only claim a match verdict when a trade was actually requested.
            # Leaving the UNKNOWN default in place made a browse-everything
            # listing render "registration does not name this trade" against
            # every properly registered plumber in it.
            pro.match = (trade_lib.match_quality(trade, pro.license_type, pro.specialty)
                         if trade else trade_lib.UNASSESSED)
            (eligible if pro.eligible else withheld).append(pro)

        eligible.sort(key=rank_key)
        withheld.sort(key=lambda p: p.name.lower())
        results.eligible = eligible[:limit]
        results.withheld = withheld[:limit]

        if trade and trade.restricted:
            # Phrased to need no indefinite article: "a Electrical Contractor"
            # is what the obvious wording produces, and a grammar slip in a
            # safety note reads as carelessness about the note itself.
            results.notes.append(
                f"{trade.label} work in this state is restricted to holders of "
                f"{trade.license_types[0].title()} registration; general "
                "contractors are not listed for it.")
        if len(matched) > 1:
            results.notes.append(
                "This request mentions more than one trade: "
                + ", ".join(t.label for t in matched[:3])
                + f". Showing {trade.label}.")
        if not results.eligible and results.withheld:
            results.notes.append(
                f"{len(results.withheld)} nearby business(es) matched but none hold a "
                "current registration, so none are recommended.")

        s.update({"trade": results.trade, "eligible": len(results.eligible),
                  "withheld": len(results.withheld), "source": results.source})
    return results
