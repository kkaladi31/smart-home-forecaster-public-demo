"""Mapping a homeowner's words onto Washington L&I's two ways of naming a trade.

L&I does not have one trade column, and assuming it does produces a confident
wrong answer. Measured against the live dataset (161,093 rows, 2026-08-16):

**Axis 1 — regulated trades live in `contractorlicensetypecodedesc`.**
    CONSTRUCTION CONTRACTOR  148,739
    ELECTRICAL CONTRACTOR      9,203
    PLUMBING CONTRACTOR        3,029
    ELEVATOR CONTRACTOR          122
Plumbing and electrical are separate licence *types*, not specialties.

**Axis 2 — everything else lives in `specialtycode1desc`, on CONSTRUCTION
CONTRACTOR rows.** ROOFING, LANDSCAPING, HANDYMAN, FENCING, Tree Removal Service
and so on.

Why this matters more than a schema detail: the obvious query — filter
`specialtycode1desc LIKE '%PLUMB%'` — returns **zero active firms**, because that
specialty value is a retired taxonomy that now appears only on EXPIRED (662),
RE-LICENSED (537), OUT OF BUSINESS (41) and INACTIVE (1) records. A product built
on it would tell a user there are no licensed plumbers in their city while 182
sit in the same five cities under the licence-type axis. That is the worst class
of failure this system can produce: authoritative, specific, and wrong.

Licence type is also *better* than matching the business name, not merely
different. `A B CONTRACTING AND DEV LLC` is a licensed plumbing contractor whose
name never says plumbing, and a name search misses it entirely.

Two data-quality facts the taxonomy has to absorb:

* **Specialty values are dirty.** HVAC exists three ways among ACTIVE rows alone:
  `'Heating/Vent/Air-Conditioning and Refrig (HVAC/R) '` (759, with a trailing
  space), `'HVAC/RFRG'` (446) and `'HVAC/RFRG-RESTRICTED'` (15). One trade maps
  to several raw strings, and every comparison is normalised.
* **GENERAL dominates.** 3,651 of 5,107 active construction contractors in a
  five-city sample are specialty GENERAL. Excluding them would empty most trade
  searches in a small city; treating them as equal to specialists would make the
  trade filter meaningless, since 71% of results would be generals. They are
  therefore kept and **ranked below specialists with the difference stated** —
  see `SPECIALIST` on the match result.

  (The sample's cities are named nowhere in this file on purpose. This module
  ships in the public demo build, where the rule is that no real location
  appears; a county name is not covered by the leak scanner but would still
  narrow where the private build's home is.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from tools.phrases import compile_phrases, find_terms

# Match quality, in rank order. A caller must be able to tell a registered
# roofer from a general contractor who is allowed to roof, because the user
# cannot.
SPECIALIST = "specialist"     # the registration names this trade
GENERAL = "general"           # a general contractor, legally able but not specialised
UNKNOWN = "unknown"           # asked, and the registration does not cover it
# Distinct from UNKNOWN, and the distinction is not pedantic: it separates "we
# asked and the answer was no" from "no trade was requested, so there was
# nothing to ask". Collapsing them makes a browse-everything listing label every
# properly registered plumber as not registered for plumbing.
UNASSESSED = "unassessed"


def normalise_specialty(value: str | None) -> str:
    """Upper-cased, whitespace-collapsed specialty, for comparison only.

    Never use this for display: `'Heating/Vent/Air-Conditioning and Refrig
    (HVAC/R) '` is what L&I publishes and what a citation should quote.
    """
    return re.sub(r"\s+", " ", (value or "")).strip().upper()


GENERAL_SPECIALTY = normalise_specialty("GENERAL")


@dataclass(frozen=True)
class Trade:
    """One trade a user might ask for, and how L&I happens to name it.

    `license_types` and `specialties` are the two axes. A trade uses one or the
    other, never both — plumbing is a licence type and roofing is a specialty,
    and pretending the schema is uniform is what the module docstring is about.
    """
    key: str
    label: str
    terms: tuple[str, ...]
    license_types: tuple[str, ...] = ()
    specialties: tuple[str, ...] = ()
    # True when the work is legally restricted to that licence type, so a general
    # contractor is NOT an acceptable substitute. Gas, electrical and plumbing
    # are the cases where "close enough" is a safety claim we must not make.
    restricted: bool = False


TRADES: tuple[Trade, ...] = (
    # --- Axis 1: regulated licence types ------------------------------------
    Trade(
        key="plumbing", label="Plumbing",
        terms=("plumber", "plumbing", "pipe", "pipes", "drain", "drains", "faucet",
               "toilet", "water heater", "sewer", "leak", "burst pipe", "sump pump",
               "water line", "repipe",
               # Gas fitting. The ordering comment further down has always claimed
               # that "a question about a gas line reaches plumbing before it
               # reaches handyman" — and it could not, because there was no gas
               # term anywhere in this file. `identify("how do I run a gas line
               # myself")` returned an EMPTY list, the caller fell through to an
               # untargeted browse of the directory, and the answer named an
               # electrician. That lands on the one class of question where the
               # high-risk short-circuit makes the referral list the WHOLE answer.
               #
               # Phrases only, never a bare "gas": "lower my gas bill" is a cost
               # question, and routing it to a plumber would swap one wrong trade
               # for another. "natural gas" is left out for the same reason —
               # "run a natural gas line" already matches on "gas line".
               "gas line", "gas lines", "gas pipe", "gas pipes", "gas piping",
               "gas fitting", "gas fitter", "gas valve", "gas meter",
               "gas appliance", "gas connection", "propane line", "propane tank"),
        license_types=("PLUMBING CONTRACTOR",),
        restricted=True,
    ),
    Trade(
        key="electrical", label="Electrical",
        terms=("electrician", "electrical", "wiring", "rewire", "outlet", "outlets",
               "breaker", "breaker box", "service panel", "circuit", "gfci",
               "light fixture", "ev charger"),
        license_types=("ELECTRICAL CONTRACTOR",),
        restricted=True,
    ),
    Trade(
        key="elevator", label="Elevator",
        terms=("elevator", "lift", "stair lift", "chair lift"),
        license_types=("ELEVATOR CONTRACTOR",),
        restricted=True,
    ),
    # --- Axis 2: construction-contractor specialties -------------------------
    # Raw values are listed exactly as L&I publishes them, including the trailing
    # space on the long HVAC string, and normalised on comparison. Listing the
    # real strings rather than a prefix keeps this auditable against the live
    # vocabulary that scripts/contract_check.py prints.
    Trade(
        key="hvac", label="Heating and cooling",
        terms=("hvac", "furnace", "air conditioning", "air conditioner",
               "heat pump", "heating", "cooling", "ductwork", "ducts",
               "thermostat", "boiler"),
        specialties=("Heating/Vent/Air-Conditioning and Refrig (HVAC/R) ",
                     "HVAC/RFRG", "HVAC/RFRG-RESTRICTED"),
    ),
    Trade(
        key="roofing", label="Roofing",
        terms=("roof", "roofer", "roofing", "shingle", "shingles", "moss",
               "flashing", "soffit", "ice dam"),
        specialties=("ROOFING",),
    ),
    Trade(
        key="landscaping", label="Landscaping",
        terms=("landscaper", "landscaping", "lawn", "yard", "grass", "sod",
               "irrigation", "sprinkler", "garden"),
        specialties=("LANDSCAPING",),
    ),
    Trade(
        key="tree_service", label="Tree service",
        terms=("tree", "trees", "branch", "branches", "limb", "limbs", "stump",
               "arborist", "prune", "pruning"),
        specialties=("Tree Removal Service",),
    ),
    Trade(
        key="fencing", label="Fencing",
        terms=("fence", "fencing", "gate post", "railing"),
        specialties=("FENCING",),
    ),
    Trade(
        key="painting", label="Painting",
        terms=("paint", "painting", "painter", "repaint", "wallpaper", "stain"),
        specialties=("PAINTING/WALLCOVERING",),
    ),
    Trade(
        key="siding", label="Siding",
        terms=("siding", "clapboard", "cladding", "exterior boards"),
        specialties=("Siding",),
    ),
    Trade(
        key="drywall", label="Drywall",
        terms=("drywall", "sheetrock", "plaster", "wall repair"),
        specialties=("DRY WALL",),
    ),
    Trade(
        key="concrete", label="Concrete",
        terms=("concrete", "driveway", "slab", "foundation crack", "patio slab"),
        specialties=("CONCRETE",),
    ),
    Trade(
        key="flooring", label="Flooring",
        terms=("flooring", "floors", "hardwood", "laminate", "carpet",
               "countertop", "countertops"),
        specialties=("Floor Covering and Counter Tops",),
    ),
    Trade(
        key="handyman", label="Handyman",
        terms=("handyman", "odd jobs", "small repairs", "hang", "mount",
               "mounting", "install a shelf", "picture rail", "general repairs"),
        specialties=("HANDYMAN",),
    ),
)

TRADES_BY_KEY = {t.key: t for t in TRADES}

# Compiled once. Order follows TRADES, so `restricted` trades are tested before
# the general ones and a question about a gas line reaches plumbing before it
# reaches handyman.
_TERM_PATTERNS = {t.key: compile_phrases(t.terms) for t in TRADES}

# Normalised lookup tables, built from the declarations so the two can never
# disagree about what a trade's raw values are.
_SPECIALTIES = {t.key: {normalise_specialty(s) for s in t.specialties} for t in TRADES}
_LICENSE_TYPES = {t.key: {normalise_specialty(lt) for lt in t.license_types} for t in TRADES}


def identify(text: str | None) -> list[Trade]:
    """Every trade the text plausibly asks for, most specific first.

    Returns a *list*, not a best guess. "My water heater is leaking and the
    breaker keeps tripping" is two trades, and collapsing it to one is how the
    old alias table produced a single wrong answer. Ordering is by number of
    matched terms, then by declaration order, so ties are reproducible.

    Restricted trades are never displaced by an unrestricted one on a tie: a
    question that mentions both a breaker and a shelf is an electrical question
    with a shelf in it.
    """
    if not text:
        return []
    scored = []
    for trade in TRADES:
        hits = find_terms(text, _TERM_PATTERNS[trade.key])
        if hits:
            scored.append((len(hits), trade.restricted, -TRADES.index(trade), trade))
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return [row[3] for row in scored]


def match_quality(trade: Trade, license_type: str | None, specialty: str | None) -> str:
    """How well one L&I record answers a request for `trade`.

    SPECIALIST when the registration names the trade on whichever axis the trade
    uses; GENERAL when the firm is a general construction contractor, which is
    legal for unrestricted work and is NOT accepted for a restricted trade;
    UNKNOWN otherwise.
    """
    lt = normalise_specialty(license_type)
    spec = normalise_specialty(specialty)

    if _LICENSE_TYPES[trade.key] and lt in _LICENSE_TYPES[trade.key]:
        return SPECIALIST
    if _SPECIALTIES[trade.key] and spec in _SPECIALTIES[trade.key]:
        return SPECIALIST
    # A general construction contractor may do unrestricted work. For plumbing,
    # electrical and elevator work they may not, and calling them a weak match
    # would still put them in front of the user as an option.
    if not trade.restricted and spec == GENERAL_SPECIALTY:
        return GENERAL
    return UNKNOWN


def soql_filter(trade: Trade) -> str:
    """The SoQL fragment selecting this trade, on whichever axis it uses.

    Built here rather than in the provider so the axis decision lives beside the
    evidence for it. Values are drawn from the declarations and contain no user
    input, so there is nothing to escape beyond the apostrophe doubling below —
    which is present because a specialty string one day containing one should
    break a query, not change it.
    """
    def lit(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    if trade.license_types:
        joined = ", ".join(lit(v) for v in trade.license_types)
        return f"contractorlicensetypecodedesc IN ({joined})"
    # Specialty comparison is normalised on the server the same way
    # normalise_specialty does locally: upper-cased and trimmed. Internal
    # whitespace is not collapsed server-side, so the raw values are listed
    # verbatim and TRIM handles the trailing space that L&I actually publishes.
    joined = ", ".join(lit(normalise_specialty(v)) for v in trade.specialties)
    return (f"contractorlicensetypecodedesc = 'CONSTRUCTION CONTRACTOR' "
            f"AND upper(trim(specialtycode1desc)) IN ({joined})")


def general_filter() -> str:
    """Selects general construction contractors, the fallback tier."""
    return ("contractorlicensetypecodedesc = 'CONSTRUCTION CONTRACTOR' "
            f"AND upper(trim(specialtycode1desc)) = '{GENERAL_SPECIALTY}'")


if __name__ == "__main__":
    from tools.phrases import leaks_across_boundaries

    print("Trade identification\n" + "=" * 68)
    for question in (
        "my water heater is leaking",
        "the breaker keeps tripping when I run the dryer",
        "there is moss all over the roof",
        "replace my lawn",                      # the original substring bug
        "I want to resurface the driveway",     # 'surface' must not match 'ac'
        "how do I hang a heavy mirror",
        "my furnace is making a noise and the pipes rattle",
        "what is the weather tomorrow",
    ):
        found = identify(question)
        print(f"  {question[:44]:<44} -> "
              f"{[t.key for t in found] or '(no trade)'}")

    print("\nBoundary safety over the whole table")
    leaks = {t.key: leaks_across_boundaries(t.terms) for t in TRADES}
    leaks = {k: v for k, v in leaks.items() if v}
    print(f"  leaks: {leaks or 'none'}")

    print("\nSoQL fragments")
    for key in ("plumbing", "roofing", "hvac"):
        print(f"  {key:<10} {soql_filter(TRADES_BY_KEY[key])}")

    print("\nMatch quality")
    plumbing, roofing = TRADES_BY_KEY["plumbing"], TRADES_BY_KEY["roofing"]
    rows = [
        (plumbing, "PLUMBING CONTRACTOR", "JOURNEY LEVEL", SPECIALIST),
        (plumbing, "CONSTRUCTION CONTRACTOR", "GENERAL", UNKNOWN),   # restricted
        (roofing, "CONSTRUCTION CONTRACTOR", "ROOFING", SPECIALIST),
        (roofing, "CONSTRUCTION CONTRACTOR", "GENERAL", GENERAL),
        (roofing, "CONSTRUCTION CONTRACTOR", "PAINTING/WALLCOVERING", UNKNOWN),
    ]
    ok = True
    for trade, lt, spec, want in rows:
        got = match_quality(trade, lt, spec)
        flag = "ok  " if got == want else "FAIL"
        ok &= got == want
        print(f"  {flag} {trade.key:<10} {lt:<24} {spec[:22]:<22} -> {got}")
    raise SystemExit(0 if ok and not leaks else 1)
