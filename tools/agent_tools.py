"""LangChain tool wrappers exposed to the LLM agent.

The agent never calls the raw functions directly — it calls these decorated
wrappers. The docstrings and type hints below are what the language model reads
to decide *when* and *how* to call each tool, so they are written for that
audience: short, explicit about arguments, and clear about what comes back.
"""
from __future__ import annotations

import re

from langchain_core.tools import tool

from memory.rag_store import search_policies as _search_policies
from memory.rerank import MIN_RERANK_SCORE
from tools.alerts import get_weather_alerts as _get_weather_alerts
from tools.elevation import get_elevation as _get_elevation
from tools.freeze_risk import assess_freeze_risk as _assess_freeze_risk
from tools.geocode import geocode_address as _geocode_address
from tools.hazard_check import run_hazard_check as _run_hazard_check
from tools.heat_risk import assess_heat_risk as _assess_heat_risk
from tools.homes import current_home_id, load_home
from tools.weather import get_weather_forecast as _get_weather_forecast

# Below this cosine similarity, retrieved passages are treated as not truly
# relevant, so the agent should refuse rather than fabricate a rule.
POLICY_RELEVANCE_THRESHOLD = 0.35


@tool
def get_home_profile() -> dict:
    """Return the saved profile for the home the user is currently working with
    (address, HOA, HVAC/filter info, exposed outdoor plumbing, appliances). Call
    this first when a question is about "my home" and no address was given.

    The user has more than one saved home. This returns the ACTIVE one — the home
    selected in the app — and lists the others under `other_homes`. If the user
    asks about a different saved home, tell them to switch homes in the app rather
    than answering from the wrong profile: HOA covenants, permit rules, and
    contractors all differ between them."""
    return load_home(current_home_id())


@tool
def geocode_address(address: str) -> dict:
    """Convert a street address or 'City, ST' into latitude/longitude coordinates.
    Returns {ok, latitude, longitude, matched_address, source, approximate?}. Continue
    the task with these coordinates and STATE the matched_address in your final answer;
    if `approximate` is true, note that the location is approximate (place-level). Do NOT
    stop to ask the user to confirm the address — finish the assessment first."""
    return _geocode_address(address)


@tool
def get_elevation(latitude: float, longitude: float) -> dict:
    """Return ground elevation (meters and feet) for coordinates. Higher elevation
    tends to run colder, which is useful context for freeze risk."""
    return _get_elevation(latitude, longitude)


@tool
def check_weather_hazards(location: str, horizon_hours: int = 48) -> dict:
    """PREFERRED tool for any weather-safety question. Give it a plain address or
    "City, ST" and it does the whole assessment in one step: geocodes the location,
    looks up elevation, fetches the forecast, checks official NWS advisories, and
    runs BOTH the freeze and heat assessors.

    Returns {ok, location:{resolved, elevation_ft, approximate}, forecast:{source,
    min_temp_f, max_temp_f, ...}, freeze:{level, headline, actions}, heat:{level,
    headline, actions}, alerts:{count, active:[...]}}.

    Report the `level` and `actions` from `freeze` and `heat` EXACTLY as returned —
    they come from the same deterministic assessors as before, so do not re-judge
    the numbers yourself. Name `forecast.source` when you cite the forecast, and
    state `location.resolved`. Use this instead of calling geocode_address,
    get_elevation, get_weather_forecast, get_weather_alerts, assess_freeze_risk and
    assess_heat_risk one at a time."""
    return _run_hazard_check(location, horizon_hours=horizon_hours)


# How many hourly points the granular tool hands back to the model. The full 48
# cost ~1400 tokens and were re-sent on every later turn; the model only ever used
# the min/max, so a coarse series is plenty for "what about tomorrow morning?".
_FORECAST_SAMPLE_EVERY = 6


@tool
def get_weather_forecast(latitude: float, longitude: float, horizon_hours: int = 48) -> dict:
    """Get the temperature forecast for COORDINATES over the next `horizon_hours`
    (default 48). Prefer `check_weather_hazards` unless you already have coordinates
    and only need raw numbers. Returns {ok, source, min_temp_f, min_temp_time,
    max_temp_f, max_temp_time, sampled_periods:[...]} where `sampled_periods` is the
    forecast thinned to roughly every 6 hours. `source` tells you whether the
    authoritative NWS feed or the Open-Meteo backup answered — mention it when you
    cite the forecast."""
    result = _get_weather_forecast(latitude, longitude, horizon_hours=horizon_hours)
    if not result.get("ok"):
        return result

    # Thin the hourly series before it reaches the model. The dashboard endpoints
    # call tools.weather.get_weather_forecast directly and still get all 48.
    periods = result.get("periods", [])
    slim = dict(result)
    slim.pop("periods", None)
    slim["sampled_periods"] = [
        {"start": p["start"], "temp_f": p.get("temp_f"),
         "humidity": p.get("humidity"), "wind_mph": p.get("wind_mph")}
        for p in periods[::_FORECAST_SAMPLE_EVERY]
    ]
    slim["periods_omitted"] = max(0, len(periods) - len(slim["sampled_periods"]))
    return slim


@tool
def assess_freeze_risk(
    min_temp_f: float,
    min_wind_mph: float | None = None,
    elevation_ft: float | None = None,
) -> dict:
    """Rate freeze/pipe risk from the forecast minimum temperature and return
    concrete protective actions. Pass `min_temp_f` from the forecast, and
    optionally the wind speed and elevation for better context. Returns
    {ok, level, headline, actions:[...], notes}. `level` is one of
    none|low|moderate|high|severe."""
    return _assess_freeze_risk(min_temp_f, min_wind_mph=min_wind_mph, elevation_ft=elevation_ft)


@tool
def assess_heat_risk(max_temp_f: float, humidity_pct: float | None = None) -> dict:
    """Rate heat risk from the forecast HIGH temperature and humidity (heat index),
    and return protective actions. Pass `max_temp_f` from the forecast (and
    `humidity_pct` = humidity_at_max if available). Returns {ok, level, heat_index_f,
    actions}. `level` is none|low|moderate|high|severe. Like freeze risk, let this
    tool decide the level — do not judge heat danger from the raw number yourself."""
    return _assess_heat_risk(max_temp_f, humidity_pct=humidity_pct)


@tool
def get_weather_alerts(latitude: float, longitude: float) -> dict:
    """Get ACTIVE official US National Weather Service advisories/watches/warnings for a
    location — heat, fire/Red Flag, severe thunderstorm, flood, high wind, winter storm,
    and more. Call this for any weather-safety question ("is it safe", "any advisories",
    heat/fire/storm/flood) so you never miss an official alert. Returns
    {ok, count, alerts:[{event, severity, headline, instruction}]}; count 0 means none
    active. US locations only."""
    return _get_weather_alerts(latitude, longitude)


@tool
def recall_memory(query: str) -> dict:
    """Search your own memory of PAST conversations with this user (including earlier
    sessions). Use this when the user refers to something previously discussed — "what did
    you tell me about...", "did I ask you about...", "last time", "you said" — or when you
    need to know what advice they already received. Returns
    {found, memories:[{when, age_days, user_query, answer}]}. If found is false, say you
    have no record of it rather than guessing. Note anything time-sensitive (like a weather
    forecast) may be stale — re-check it with a live tool."""
    from memory import episodic

    memories = episodic.recall(query, limit=4)
    return {
        "found": bool(memories),
        "memories": [
            {"when": m["when"], "age_days": m["age_days"],
             "user_query": m["user_query"], "answer": m["answer"][:400]}
            for m in memories
        ],
    }


def e_score(node: dict) -> float | None:
    """Node score for display: None when a hard gate vetoed it.

    A gated node scores -inf internally, which is not JSON-serialisable and is
    not a number a person should be shown. The UI renders these as "ruled out"
    with the reason rather than as a score of zero, which would read as "we
    considered it and it was bad" instead of "a rule forbade it".
    """
    total = node.get("total")
    if total is None or total == float("-inf"):
        return None
    return round(float(total), 1)


@tool
def ask_advisor(question: str) -> dict:
    """Delegate an open-ended DIY, maintenance, or home-improvement decision to the
    specialist Advisor agent. Use this for "how do I…", "what's the best way to…",
    "should I…", or install/repair/mount/hang/maintenance questions (e.g. hanging heavy
    decor, HVAC-filter timing). The Advisor uses Tree-of-Thought — it proposes several
    approaches, scores each against the home's constraints, and picks the best — grounded
    in the home profile, the home's rules, a contractor directory, and web search.
    Returns the full recommendation plus the options it compared. Present its
    recommendation to the user and keep its citations."""
    # Imported lazily to keep tool import light and avoid import cycles.
    from agents.advisor import run_advisor
    from agents.beam import depth_for_complexity

    result = run_advisor(question, persona=_current_persona() or "owner",
                         home_id=current_home_id(),
                         depth=depth_for_complexity(_current_routing().get("complexity")))
    return {
        "recommendation": result["final_answer"],
        "options_compared": [
            {"name": e["name"], "score": e["score"]} for e in result["evaluations"]
        ],
        # A compact view of the search itself, for the UI's reasoning panel. Kept
        # small on purpose: this rides in the tool result, which the stream
        # truncates, and the model does not need it — it already received the
        # chosen node. Pruned branches are included WITH their reason, because a
        # branch discarded invisibly is the one failure mode this design cannot
        # detect on its own.
        "reasoning_tree": [
            {"name": n["name"], "depth": n["depth"], "score": e_score(n),
             "status": n["status"], "reason": n["prune_reason"]}
            for n in result["tree"]
        ],
        "strategy": result["strategy"],
        "truncated": result["truncated"],
        "sources": result["sources"],
    }


@tool
def analyze_utility_costs(question: str, user_rate_cents_kwh: float | None = None) -> dict:
    """Delegate a utility-bill, energy-cost, or money-saving question to the specialist
    Cost agent. Use this for "how can I lower my utility bills", "what's my energy costing
    me", "how do I save money on electricity/gas". It combines LIVE state energy prices
    (EIA) with the home's size/climate to compute itemized annual savings. If the user
    tells you their actual rate from their bill, pass it as `user_rate_cents_kwh` — that is
    more accurate than a state average (especially in deregulated markets like Texas, where
    the address does not determine the retail rate). Present its answer and say whether the
    prices were live."""
    from agents.cost import run_cost_analysis

    result = run_cost_analysis(question, user_rate_cents_kwh=user_rate_cents_kwh,
                               home_id=current_home_id())
    return {
        "analysis": result["answer"],
        "home_label": result["home_label"],
        "rate_used_cents_kwh": result["rate_used_cents_kwh"],
        "rate_source": result["rate_source"],
        "prices_live": result["prices"]["live"],
        "total_annual_savings_usd": result["savings"]["total_annual_savings_usd"],
    }


def _current_routing() -> dict:
    """The Router's verdict for this turn, read from the run config.

    Empty outside a graph run — a direct call, the advisor's own `__main__`, or
    an eval case that pins its own depth. Every consumer must treat `{}` as "no
    opinion" and fall back to its own default, because that is not an error
    state: it is the system working exactly as it did before the Router existed.
    """
    try:
        from langgraph.config import get_config

        routing = (get_config().get("configurable") or {}).get("routing")
    except Exception:
        return {}
    return routing if isinstance(routing, dict) else {}


def _current_persona() -> str | None:
    """The persona for this turn, read from the run config rather than the model.

    Deliberately not a tool argument. A model that picks its own metadata filter
    can filter itself into an empty result set, and the failure is invisible — it
    looks exactly like "no source exists for this", which is the one answer this
    system must never give wrongly. The orchestrator knows the persona from the
    UI toggle, so it passes it down and the tool cannot get it wrong.
    """
    try:
        from langgraph.config import get_config

        persona = (get_config().get("configurable") or {}).get("persona")
    except Exception:
        return None  # outside a graph run (direct call, tests) — no filtering
    persona = (persona or "").strip().lower()
    return persona if persona in ("owner", "renter") else None


# A permission question asked by a renter has TWO halves, and the model's query
# only ever expresses one of them.
#
# "Can I replace the backyard grass with stones?" is a question about the change.
# The model searches for the change, the cross-encoder correctly ranks the
# landscaping covenant first, and the document that says whether a *tenant* may
# authorise anything scores -5.94 against that query — below the -4.0 floor, so
# it is dropped and the model never sees it.
#
# The reranker is not wrong. The tenant document is a poor answer to "how do I
# replace grass with stones"; it is the only answer to "may I decide that". The
# defect is that nobody asked the second question.
#
# Measured before this fix, on "replace backyard grass with stones rental
# landscaping changes":
#     KEPT     2.80   Lakeshore Commons HOA — CC&Rs
#     DROPPED -5.94   Renter / Tenant Policy Summary
# and on the anchor query below, the same document scores +2.84.
#
# So the occupant's own governing document is retrieved deterministically, with a
# fixed query, rather than hoping the model phrases its search in a way that
# happens to surface it. This is the same principle the audience filter and the
# home_id filter already follow: scoping decisions belong in code, because a model
# that picks its own scope will eventually pick the wrong one — and here the
# wrong one produced an answer that was *correct by luck*, sourced from the
# model's general knowledge of tenancy rather than from this home's lease terms.
# An ungrounded answer that happens to be right is the failure `grounded` exists
# to make visible.
_OCCUPANT_ANCHOR = {
    "renter": "tenant alterations require landlord written permission lease",
}

# Only for questions about doing something TO the property. A renter asking when
# the bins go out does not need their alteration rights appended.
_ALTERATION_QUERY = re.compile(
    r"\b(replace|install|remove|alter|modify|change|paint|mount|hang|build|"
    r"add|renovat\w*|attach|drill|landscap\w*|allowed|permitted|permission|"
    r"can i|may i|am i able)\b",
    re.IGNORECASE,
)


def _occupant_rules(persona: str | None, home_id: str | None, query: str,
                    already: set[str]) -> list[dict]:
    """The occupant's own governing document, when their authority is in question.

    Returns at most one passage, and only one that clears the same bar every
    other passage clears — this widens the *query*, never the threshold.
    """
    anchor = _OCCUPANT_ANCHOR.get((persona or "").lower())
    if not anchor or not _ALTERATION_QUERY.search(query or ""):
        return []
    try:
        found = _search_policies(anchor, k=3, audience=persona, home_id=home_id)
    except Exception:
        return []          # grounding is best-effort; never break the answer
    for p in found:
        score = p.get("rerank_score")
        if score is not None and score < MIN_RERANK_SCORE:
            continue
        if p["citation"] in already:
            continue
        return [p]
    return []


@tool
def search_home_policies(query: str) -> dict:
    """Search the home knowledge base (HOA covenants, city permit checklist, short-term
    rental/Airbnb rules, tenant rights, seasonal and freeze-prevention guides) for
    passages relevant to a question like "Am I allowed to ...?" or "Do I need a permit
    for ...?".

    Searches ONLY the documents that govern the home the user is currently working
    with, plus guides that apply to any home. Returns {query, grounded, jurisdiction,
    passages:[{citation, jurisdiction, text, score}]}. IMPORTANT: only answer policy
    questions using these passages and cite the `citation` of each one you use, and
    name the `jurisdiction` when you quote a city or state rule. If `grounded` is
    false, do NOT invent a rule — tell the user you don't have a source and suggest
    they verify with their actual HOA, city, or lease."""
    home_id = current_home_id()
    passages = _search_policies(query, k=4, audience=_current_persona(), home_id=home_id)

    # Prefer the cross-encoder's judgement when it is available: it scores the
    # query and passage together, so unlike the dense similarity it can tell
    # "mentions a backyard" apart from "answers a question about backyards".
    # Falls back to the dense threshold whenever the reranker is unavailable, so
    # behaviour degrades to what it was before rather than failing open.
    reranked = [p for p in passages if p.get("rerank_score") is not None]
    if reranked:
        relevant = [p for p in reranked if p["rerank_score"] >= MIN_RERANK_SCORE]
        grounded = bool(relevant)
        top_score = reranked[0]["rerank_score"]
        scorer = "cross-encoder"
    else:
        relevant = [p for p in passages if (p.get("score") or 0.0) >= POLICY_RELEVANCE_THRESHOLD]
        grounded = bool(relevant)
        top_score = passages[0].get("score") if passages else 0.0
        scorer = "embedding"

    # The occupant's own rules, retrieved on their own terms. Appended rather than
    # ranked in: it answers a different question from the one the model asked, so
    # ordering it against those results would be comparing two different queries'
    # scores. It clears the same threshold they did.
    persona = _current_persona()
    extra = _occupant_rules(persona, home_id, query, {p["citation"] for p in relevant})
    if extra:
        relevant = list(relevant) + extra
        grounded = True

    # Passages below the bar are dropped rather than shipped with a low score.
    # Handing the model text it must then decide to ignore is how invented rules
    # happen; returning nothing is an unambiguous "no source".
    return {
        "query": query,
        "grounded": grounded,
        "home_id": home_id,
        "top_score": top_score,
        "scored_by": scorer,
        "passages": [
            {"citation": p["citation"], "jurisdiction": p.get("jurisdiction"),
             "text": p["text"],
             "score": p.get("rerank_score") if reranked else p.get("score")}
            for p in relevant
        ],
    }


@tool
def find_licensed_pros(need: str) -> dict:
    """Find licensed professionals for a home task, checked against the contractor
    registry. Use this whenever the user asks who to hire, who to call, or whether
    someone is licensed — and always for work that must not be DIY (gas, service
    panel, structural, roof). Pass the user's own words as `need` (e.g. "my water
    heater is leaking"); it identifies the trade itself.

    Returns `recommended` plus `withheld` — businesses that matched but hold no
    current registration, each with the reason. **Never present a withheld business
    as an option.** They are returned so you can tell the user honestly that
    matches exist but are not currently registered; that is useful information, not
    a shortlist. Repeat the licence status and the registry name in your answer."""
    from tools.pros.core import find_pros

    results = find_pros(need, home_id=current_home_id(), limit=4)
    return {
        "trade": results.trade_label or "general",
        "source": results.source,
        "area": results.jurisdiction,
        "recommended": [
            {"name": p.name, "match": p.match, "license_status": p.license_status,
             "license_number": p.license_number, "license_expires": p.license_expires,
             "license_type": p.license_type, "specialty": (p.specialty or "").strip(),
             "phone": p.phone, "rating": p.rating, "reviews": p.review_count,
             "bond_usd": p.bond_amount,
             "hourly_rate_usd": p.extra.get("hourly_rate_usd"),
             "availability": p.extra.get("availability")}
            for p in results.eligible
        ],
        "withheld": [
            {"name": p.name, "reason": p.withheld_reason, "rating": p.rating}
            for p in results.withheld
        ],
        "notes": results.notes,
    }


# The ordered list the agent is given. Two things matter about this order:
#   * get_home_profile stays first, so the model is nudged to establish location
#     before reaching for weather tools;
#   * check_weather_hazards sits directly after it, because it is the intended
#     one-shot path for weather questions and the granular tools below it are the
#     fallback for when the model needs a single number.
# The order must also stay STABLE: tool schemas render at the very front of the
# prompt, so reordering this list invalidates the prompt cache (see agents/llm.py).
AGENT_TOOLS = [
    get_home_profile,
    check_weather_hazards,
    geocode_address,
    get_elevation,
    get_weather_forecast,
    assess_freeze_risk,
    assess_heat_risk,
    get_weather_alerts,
    search_home_policies,
    ask_advisor,
    analyze_utility_costs,
    recall_memory,
    # Appended, never inserted — see the note above about prompt caching.
    find_licensed_pros,
]
