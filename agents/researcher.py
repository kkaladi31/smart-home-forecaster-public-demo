"""The Researcher: turns a question into a ranked, cited, screened evidence pack.

Replaces the single raw DuckDuckGo call the Advisor used to make. That call gave
the model **four results truncated to 160 characters each** — 640 characters of
search-blurb, from which it was expected to reason about load ratings and
building practice.

    plan  ->  fan out  ->  merge + dedupe  ->  screen  ->  rank  ->  pack

Query planning is **templates, not a model**. Three variations cover what this
product actually asks — the question as posed, the question grounded in the
home's construction, and the question aimed at primary sources — and a fourth
LLM call per turn is real latency on a free model for a gain nobody could
measure.

An `SHF_RESEARCH_LLM_PLAN` escape hatch used to be *checked* here without ever
being *read*, on the theory that a flag keeps a decision visible in code. It does
not. A function with no call sites is dead code that reads as live, and the next
person to find it reasonably concludes LLM planning exists behind the flag. The
decision belongs in this paragraph, where it cannot quietly stop being true.

Everything here is best-effort. A failed search costs the caller its citations,
never its answer.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import telemetry
from tools.research import evidence, providers

MAX_QUERIES = 4
RESULTS_PER_QUERY = 5
PACK_PASSAGES = 6

# Words that make a query find primary sources rather than listicles. Chosen per
# intent because "installation instructions" surfaces manufacturer PDFs while
# "code requirements" surfaces the jurisdictions that actually write the rule.
_INTENT_QUALIFIER = [
    (re.compile(r"\b(install|mount|hang|attach|anchor|fasten|bracket)\b", re.I),
     "installation instructions manufacturer"),
    (re.compile(r"\b(permit|code|allowed|legal|regulation|ordinance|hoa|covenant)\b", re.I),
     "building code requirements"),
    (re.compile(r"\b(replace|repair|fix|leak|broken|fault|troubleshoot)\b", re.I),
     "repair guide manufacturer"),
    (re.compile(r"\b(maintain|clean|service|filter|inspect|seasonal)\b", re.I),
     "maintenance schedule manufacturer"),
    (re.compile(r"\b(insulate|freeze|frozen|burst|heat|cool|hvac|energy)\b", re.I),
     "energy.gov guidance"),
]
_DEFAULT_QUALIFIER = "how to guide"

# A SITE RESTRICTION, not another qualifier phrase — and the difference is the
# whole point of this table.
#
# Measured on DuckDuckGo: appending the text "energy.gov guidance" to a
# burst-pipe question returned **zero** energy.gov pages across twelve results,
# and nothing the authority table rated at all. `site:energy.gov` on the same
# question returned energy.gov three times out of three. A qualifier is a hint
# the engine may ignore; `site:` is a filter it cannot.
#
# This matters beyond tidiness. `evidence.rank_passages` weights authority at
# 0.25, and free-tier search returns almost entirely unrated SEO pages — so
# without this the authority term had nothing to discriminate on and ranking was
# effectively relevance-only. The demo was showing the pipeline while quietly not
# showing the thing the pipeline is for.
#
# One restricted query is added ALONGSIDE the unrestricted ones, never instead of
# them: a site filter that finds nothing must not be able to empty the pack.
_INTENT_SITES = [
    (re.compile(r"\b(permit|code|allowed|legal|regulation|ordinance|zoning|"
                r"inspection)\b", re.I),
     "site:.gov"),
    (re.compile(r"\b(insulate|insulation|freeze|frozen|burst|draft|energy|"
                r"heating|cooling|hvac|efficien\w*)\b", re.I),
     "site:energy.gov OR site:.gov"),
    (re.compile(r"\b(install|mount|hang|attach|anchor|fasten|bracket|stud)\b", re.I),
     "site:familyhandyman.com OR site:thisoldhouse.com OR site:finehomebuilding.com"),
    (re.compile(r"\b(replace|repair|fix|leak|maintain|clean|service|filter|"
                r"inspect|gutter)\b", re.I),
     "site:familyhandyman.com OR site:thisoldhouse.com"),
]


def plan_queries(question: str, home_context: str = "") -> list[str]:
    """Two to three targeted searches from one question. No LLM call."""
    base = " ".join((question or "").split())
    if not base:
        return []
    queries = [base]

    if home_context.strip():
        queries.append(f"{base} {home_context.strip()}")

    qualifier = _DEFAULT_QUALIFIER
    for pattern, phrase in _INTENT_QUALIFIER:
        if pattern.search(base):
            qualifier = phrase
            break
    queries.append(f"{base} {qualifier}")

    # The site-restricted variant goes LAST, so if MAX_QUERIES is ever tightened
    # the unrestricted searches are the ones that survive. Authoritative sources
    # are the upside; being able to answer at all is the floor.
    for pattern, restriction in _INTENT_SITES:
        if pattern.search(base):
            queries.append(f"{base} {restriction}")
            break

    seen, unique = set(), []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique[:MAX_QUERIES]


def home_context_for(home: dict | None) -> str:
    """The few home facts that change which answer is right.

    Climate zone and wall construction, not the whole profile: a search string is
    not a place to put an address, and the point is to bias results toward the
    right physical situation, not to describe the house.
    """
    if not home:
        return ""
    bits = [home.get("climate_zone"), home.get("wall_construction")]
    text = " ".join(str(b) for b in bits if b)
    return " ".join(text.split()[:12])


def research(question: str, home: dict | None = None, *,
             passages: int = PACK_PASSAGES) -> dict:
    """Run the full pipeline and return an evidence pack.

    Returns `{ok, question, queries, provider, pack, error}`. `pack` is ready for
    `evidence.render()` and `evidence.resolve_citations()`.
    """
    queries = plan_queries(question, home_context_for(home))
    if not queries:
        return {"ok": False, "question": question, "queries": [], "provider": "",
                "pack": {"passages": [], "dropped": [], "domains": []},
                "error": "empty question"}

    with telemetry.span("research", "research.run", f"Researching: {question[:60]}") as span:
        # Independent searches, and the network is the whole cost — so run them
        # together. Threads rather than asyncio because every provider client
        # here is blocking.
        merged: list = []
        provider_used = ""
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=min(len(queries), MAX_QUERIES)) as pool:
            futures = [pool.submit(providers.search, q, RESULTS_PER_QUERY) for q in queries]
            for future in futures:
                try:
                    out = future.result()
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    continue
                if out.get("ok"):
                    provider_used = provider_used or out.get("provider", "")
                    merged.extend(out.get("_results") or [])
                elif out.get("error"):
                    errors.append(out["error"])

        # Dedupe ACROSS queries as well as within one. Three variations of the
        # same question return overlapping results by design; without this the
        # pack would spend its slots on the same page found three ways.
        merged = providers.dedupe(merged, per_domain=2)
        pack = evidence.build_pack(question, merged, limit=passages)

        injections = [d for d in pack["dropped"] if d.get("injection")]
        irrelevant = [d for d in pack["dropped"] if d.get("irrelevant")]
        crowding = [d for d in pack["dropped"] if d.get("crowding")]
        span.update({"queries": len(queries), "results": len(merged),
                     "passages": len(pack["passages"]), "provider": provider_used,
                     "dropped": len(pack["dropped"]), "injections": len(injections),
                     "irrelevant": len(irrelevant), "crowding": len(crowding)})

    if injections:
        # Visible, not silent: a page trying to steer the assistant is exactly the
        # thing an operator should be able to see happened.
        telemetry.record(
            "safety", "safety.injection_dropped",
            f"Dropped {len(injections)} retrieved passage(s) that tried to issue instructions",
            level="warn",
            data={"sources": [d["domain"] for d in injections][:5]})

    if irrelevant and not pack["passages"]:
        # "The search found nothing" and "the search worked and none of it was
        # about the question" look identical from outside and are completely
        # different faults — one is a provider problem, the other is a query
        # problem. Recording the second is what makes refusing to cite legible
        # rather than looking like a silent failure.
        telemetry.record(
            "research", "research.no_relevant_evidence",
            f"Searched {len(merged)} result(s) and kept none — every passage scored "
            f"below the relevance floor",
            level="warn",
            data={"dropped": len(irrelevant),
                  "sources": [d["domain"] for d in irrelevant][:5]})

    return {"ok": bool(pack["passages"]), "question": question, "queries": queries,
            "provider": provider_used, "pack": pack,
            "error": "" if pack["passages"] else ("; ".join(errors) or "no usable evidence")}


if __name__ == "__main__":
    import json
    import sys

    import config
    from tools.homes import load_home

    question = " ".join(sys.argv[1:]) or "How do I hang a 20 lb mirror on drywall?"
    home = load_home()
    print(f"profile={config.build_profile()}  providers={providers.available_providers()}")
    print(f"home context: {home_context_for(home)!r}\n")

    for q in plan_queries(question, home_context_for(home)):
        print("  query:", q)

    out = research(question, home)
    pack = out["pack"]
    print(f"\nok={out['ok']} provider={out['provider']} "
          f"passages={len(pack['passages'])} dropped={len(pack['dropped'])}")
    print(f"domains: {', '.join(pack['domains']) or '-'}\n")
    for c in pack["passages"]:
        print(f"  {c['ref']}  score={c['score']:.3f}  rel={c['relevance']:.2f}  "
              f"auth={c['authority']:.2f} [{c['authority_label']}]  {c['domain']}")
        print(f"       {c['text'][:110]}...")
    for d in pack["dropped"][:6]:
        print(f"  DROP {d['domain']:<24} {d['reason'][:60]}")
    print(f"\nrendered pack: {len(evidence.render(pack))} chars "
          f"(the old pipeline gave the model ~640)")
