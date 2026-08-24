"""The Advisor: a Tree-of-Thought sub-agent for DIY & maintenance decisions.

Unlike the main orchestrator (a ReAct loop), the Advisor runs an explicit
**beam search** over candidate approaches, so the reasoning is visible and the
pruning is inspectable:

    gather grounding -> PROPOSE (b=4) -> CRITIQUE -> PRUNE (k=2)
                     -> EXPAND (b=2)  -> CRITIQUE -> SELECT (argmax) -> compose

One question never reaches that loop. If `tools.safety` calls the question
high-risk work, the search has no decision left to make — every do-it-yourself
branch is gated and the survivor is "hire a professional", which is the hazard
table's own verdict. That case returns a referral directly, at zero model calls
and no web fetch, with an empty tree and a strategy that says the search was
skipped rather than dressing up a search that never ran.

"Gather" grounds the reasoning in real inputs: the home profile, the policy
knowledge base (e.g. renter drilling rules), the contractor directory, and a
live web search (cited). Everything after that is delegated:

  * `agents/beam.py`   the search loop and the budget
  * `agents/rubric.py` weights, gates, pruning rules, tie-breaks, argmax
  * `agents/critic.py` evaluation: deterministic for safety/permission/cost,
                       one batched LLM call for the three subjective criteria

This module is the glue plus the prompts that are genuinely its own: how to
propose approaches, how to expand them, and how to write the chosen one up.

Selection is a pure-Python argmax. It used to be a third LLM call handed all the
scores and asked to "choose the best (or top two if close)" — so the options
table the UI displayed need not correspond to the recommendation, and nothing
detected the mismatch.

See docs/reasoning.md for the design and its rationale.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage

import telemetry
from agents.beam import AdvisorBudget
from agents.beam import search as beam_search
from agents.critic import critique, evaluate_deterministic
from agents.llm import build_llm
from memory.rag_store import search_policies
from tools.contractors import find_contractors
from tools.homes import current_home_id, load_home
from tools.pros import trades
from tools.safety import check_high_risk
from agents.researcher import research
from tools.research import evidence


def _extract_json(text: str):
    """Best-effort parse of the first JSON array/object in an LLM reply."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


def _gather(question: str, persona: str = "owner", home_id: str | None = None) -> dict:
    """Collect grounding for the decision (home, rules, contractors, web).

    The three lookups are independent, and the web search is by far the slowest
    (a live DuckDuckGo round trip against a local vector query and a CSV read), so
    they run concurrently instead of queueing behind it. Threads rather than
    asyncio because each one is blocking I/O.

    `home_id` is passed in explicitly rather than resolved here, and that is not
    incidental: the active home is read from a LangGraph context variable, and
    context variables do NOT propagate into ThreadPoolExecutor workers. Resolved
    inside these threads it would silently fall back to the primary home, so the
    Dallas home would quietly get Minneapolis rules — a wrong answer with no error.
    """
    home = load_home(home_id)

    def _policies() -> list[dict]:
        try:
            # Narrowed to the occupant's role for the same reason the orchestrator
            # does it: a renter must not be advised against owner-only covenants.
            return search_policies(question, k=3, audience=persona, home_id=home_id)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_policies = pool.submit(_policies)
        f_web = pool.submit(research, question, home)
        f_contractors = pool.submit(find_contractors, question, limit=3, home_id=home_id)
        policies = f_policies.result()
        # Grounding is best-effort: a failed web search should cost the advisor
        # its citations, not the whole recommendation.
        try:
            web = f_web.result()
        except Exception:
            web = {}
        try:
            contractors = f_contractors.result()
        except Exception:
            contractors = []

    return {"home": home, "policies": policies, "web": web, "contractors": contractors}


def _grounding_text(ctx: dict, persona: str) -> str:
    home = ctx["home"]
    # Location and climate zone are in the grounding because the right answer is
    # regional: moss treatment and crawlspace freeze prep in a marine climate,
    # attic heat and Arctic-front pipe protection in a mixed-humid one.
    lines = [
        f"HOME: {home.get('dwelling_type')} at {home.get('address')} "
        f"(climate zone {home.get('climate_zone')}), built {home.get('year_built')}, "
        f"walls: {home.get('wall_construction')}. Occupant role: {persona}.",
    ]
    if ctx["policies"]:
        lines.append("RELEVANT RULES (from the home's documents):")
        for p in ctx["policies"][:3]:
            lines.append(f"  - {p['citation']}: {p['text'][:220]}")
    # Web evidence is deliberately NOT here. This text is interpolated into system
    # prompts, and retrieved pages must never reach one — see _evidence_block.
    return "\n".join(lines)


def _evidence_block(ctx: dict) -> str:
    """Retrieved web evidence, delimited and labelled as untrusted quoted data.

    Kept out of `_grounding_text` and appended by callers separately, because
    grounding text goes into a SystemMessage and this must not. That separation is
    the load-bearing layer of the R14 mitigation (tools/research/untrusted.py), and
    it only holds while the two stay apart — which is why they are two functions
    rather than one with a flag.

    This replaces the old grounding line, which gave the model four search results
    truncated to 160 characters each.
    """
    return evidence.render((ctx.get("web") or {}).get("pack") or {})


def _evidence_summary(ctx: dict) -> dict:
    """A compact, renderable view of the evidence pack, for the UI.

    Deliberately small: this rides inside a tool result that the stream
    truncates, and it is paid for in tokens on a free model. Enough to show
    *which* sources were used, how far each was trusted, and — the part with no
    other surface — **what was screened out and why**.

    The drops are the point. A pipeline that discards a hostile page silently is
    indistinguishable, from the outside, from one that never saw it; and a pack
    that came back empty because everything was irrelevant looks exactly like a
    search that failed. Both distinctions were invisible until this existed.
    """
    web = ctx.get("web") or {}
    pack = web.get("pack") or {}
    return {
        "provider": web.get("provider", ""),
        "queries": [q[:110] for q in (web.get("queries") or [])],
        "passages": [
            {"ref": c["ref"], "domain": c["domain"], "url": c["url"],
             "authority": c["authority_label"], "score": c["score"]}
            for c in pack.get("passages", [])
        ],
        "dropped": [
            {"domain": d.get("domain") or "unknown", "reason": (d.get("reason") or "")[:72]}
            for d in pack.get("dropped", [])
        ][:6],
    }


_NO_EVIDENCE = {"provider": "", "queries": [], "passages": [], "dropped": []}


_PROPOSE_SYSTEM = (
    "You are a home-improvement advisor generating the first level of a "
    "Tree-of-Thought search. Propose exactly 4 DISTINCT candidate approaches to the "
    "task — genuinely different strategies, not rewordings of one. Include a "
    "'hire a professional' option when the job plausibly warrants it. Do NOT rank "
    "them and do NOT recommend one; scoring happens elsewhere.\n"
    'Return ONLY a JSON array: [{"name": "<short label>", "summary": "<one sentence>"}]. '
    "No prose, no markdown, no code fence."
)

_EXPAND_SYSTEM = (
    "You are expanding the surviving branches of a Tree-of-Thought search into "
    "concrete execution plans. For EACH strategy you are given, propose exactly 2 "
    "specific ways to carry it out that differ in materials, effort or finish — not "
    "restatements of the strategy.\n"
    'Return ONLY a JSON object mapping each strategy id to its variants: '
    '{"<id>": [{"name": "<short label>", "summary": "<one sentence>"}]}. '
    "No prose, no markdown, no code fence."
)

_COMPOSE_SYSTEM = (
    "You are the home advisor writing the final recommendation. The approach has "
    "ALREADY BEEN CHOSEN for you by a scored search — your job is to explain and "
    "detail it, NOT to reconsider it. Do not substitute a different approach.\n"
    "\n"
    "Write: (1) the recommended method and why, in one bold opening line; (2) a short "
    "'Options I compared' list with each option's score, marking any that were ruled "
    "out and why; (3) the exact tools and materials; (4) key safety notes. Add a "
    "one-line disclaimer that this is general guidance, not professional advice.\n"
    "\n"
    # This used to read "cite any web source by URL", which contradicted the
    # evidence preamble travelling in the same turn ("cite a passage by its
    # reference, e.g. [E2]") — and the system message won. The model therefore
    # wrote its own trailing list of bare domains, `resolve_citations` found no
    # references to resolve, and the answer ended with unlinked plain text while
    # the real URLs sat unused in the pack.
    "CITATIONS: cite a web source ONLY by its bracketed reference — [E1], [E2] — "
    "placed inline where you use it. Never write a URL, a bare domain, or a "
    "trailing 'Sources:' list; the references are turned into links for you, and "
    "an invented one is rendered as '[unknown source]'. Cite a rule from the "
    "home's documents by its document name.\n"
    "\n"
    # Markdown reaches a renderer that deliberately does not execute HTML, since
    # answers carry retrieved web text. A <br> therefore printed as literal
    # characters inside table cells.
    "FORMATTING: markdown only, never HTML — no <br>, no <b>, no tags of any "
    "kind. For several points inside a table cell, separate them with '; ' or "
    "use a list outside the table instead."
)


def _propose(llm, question: str, grounding: str, ev: str = "") -> list[dict]:
    """Level 1 of the tree: distinct strategies. One LLM call."""
    raw = llm.invoke([
        SystemMessage(content=_PROPOSE_SYSTEM),
        # Retrieved evidence rides in the HUMAN message, never the system one.
        HumanMessage(content=f"TASK: {question}\n\nCONTEXT:\n{grounding}\n\n{ev}"),
    ]).content
    parsed = _extract_json(raw)
    if isinstance(parsed, list) and parsed:
        return [{"name": str(o.get("name", f"Option {i + 1}")),
                 "summary": str(o.get("summary", ""))}
                for i, o in enumerate(parsed) if isinstance(o, dict)]
    # Never dead-end: a single node still gets scored, gated and reported honestly.
    return [{"name": "General approach", "summary": str(raw)[:200]}]


def _expand(llm, question: str, grounding: str, nodes, ev: str = "") -> dict:
    """Level 2: concrete executions for the survivors. One batched call for all."""
    payload = [{"id": n.id, "name": n.name, "summary": n.summary} for n in nodes]
    raw = llm.invoke([
        SystemMessage(content=_EXPAND_SYSTEM),
        HumanMessage(content=(f"TASK: {question}\n\nCONTEXT:\n{grounding}\n\n{ev}\n\n"
                              f"STRATEGIES:\n{json.dumps(payload, indent=2)}")),
    ]).content
    parsed = _extract_json(raw)
    if isinstance(parsed, dict):
        return {k: v for k, v in parsed.items() if isinstance(v, list)}
    return {}


def _compose(llm, question: str, grounding: str, result: dict,
             contractors: list[dict], persona: str, ev: str = "") -> str:
    """Write up the ALREADY-CHOSEN node. One LLM call, and it cannot re-decide."""
    considered = [
        {"name": n["name"],
         "score": None if n["total"] == float("-inf") else n["total"],
         "status": n["status"], "ruled_out_because": n["prune_reason"]}
        for n in result["nodes"]
    ]
    note = ""
    if contractors:
        c = contractors[0]
        note = (f"If a pro is warranted, a suitable option is {c['name']} "
                f"({c['trade']}, {c['rating']}★, ${c['hourly_rate_usd']}/hr, "
                f"{c['availability']}).")
    return llm.invoke([
        SystemMessage(content=_COMPOSE_SYSTEM),
        HumanMessage(content=(
            f"TASK: {question}\nOccupant role: {persona}\n\nCONTEXT:\n{grounding}\n\n{ev}\n\n"
            f"THE CHOSEN APPROACH (write this one up):\n"
            f"{json.dumps(result['winner'], indent=2)}\n\n"
            f"ALL OPTIONS CONSIDERED:\n{json.dumps(considered, indent=2)}\n\n{note}"
        )),
    ]).content


def _escalate(result: dict, contractors: list[dict]) -> str:
    """Every branch was pruned. Say so plainly instead of inventing a survivor.

    Written in Python, deliberately. This is the outcome where a model would be
    most tempted to be helpful by relaxing the very constraint that produced it,
    and being correct here costs no LLM call at all.
    """
    reasons, seen = [], set()
    for n in result["nodes"]:
        reason = n.get("prune_reason")
        if reason and reason not in seen:
            seen.add(reason)
            reasons.append(f"- **{n['name']}** — {reason}")
    pro = ""
    if contractors:
        c = contractors[0]
        pro = (f"\n\nA licensed option for this work: **{c['name']}** "
               f"({c['trade']}, {c['rating']}★, ${c['hourly_rate_usd']}/hr, "
               f"{c['availability']}).")
    return (
        "**I can't recommend a safe do-it-yourself approach for this job.**\n\n"
        "Every approach I considered was ruled out:\n"
        + "\n".join(reasons)
        + "\n\nThis is a case for a licensed professional rather than a workaround."
        + pro
        + "\n\nThis is general guidance, not professional advice."
    )


def _referral_contractors(question: str,
                          home_id: str | None) -> tuple[list[dict], str | None]:
    """The pro list for a referral, **and the trade it was actually matched on**.

    Best-effort: a directory miss is not a failure. A CSV read in the demo build
    and a licence-registry query in the full one — milliseconds either way, which
    is the whole reason the referral path can skip the search and still name
    someone real.

    The trade label travels with the rows because `find_pros` returns a *browse*
    of the directory when nothing matched, and a browse rendered under the same
    heading as a match is a false claim. Nothing here can tell the two apart from
    the rows alone — every row looks equally legitimate, because every row IS a
    legitimately licensed professional. The difference is only whether any of
    them do this job.
    """
    try:
        rows = find_contractors(question, limit=3, home_id=home_id)
    except Exception:
        return [], None
    try:
        matched = trades.identify(question)
    except Exception:
        matched = []
    return rows, (matched[0].label if matched else None)


def _refer_to_professional(risk: dict, contractors: list[dict],
                           trade_label: str | None = None) -> str:
    """The referral itself, written in Python.

    Deliberately not a model call. The content is fixed by the hazard table —
    what the category is, who it belongs to, and who is available — so generating
    it would spend a request to reproduce three values we already hold, on the
    one class of question where being talked out of the answer is least
    acceptable. `_escalate` makes the same argument for the same reason.

    The orchestrator still writes the user-facing prose around this; it arrives
    there as a tool result, not as the final answer.
    """
    lines = [
        f"**This is {risk['category']} — I can't walk you through doing it yourself.**",
        "",
        f"Work of this kind belongs to {risk['refer_to']}. Doing it without the "
        "licence, permit and inspection risks fire, electrocution, an insurance "
        "claim being denied, and a failed sale inspection later.",
        "",
        "What I can help with instead: what the job involves, what it typically "
        "costs, what permits and inspections apply, and how to pick and check a "
        "contractor.",
    ]
    if contractors:
        # Two different claims, so two different headings. With a matched trade
        # these are candidates FOR THE JOB; without one they are simply the
        # nearby licensed professionals, and saying so costs a line of prose
        # against a user ringing a handyman about a gas line.
        lines += ["", (
            f"Licensed {trade_label.lower()} professionals from your directory:"
            if trade_label else
            "From your directory — no listed trade matches this job, so these are "
            "nearby licensed professionals rather than candidates for it:")]
        for c in contractors:
            lines.append(
                f"- **{c['name']}** — {c['trade']}, {c['rating']}★, "
                f"${c['hourly_rate_usd']}/hr, {c['availability']}"
            )
    lines += ["", "This is general guidance, not professional advice."]
    return "\n".join(lines)


def run_advisor(question: str, persona: str = "owner", model: str | None = None,
                home_id: str | None = None, depth: int | None = None) -> dict:
    """Run the beam search and return the full tree plus a written recommendation."""
    # Resolved here, on the calling thread, while the graph's context variable is
    # still readable — see the note in `_gather`. The beam adds more fan-out, so
    # this hazard gets worse rather than better.
    home_id = home_id or current_home_id()

    # High-risk work: the outcome of the search is already determined, so don't
    # run it.
    #
    # Every do-it-yourself branch is gated by `evaluate_deterministic`, and the
    # only survivor is "hire a professional" — which is the answer the hazard
    # table produced before the first token was generated. Searching anyway cost
    # a live web fetch and five model calls to arrive back where it started, and
    # on a free-tier model that is the difference between a few seconds and
    # several minutes of a user watching a spinner after the guardrail has
    # visibly already fired.
    #
    # This is the same ladder the orchestrator already climbs: an emergency
    # bypasses the model entirely, an ordinary question gets the full loop. High
    # risk sits between the two and had been lumped in with "ordinary".
    #
    # The tree is returned EMPTY and the strategy says so, rather than
    # manufacturing a plausible-looking search that never happened. A reasoning
    # panel showing four branches that were never proposed would be a nicer
    # picture and a false one.
    risk = check_high_risk(question)
    if risk.get("high_risk"):
        contractors, referral_trade = _referral_contractors(question, home_id)
        telemetry.record("agent", "advisor.high_risk_referral",
                         f"Skipped the beam — {risk['category']} is a referral, not a decision",
                         level="warn",
                         data={"category": risk["category"], "llm_calls": 0})
        return {
            "question": question,
            "home_id": home_id,
            "final_answer": _refer_to_professional(risk, contractors, referral_trade),
            "evaluations": [],
            "tree": [],
            "winner": None,
            "strategy": "skipped — high-risk work is a referral, not a decision",
            "llm_calls": 0,
            "truncated": False,
            "truncated_because": None,
            "high_risk": risk,
            # No search ran, so there is nothing to show — and an empty panel
            # saying so is more honest than omitting the field, which the UI
            # would render identically to "the search found nothing".
            "evidence": _NO_EVIDENCE,
            "sources": {"web": [], "policies": [],
                        "contractors": [c["name"] for c in contractors]},
        }

    llm = build_llm(model=model)
    ctx = _gather(question, persona=persona, home_id=home_id)
    grounding = _grounding_text(ctx, persona)
    ev = _evidence_block(ctx)

    hire_rate = None
    if ctx["contractors"]:
        try:
            hire_rate = float(ctx["contractors"][0]["hourly_rate_usd"])
        except (KeyError, TypeError, ValueError):
            hire_rate = None

    budget = AdvisorBudget()
    with telemetry.span("agent", "advisor.search",
                        f"Beam search: {question[:60]}") as span:
        result = beam_search(
            propose=lambda: _propose(llm, question, grounding, ev),
            expand=lambda nodes: _expand(llm, question, grounding, nodes, ev),
            critique=lambda nodes: critique(nodes, llm=llm, task=question,
                                            grounding=grounding),
            # `question_high_risk` is False on this path by construction — the
            # high-risk case returned above — but it is passed explicitly rather
            # than defaulted, so the gate's dependency on the question's verdict
            # is visible at the call site and survives the two being reconnected
            # differently later.
            evaluate_deterministic=lambda node: evaluate_deterministic(
                node, persona=persona, policies=ctx["policies"],
                hire_rate_usd=hire_rate, question_high_risk=False),
            depth=depth,
            budget=budget,
        )
        span.update({"strategy": result["strategy"], "llm_calls": result["llm_calls"],
                     "nodes": len(result["nodes"]),
                     "winner": (result["winner"] or {}).get("name"),
                     "truncated": result["truncated"]})

    if result["winner"] is None:
        final_answer = _escalate(result, ctx["contractors"])
        telemetry.record("agent", "advisor.no_safe_option",
                         "Every branch was pruned; escalated to a professional",
                         level="warn", data={"nodes": len(result["nodes"])})
    else:
        final_answer = _compose(llm, question, grounding, result,
                                ctx["contractors"], persona, ev)
        # The model writes [E3]; Python resolves it to the real link. A
        # fabricated reference becomes "[unknown source]" rather than a
        # plausible URL nobody checks — the fourth R14 layer, and the one that
        # also catches ordinary citation invention.
        final_answer = evidence.resolve_citations(
            final_answer, (ctx["web"] or {}).get("pack") or {})

    # `evaluations` keeps the shape `ask_advisor` and the UI already consume: one
    # row per candidate with a comparable score, best first, gated options last.
    evaluations = [
        {"name": n["name"],
         "score": 0 if n["total"] == float("-inf") else round(n["total"], 1),
         "pros": n["why"], "cons": n["prune_reason"] or "",
         "status": n["status"], "depth": n["depth"]}
        for n in sorted(result["nodes"],
                        key=lambda n: (n["total"] != float("-inf"), n["total"]),
                        reverse=True)
    ]

    return {
        "question": question,
        "home_id": home_id,
        "final_answer": final_answer,
        "evaluations": evaluations,
        # The whole tree, pruned branches and reasons included. The UI renders
        # those struck through: a wrongly-pruned branch has to be *visible*, or
        # the risk that weak evaluation signals discard the best option stays
        # silent — which is the one failure mode this design cannot self-detect.
        "tree": result["nodes"],
        "winner": result["winner"],
        "strategy": result["strategy"],
        "llm_calls": result["llm_calls"] + 1,   # + the composing call
        "truncated": result["truncated"],
        "truncated_because": result["truncated_because"],
        "evidence": _evidence_summary(ctx),
        "sources": {
            "web": [c["url"] for c in
                    ((ctx["web"] or {}).get("pack") or {}).get("passages", [])],
            "policies": [p["citation"] for p in ctx["policies"]],
            "contractors": [c["name"] for c in ctx["contractors"]],
        },
    }


if __name__ == "__main__":
    import sys

    # The answer contains ★ and — , and a Windows console defaults to cp1252,
    # which cannot encode either. Only this smoke-test path prints to a console;
    # the API serialises to UTF-8 JSON and is unaffected.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    question = " ".join(sys.argv[1:]) or "How do I hang a 20 lb mirror on my wall?"
    r = run_advisor(question, persona="renter", depth=2)
    print(f"STRATEGY: {r['strategy']}   llm_calls={r['llm_calls']}   "
          f"truncated={r['truncated']}\n")
    for n in r["tree"]:
        total = "  -inf" if n["total"] == float("-inf") else f"{n['total']:6.2f}"
        mark = {"live": "*", "superseded": " ", "pruned": "x"}.get(n["status"], "?")
        print(f"  {mark} d{n['depth']} {total}  {n['name'][:38]:<38} "
              f"{n['prune_reason'] or ''}")
    print(f"\nWINNER: {(r['winner'] or {}).get('name', '(none — escalated)')}\n")
    print(r["final_answer"])
