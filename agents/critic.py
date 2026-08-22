"""The Critic: evaluates candidate approaches for the beam search.

Evaluation is deliberately split in two, and the split is the point.

`evaluate_deterministic` computes the three criteria that decide whether an
approach is ACCEPTABLE — safety, permission, cost — plus the three hard gates. It
uses the existing safety guardrails and the retrieved policy passages, never a
model. 63% of the rubric weight lives here.

`critique` asks a model for the three genuinely subjective criteria only:
suitability, reversibility, effort. That is 37% of the weight, and it is the only
part a noisy evaluator can move.

Measured justification for the split, on the then-pinned
openai/gpt-oss-20b:free: asking the model for all six criteria plus the gates
averaged 60.7s per call; asking for these three averaged 21.7s. The design and
the latency point the same way. (Those absolutes are HISTORICAL - that slug was
withdrawn from the free tier on 2026-08-21. The argument is the 3x ratio, not
the seconds, and the split would be right even on an instant model.)
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents.rubric import GATE_NAMES, NEUTRAL, Node
from tools.safety import check_high_risk

# Language that means a node leaves something permanent behind. A renter doing
# any of these needs written permission, which is a gate rather than a penalty.
_PERMANENT = (
    "drill", "drilled", "drilling", "screw", "screws", "anchor", "anchors",
    "toggle", "stud", "nail", "nails", "bolt", "bolts", "mount permanently",
    "permanent", "cut into", "patch the wall", "hardwire", "hard-wire",
)

# Reversible / non-destructive approaches. Used to reward, never to gate.
_REVERSIBLE = (
    "adhesive", "command strip", "removable", "tension", "freestanding",
    "leaning", "no-drill", "damage-free", "rail system", "over-the-door",
)

# Prohibition language in a retrieved passage. Deliberately narrow: a passage
# saying "requires approval" is a restriction, not a prohibition, and treating
# the two the same would gate away perfectly legal options.
_PROHIBITION = (
    "not permitted", "are prohibited", "is prohibited", "may not", "must not",
    "shall not", "no owner may", "forbidden",
)
_RESTRICTION = (
    "requires written", "requires approval", "requires arc", "prior approval",
    "written permission", "approval before", "must be approved",
)

# Hiring a professional. Scores cost lower, safety higher.
_HIRE = ("hire", "professional", "contractor", "handyman", "pro ", " pro")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "with", "for", "into",
    "your", "my", "this", "that", "it", "is", "are", "be", "use", "using", "then",
}


def _terms(text: str) -> set[str]:
    return {w.strip(".,;:()").lower() for w in text.split()
            if len(w) > 3 and w.lower() not in _STOPWORDS}


def _mentions(text: str, needles) -> bool:
    low = text.lower()
    return any(n in low for n in needles)


def evaluate_deterministic(node: Node, *, persona: str = "owner",
                           policies: list[dict] | None = None,
                           hire_rate_usd: float | None = None,
                           question_high_risk: bool = False) -> tuple[dict, dict]:
    """Score safety, permission and cost, and evaluate the three hard gates.

    Returns `(scores, gates)`. No LLM is involved, so this is reproducible and
    unit-testable, and a model cannot argue its way past a gate.

    `question_high_risk` is the verdict the safety screen already reached about
    the QUESTION, and passing it is not an optimisation — see gate 1.
    """
    policies = policies or []
    text = f"{node.name} {node.summary}".strip()
    permanent = _mentions(text, _PERMANENT)
    reversible = _mentions(text, _REVERSIBLE)
    hiring = _mentions(text, _HIRE)

    scores: dict[str, float] = {}
    gates: dict[str, bool] = {g: False for g in GATE_NAMES}

    # --- gate 1: high-risk work -------------------------------------------------
    # Reuses the same classifier that makes the assistant refuse to describe this
    # work at all. If we refuse to explain it, we must not rank it either.
    #
    # The node's own text is NOT sufficient evidence, and relying on it alone was
    # a hole. The classifier reads the words in front of it, and the words in
    # front of it here are the model's paraphrase of an approach, not the user's
    # question. Asked "how do I replace my breaker box myself", a model proposes
    # branches like "Install a subpanel instead" or "Swap individual breakers
    # only" — both unmistakably service-panel work, neither containing a phrase
    # the hazard table matches. Measured: those two branches scored 8.0 on safety
    # and entered the beam, on a question the orchestrator had already refused to
    # answer. The refusal and the ranking disagreed about the same job.
    #
    # So the question's verdict is authoritative and the node's can only add to
    # it. A branch cannot escape the gate by being described in safer words than
    # the thing it is a branch of.
    risk = check_high_risk(text)
    high_risk = bool(question_high_risk) or bool(risk and risk.get("high_risk"))
    # Hiring a professional for high-risk work is the CORRECT answer, so the gate
    # applies to doing it yourself, not to delegating it.
    gates["high_risk_work"] = high_risk and not hiring

    # --- gate 2: the occupant's role --------------------------------------------
    gates["occupant_not_permitted"] = (
        persona == "renter" and permanent and not hiring)

    # --- gate 3 + permission_fit: what the home's own documents say --------------
    node_terms = _terms(text)
    prohibited = False
    restricted = False
    for p in policies:
        passage = str(p.get("text", ""))
        if not passage:
            continue
        # A tenant-scoped rule does not bind an owner. The retrieval audience
        # filter normally keeps these apart, but a gate that silently depends on
        # an upstream filter being correct is a gate that will eventually fire on
        # the wrong person: without this, an owner's perfectly legal toggle bolt
        # is vetoed because a tenant clause happens to share the word "anchors".
        if persona == "owner" and _mentions(passage, ("tenant", "renter", "lease")):
            continue
        shared = node_terms & _terms(passage)
        # One shared word is not evidence. An early version gated on any overlap
        # at all, and pruned EVERY branch of a renter's mirror question — including
        # the adhesive rail, which is precisely the approach that avoids the rule,
        # and hiring a professional. It shared the word "mounting" with a clause
        # about mounting a television bracket, and that was enough.
        if len(shared) < 2:
            continue
        if _mentions(passage, _PROHIBITION):
            # A prohibition only binds a node that actually has the prohibited
            # property. These rules forbid PERMANENT alterations, so an explicitly
            # reversible approach is not what they are talking about. Gating it
            # anyway is not "erring on the safe side" — it removes the safe option
            # and leaves the user with nothing.
            #
            # Hiring is excluded for the same reason it is excluded from the
            # occupant gate: it is a delivery mechanism, not a change of
            # permission. "Get written permission, then have someone do it" is a
            # real path, so this is a restriction. Gating it produced a live
            # answer asserting "the home documents prohibit hiring a professional"
            # — which no document says, and which the gate itself invented.
            if permanent and not reversible and not hiring:
                prohibited = True
            elif not reversible:
                # Hiring, or an approach whose permanence is unclear: the rule is
                # live but a permission path exists, so restrict rather than veto.
                restricted = True
            # A clearly reversible approach falls through entirely. A rule about
            # PERMANENT alterations is not about an adhesive strip, and marking it
            # "restricted" is not a harmless caution: it costs 4 points of
            # permission fit and handed a renter's mirror question to "hire a
            # handyman", beating the cheaper, safer, fully-permitted option. An
            # over-cautious score is still a wrong answer.
        elif _mentions(passage, _RESTRICTION):
            restricted = True
    gates["prohibited_by_rule"] = prohibited

    if prohibited:
        scores["permission_fit"] = 0.0
    elif restricted:
        scores["permission_fit"] = 4.0          # allowed, but needs approval first
    elif persona == "renter" and permanent:
        scores["permission_fit"] = 2.0
    elif policies:
        scores["permission_fit"] = 8.0
    else:
        # No retrieved passage addresses this. Neutral, never optimistic: an
        # unexamined permission is not the same as a granted one.
        scores["permission_fit"] = NEUTRAL

    # --- safety -----------------------------------------------------------------
    if hiring:
        scores["safety_risk"] = 10.0
    elif high_risk:
        scores["safety_risk"] = 1.0
    elif permanent:
        scores["safety_risk"] = 7.0
    elif reversible:
        scores["safety_risk"] = 9.0
    else:
        scores["safety_risk"] = 8.0

    # --- cost -------------------------------------------------------------------
    # 10 is cheapest, so the whole score vector points the same way.
    if hiring:
        rate = hire_rate_usd or 120.0
        scores["cost"] = max(1.0, min(6.0, 6.0 - (rate - 80.0) / 40.0))
    elif reversible:
        scores["cost"] = 9.0
    elif permanent:
        scores["cost"] = 8.0
    else:
        scores["cost"] = 7.0

    return scores, gates


_CRITIC_SYSTEM = (
    "You are the CRITIC in a Tree-of-Thought search over home-improvement "
    "approaches. You do NOT choose a winner and you do NOT write advice. You score.\n"
    "\n"
    "For each candidate, score three criteria as INTEGERS 0-10 (10 is best):\n"
    "  suitability   - how well it fits the item's weight, the wall type and the climate\n"
    "  reversibility - how little damage remains when it is removed (10 = none)\n"
    "  effort_skill  - how easy it is for an average person (10 = easiest)\n"
    "\n"
    "Safety, permission and cost are scored elsewhere. Do not report them.\n"
    'Return ONLY a JSON array, one object per candidate: [{"name": "<the exact '
    'candidate name>", "suitability": 0, "reversibility": 0, "effort_skill": 0, '
    '"why": "<one short line>"}]. No prose, no markdown, no code fence.'
)


def critique(nodes: list[Node], *, llm, task: str, grounding: str) -> list[dict]:
    """Ask the model for the three subjective criteria, batched into ONE call.

    Batching is what keeps a depth-2 search at four calls rather than twelve. A
    parse failure returns [] rather than a default, so `weighted_total` falls back
    to neutral per missing criterion — the old behaviour of scoring every branch a
    flat 5 silently defeated ranking altogether.
    """
    from agents.advisor import _extract_json

    candidates = [{"name": n.name, "summary": n.summary} for n in nodes]
    messages = [
        SystemMessage(content=_CRITIC_SYSTEM),
        HumanMessage(content=(
            f"TASK: {task}\n\nCONTEXT:\n{grounding}\n\n"
            f"CANDIDATES:\n{json.dumps(candidates, indent=2)}"
        )),
    ]
    try:
        raw = llm.invoke(messages).content
    except Exception:
        return []
    parsed = _extract_json(raw)
    return parsed if isinstance(parsed, list) else []


if __name__ == "__main__":
    # Deterministic half only — no LLM, no network.
    POLICIES = [{"text": "Tenants generally may not make permanent alterations "
                         "without written permission. That includes mounting a "
                         "television bracket and anything else that leaves anchors "
                         "in the wall."}]

    cases = [
        # The four that a live renter run pruned to nothing before the fix. Only
        # the two permanent ones may be gated.
        ("Adhesive Hook System", "Mounting the mirror with removable adhesive strips", "renter"),
        ("Professional Installation", "A handyman mounts the mirror on the wall", "renter"),
        ("Hire Professional", "A pro drills into wall studs and installs anchors", "renter"),
        ("Stud Mount with Screw", "Screw into a wall stud to carry the load", "renter"),
        ("Drywall Toggle Anchor", "Toggle anchors mounted in the wall", "renter"),
        ("Adhesive rail system", "Damage-free hanging rail rated over 20 lb", "renter"),
        ("Toggle bolt into drywall", "Toggle anchors carry the load", "renter"),
        ("Toggle bolt into drywall", "Toggle anchors carry the load", "owner"),
        ("Hire a handyman", "A professional mounts it", "renter"),
        ("Replace the electrical service panel", "Swap the main breaker panel", "owner"),
        ("Hire an electrician for the panel", "A licensed pro replaces it", "owner"),
    ]
    print(f"{'candidate':<38} {'persona':<7} {'safety':>6} {'perm':>5} {'cost':>5}  gates")
    for name, summary, persona in cases:
        n = Node(id="x", name=name, summary=summary)
        s, g = evaluate_deterministic(n, persona=persona, policies=POLICIES)
        fired = [k for k, v in g.items() if v] or ["-"]
        print(f"{name:<38} {persona:<7} {s['safety_risk']:>6.0f} "
              f"{s['permission_fit']:>5.0f} {s['cost']:>5.0f}  {', '.join(fired)}")
