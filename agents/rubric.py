"""The Advisor's evaluation rubric: gates, weights, scoring, pruning, tie-breaks.

Everything in this module is deterministic. No LLM call happens here — the Critic
supplies three subjective sub-scores and this module does the rest, so a run can
be reproduced exactly and a decision can be explained without replaying a model.

The central design decision is that **hard constraints are gates, not weights**.
A weighted average lets a cheap, easy, fast option outvote "the lease forbids
this" or "this is a gas line". Averaging a veto is how a scoring system produces
confident, catastrophic recommendations. So a gated node is removed for a stated
reason rather than scored low, and no amount of critic noise can promote it.

See docs/reasoning.md for the full design and its rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- Weights -------------------------------------------------------------------
# Sum to 1.0. Every criterion is scored 0-10, higher is better (including cost:
# 10 means cheapest, so the whole vector points the same way and the weighted sum
# needs no sign handling).
#
# Note the split: 0.63 of the weight comes from criteria this system computes
# itself, and only 0.37 from the Critic's judgement. That is the primary
# mitigation for the risk that weak evaluation signals prune the best branch --
# noise moves the ranking AMONG acceptable options rather than deciding
# acceptability.
WEIGHTS: dict[str, float] = {
    "safety_risk": 0.30,      # deterministic  - hazard table + safety module
    "permission_fit": 0.25,   # RAG-grounded   - must cite a passage, else neutral
    "suitability": 0.20,      # Critic LLM
    "reversibility": 0.10,    # Critic LLM
    "cost": 0.08,             # deterministic  - materials + pro rate
    "effort_skill": 0.07,     # Critic LLM
}

# What the Critic is actually asked for. Keeping this list short is not just
# tidiness: measured on the free model, asking for all six criteria plus the
# gates took 60.7s per call, while asking for these three took 21.7s -- a 2.8x
# difference that decides whether a depth-2 search is affordable at all.
CRITIC_CRITERIA = ("suitability", "reversibility", "effort_skill")

# Scored in Python, never by the model.
DETERMINISTIC_CRITERIA = ("safety_risk", "permission_fit", "cost")

NEUTRAL = 5.0   # used when a signal is genuinely unavailable, never as a default

# --- Pruning thresholds --------------------------------------------------------
ABSOLUTE_FLOOR = 4.0   # weighted total below this is pruned outright
RELATIVE_DROP = 3.0    # pruned when more than this far behind the best node
BEAM_WIDTH = 2         # survivors carried to the next depth

GATE_NAMES = ("occupant_not_permitted", "prohibited_by_rule", "high_risk_work")

GATE_REASONS = {
    "occupant_not_permitted": "the occupant's role does not permit this change",
    "prohibited_by_rule": "a rule in this home's documents prohibits it",
    "high_risk_work": "this is high-risk work that should not be attempted DIY",
}


@dataclass
class Node:
    """One candidate approach plus its evaluation.

    `scores` holds every criterion 0-10. `gates` holds the boolean vetoes.
    `status` is "live" until something prunes it, then "pruned" with a reason.
    """
    id: str
    name: str
    summary: str = ""
    depth: int = 1
    parent_id: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    gates: dict[str, bool] = field(default_factory=dict)
    why: str = ""
    total: float = 0.0
    status: str = "live"
    prune_reason: str | None = None
    order: int = 0            # stable proposal order, the final tie-break

    @property
    def gated(self) -> bool:
        return any(self.gates.get(g) for g in GATE_NAMES)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "summary": self.summary,
            "depth": self.depth, "parent_id": self.parent_id,
            "scores": dict(self.scores), "gates": dict(self.gates),
            "why": self.why, "total": round(self.total, 2),
            "status": self.status, "prune_reason": self.prune_reason,
        }


def weighted_total(scores: dict[str, float]) -> float:
    """Weighted sum over WEIGHTS. Missing criteria count as NEUTRAL, not zero.

    Zero would be a silent penalty for a field the Critic simply failed to emit,
    which turns a parsing hiccup into a ranking decision.
    """
    return round(sum(WEIGHTS[k] * float(scores.get(k, NEUTRAL)) for k in WEIGHTS), 3)


def score_node(node: Node) -> Node:
    """Apply gates and compute the weighted total. Mutates and returns the node."""
    fired = [g for g in GATE_NAMES if node.gates.get(g)]
    if fired:
        node.total = float("-inf")
        node.status = "pruned"
        node.prune_reason = GATE_REASONS[fired[0]]
        return node
    node.total = weighted_total(node.scores)
    return node


def prune(nodes: list[Node], beam_width: int = BEAM_WIDTH) -> list[Node]:
    """Apply the four pruning rules in order and return the surviving beam.

    Rules, in order:
      1. gate veto        - already applied by score_node
      2. absolute floor   - total < ABSOLUTE_FLOOR
      3. relative floor   - total < best - RELATIVE_DROP
      4. beam width       - keep the top `beam_width` of what remains

    Every pruned node keeps its reason. Nothing is deleted: the caller returns
    the whole tree so a wrongly-pruned branch is visible rather than invisible,
    which is what turns a silent failure into an observable one.
    """
    live = [n for n in nodes if n.status == "live"]

    for n in live:
        if n.total < ABSOLUTE_FLOOR:
            n.status, n.prune_reason = "pruned", (
                f"scored {n.total:.1f}, below the {ABSOLUTE_FLOOR:.1f} minimum")

    live = [n for n in nodes if n.status == "live"]
    if live:
        best = max(n.total for n in live)
        for n in live:
            if n.total < best - RELATIVE_DROP:
                n.status, n.prune_reason = "pruned", (
                    f"scored {n.total:.1f}, more than {RELATIVE_DROP:.1f} behind "
                    f"the leading option at {best:.1f}")

    live = sorted((n for n in nodes if n.status == "live"), key=rank_key)
    for n in live[beam_width:]:
        n.status, n.prune_reason = "pruned", (
            f"ranked below the top {beam_width} approaches")
    return live[:beam_width]


def rank_key(node: Node):
    """Sort key: best first. Deterministic, so a re-run reproduces the decision.

    Ordered tie-break, after the weighted total:
      1. higher safety sub-score
      2. higher permission fit (a grounded allowance beats an assumed one)
      3. higher cost score (cheaper)
      4. higher effort score (easier)
      5. stable proposal order

    No voting and no second model call: a tie-break that consults an LLM is a
    tie-break that cannot be reproduced or explained.
    """
    s = node.scores
    return (
        -node.total,
        -float(s.get("safety_risk", NEUTRAL)),
        -float(s.get("permission_fit", NEUTRAL)),
        -float(s.get("cost", NEUTRAL)),
        -float(s.get("effort_skill", NEUTRAL)),
        node.order,
    )


def select(nodes: list[Node]) -> Node | None:
    """Pure-Python argmax over live nodes. Returns None when everything is pruned.

    Returning None is a correct and meaningful outcome, not an error: it means no
    safe approach survived, and the caller must escalate to a professional rather
    than relax a threshold and recommend the least-bad option.
    """
    live = [n for n in nodes if n.status == "live"]
    return min(live, key=rank_key) if live else None


if __name__ == "__main__":
    # Self-test: the whole evaluation policy, exercised with no LLM and no network.
    def mk(i, name, safety=8, perm=8, suit=8, rev=8, cost=8, eff=8, **gates):
        return Node(id=f"n{i}", name=name, order=i, scores={
            "safety_risk": safety, "permission_fit": perm, "suitability": suit,
            "reversibility": rev, "cost": cost, "effort_skill": eff},
            gates={g: gates.get(g, False) for g in GATE_NAMES})

    print("weights sum:", round(sum(WEIGHTS.values()), 6))
    print("deterministic share:", round(
        sum(WEIGHTS[k] for k in DETERMINISTIC_CRITERIA), 3))

    nodes = [
        mk(0, "Adhesive rail", safety=9, perm=10, suit=7, rev=10, cost=9, eff=9),
        mk(1, "Toggle bolt", safety=7, perm=2, suit=9, rev=3, cost=8, eff=6),
        mk(2, "Stud mount", safety=8, perm=1, suit=10, rev=2, cost=9, eff=5,
           occupant_not_permitted=True),
        mk(3, "Hire a pro", safety=10, perm=10, suit=8, rev=9, cost=2, eff=10),
    ]
    for n in nodes:
        score_node(n)
    beam = prune(nodes)

    print("\nscored:")
    for n in sorted(nodes, key=rank_key):
        t = "-inf" if n.total == float("-inf") else f"{n.total:5.2f}"
        print(f"  {t}  {n.name:<16} {n.status:<7} {n.prune_reason or ''}")
    print(f"\nbeam ({len(beam)}): {[n.name for n in beam]}")
    print("winner:", select(nodes).name)

    print("\n--- every branch gated -> select() must return None ---")
    allgated = [mk(i, f"opt{i}", high_risk_work=True) for i in range(3)]
    for n in allgated:
        score_node(n)
    prune(allgated)
    print("select:", select(allgated), "(None means: escalate to a professional)")
