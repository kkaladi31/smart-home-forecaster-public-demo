"""Beam search over candidate approaches — the Advisor's controller.

The search itself is pure control flow. Every operation that needs a language
model is injected as a callable, so the whole loop — expansion, gating, pruning,
tie-breaking, selection, budget exhaustion, the all-pruned path — is testable
with no model, no network and no cost. `python -m agents.beam` runs exactly that.

Why beam rather than BFS, DFS or MCTS, and where the parameters come from, is in
docs/reasoning.md. The short version: beam is BFS plus pruning, which is the
property this problem wants — compare fairly at each level, then stop paying for
what has already lost.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import config
from agents.rubric import BEAM_WIDTH, Node, prune, rank_key, score_node

# Branching factors per depth. b1 > BEAM_WIDTH on purpose: pruning has to be
# structural, or the "beam" is just BFS with extra steps.
BRANCH_AT_DEPTH_1 = 4
BRANCH_AT_DEPTH_2 = 2

# Demo builds run on the free model, where every call costs 20-30s. Depth 1 keeps
# an ordinary DIY question near 110s instead of ~140s. The capability is proven by
# the eval suite, which pins depth 2 regardless of profile.
#
# Measured, so the trade-off is a decision rather than a guess: on the
# then-pinned openai/gpt-oss-20b:free a propose call averaged 22.8s and a
# three-criterion critique 21.7s, so depth 2 cost roughly one extra propose
# plus one extra critique. Those numbers are HISTORICAL - that slug was
# withdrawn from the free tier on 2026-08-21 and the demo now runs on
# nvidia/nemotron-3-super-120b-a12b:free, which is roughly an order of
# magnitude faster. The RATIO is what this default rests on, and it holds;
# re-measure before quoting the absolute figures anywhere.
# Flip DEMO_DEPTH to 2 to make deep search the demo default.
DEMO_DEPTH = 1
FULL_DEPTH = 2


def default_depth() -> int:
    """Depth to search when the caller does not ask for a specific one."""
    return DEMO_DEPTH if config.is_demo_build() else FULL_DEPTH


def depth_for_complexity(complexity: str | None) -> int:
    """Search depth for a turn the Router labelled `complexity`.

    Depth policy lives here rather than in the Router: the Router's job is to
    label a turn, and what a label is worth in model calls is the search's
    business. That separation is what lets the depth trade-off be re-tuned — or
    the Router removed entirely — without touching the other.

    The mapping only ever moves depth *up* from the profile default, never down.
    A demo build asked a genuinely complex question spends the extra ~30s to
    compare refined options, which is the one case where it is clearly worth it;
    but "simple" does not buy a full build a cheaper answer, because a router
    miss must never make an answer worse than it would have been without a
    router at all. That asymmetry is the whole "advises, never overrides" rule
    expressed as one `max()`.
    """
    base = default_depth()
    return max(base, FULL_DEPTH) if complexity == "complex" else base


@dataclass
class AdvisorBudget:
    """Hard cap on what one advisory run may spend.

    Exceeding it degrades the search to whatever depth it reached and marks the
    result `truncated`, rather than either blocking indefinitely or silently
    returning a shallower answer as though it were the intended one.
    """
    max_llm_calls: int = 4          # the composing call is the caller's, not ours
    max_wall_seconds: float = 45.0
    max_nodes: int = 8
    calls_used: int = 0
    started_at: float = field(default_factory=time.monotonic)
    truncated_because: str | None = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def can_spend(self, nodes_so_far: int = 0) -> bool:
        if self.calls_used >= self.max_llm_calls:
            self.truncated_because = f"reached the {self.max_llm_calls}-call limit"
            return False
        if self.elapsed >= self.max_wall_seconds:
            self.truncated_because = f"reached the {self.max_wall_seconds:.0f}s time limit"
            return False
        if nodes_so_far >= self.max_nodes:
            self.truncated_because = f"reached the {self.max_nodes}-node limit"
            return False
        return True

    def spend(self) -> None:
        self.calls_used += 1


def search(
    *,
    propose,
    critique,
    evaluate_deterministic,
    expand=None,
    depth: int | None = None,
    budget: AdvisorBudget | None = None,
) -> dict:
    """Run the beam search and return the whole tree, not just the winner.

    Injected callables:
      propose()                -> [{"name","summary"}, ...]
      expand(nodes)            -> {parent_id: [{"name","summary"}, ...]}  (depth 2)
      critique(nodes)          -> [{"name", <critic criteria>, "why"}, ...]
      evaluate_deterministic(node) -> ({criterion: score}, {gate: bool})

    Returns {nodes, winner, depth_reached, llm_calls, truncated, truncated_because}.
    `winner` is None when every branch was pruned — a correct outcome meaning "no
    safe approach survived", which the caller must escalate rather than paper over.
    """
    depth = default_depth() if depth is None else depth
    budget = budget or AdvisorBudget()
    all_nodes: list[Node] = []

    # --- depth 1: propose strategies -------------------------------------------
    budget.spend()
    proposals = (propose() or [])[:BRANCH_AT_DEPTH_1]
    level = [
        Node(id=f"d1-{i}", name=str(p.get("name") or f"Option {i + 1}"),
             summary=str(p.get("summary") or ""), depth=1, order=i)
        for i, p in enumerate(proposals)
    ]
    if not level:
        return _result([], None, 0, budget)
    all_nodes += level

    level = _score_level(level, critique, evaluate_deterministic, budget)
    beam = prune(level, BEAM_WIDTH)
    depth_reached = 1

    # --- depth 2: expand the survivors ------------------------------------------
    if depth >= 2 and beam and expand is not None and budget.can_spend(len(all_nodes)):
        budget.spend()
        # `expand` takes the WHOLE beam and returns {parent_id: [children]}, so the
        # level costs one call rather than one per survivor. Calling it per parent
        # would charge the budget once and spend it k times, which is how a
        # "5-call" search quietly becomes a 7-call one.
        expansions = expand(beam) or {}
        children: list[Node] = []
        for parent in beam:
            for j, child in enumerate((expansions.get(parent.id) or [])[:BRANCH_AT_DEPTH_2]):
                children.append(Node(
                    id=f"{parent.id}-{j}",
                    name=str(child.get("name") or f"{parent.name} variant {j + 1}"),
                    summary=str(child.get("summary") or ""),
                    depth=2, parent_id=parent.id, order=len(children),
                ))
        if children:
            all_nodes += children
            children = _score_level(children, critique, evaluate_deterministic, budget)
            # A parent that produced children is superseded by them, not competing
            # with them: keeping both would let a vague strategy outrank its own
            # concrete execution purely because it was scored more generously.
            for parent in beam:
                if any(c.parent_id == parent.id for c in children):
                    parent.status = "superseded"
            prune(children, BEAM_WIDTH)
            depth_reached = 2

    winner = _select_final(all_nodes)
    return _result(all_nodes, winner, depth_reached, budget)


def _score_level(level, critique, evaluate_deterministic, budget) -> list[Node]:
    """Attach deterministic scores + gates, then the Critic's subjective scores."""
    for n in level:
        scores, gates = evaluate_deterministic(n)
        n.scores.update(scores or {})
        n.gates.update(gates or {})

    # Only nodes that survive the gates are worth an LLM call. Gated nodes are
    # already decided, and paying a model to rank a vetoed option is pure waste.
    ungated = [n for n in level if not n.gated]
    if ungated and budget.can_spend():
        budget.spend()
        by_name = {n.name.strip().lower(): n for n in ungated}
        for row in critique(ungated) or []:
            node = by_name.get(str(row.get("name", "")).strip().lower())
            if node is None:
                continue
            for key, value in row.items():
                if key in ("name", "why"):
                    continue
                try:
                    node.scores[key] = max(0.0, min(10.0, float(value)))
                except (TypeError, ValueError):
                    continue  # leave it unset; weighted_total treats it as neutral
            node.why = str(row.get("why", ""))[:200]

    for n in level:
        score_node(n)
    return level


def _select_final(all_nodes: list[Node]) -> Node | None:
    """Argmax over the deepest live level. Pure Python — no model involved.

    Prefers the deepest surviving nodes, because a concrete execution plan is a
    better answer than the strategy that generated it. `superseded` parents are
    excluded so a parent cannot beat its own refinement.
    """
    live = [n for n in all_nodes if n.status == "live"]
    if not live:
        return None
    deepest = max(n.depth for n in live)
    return min((n for n in live if n.depth == deepest), key=rank_key)


def _result(nodes, winner, depth_reached, budget) -> dict:
    return {
        "nodes": [n.as_dict() for n in nodes],
        "winner": winner.as_dict() if winner else None,
        "depth_reached": depth_reached,
        "llm_calls": budget.calls_used,
        "truncated": budget.truncated_because is not None,
        "truncated_because": budget.truncated_because,
        "strategy": (f"beam(b1={BRANCH_AT_DEPTH_1},b2={BRANCH_AT_DEPTH_2},"
                     f"k={BEAM_WIDTH},D={depth_reached})"),
    }


if __name__ == "__main__":
    # The entire control flow, exercised with stub callables. No LLM, no network.
    import json

    PROPOSALS = [
        {"name": "Adhesive rail", "summary": "Damage-free rail rated over 20 lb."},
        {"name": "Toggle bolt", "summary": "Toggle anchors in hollow drywall."},
        {"name": "Stud mount", "summary": "Screw directly into a stud."},
        {"name": "Hire a pro", "summary": "A handyman mounts it."},
    ]
    DETERMINISTIC = {
        "Adhesive rail": ({"safety_risk": 9, "permission_fit": 10, "cost": 9}, {}),
        "Toggle bolt": ({"safety_risk": 7, "permission_fit": 2, "cost": 8}, {}),
        # A renter drilling into a stud is a permanent modification -> gated.
        "Stud mount": ({"safety_risk": 8, "permission_fit": 1, "cost": 9},
                       {"occupant_not_permitted": True}),
        "Hire a pro": ({"safety_risk": 10, "permission_fit": 10, "cost": 2}, {}),
    }

    def det(node):
        for key, val in DETERMINISTIC.items():
            if node.name.startswith(key):
                return val
        return ({"safety_risk": 8, "permission_fit": 8, "cost": 6}, {})

    def critique(nodes):
        return [{"name": n.name, "suitability": 8, "reversibility": 7,
                 "effort_skill": 7, "why": "stub"} for n in nodes]

    def expand(nodes):
        return {n.id: [{"name": f"{n.name} — careful", "summary": "slower, safer"},
                       {"name": f"{n.name} — quick", "summary": "faster, rougher"}]
                for n in nodes}

    print("=== depth 2 ===")
    r = search(propose=lambda: PROPOSALS, critique=critique,
               evaluate_deterministic=det, expand=expand, depth=2)
    print(f"{r['strategy']}  llm_calls={r['llm_calls']}  truncated={r['truncated']}")
    for n in r["nodes"]:
        t = "-inf " if n["total"] == float("-inf") else f"{n['total']:5.2f}"
        print(f"  d{n['depth']}  {t}  {n['name']:<28} {n['status']:<10} {n['prune_reason'] or ''}")
    print("WINNER:", r["winner"]["name"])
    assert r["winner"]["depth"] == 2, "should select a depth-2 leaf"
    assert any(n["prune_reason"] and "role does not permit" in n["prune_reason"]
               for n in r["nodes"]), "the gated node should be pruned with a reason"

    print("\n=== depth 1 (demo default) ===")
    r1 = search(propose=lambda: PROPOSALS, critique=critique,
                evaluate_deterministic=det, expand=expand, depth=1)
    print(f"{r1['strategy']}  llm_calls={r1['llm_calls']}  winner={r1['winner']['name']}")
    assert r1["llm_calls"] < r["llm_calls"], "depth 1 must be cheaper"

    print("\n=== every branch gated -> no winner, escalate ===")
    r2 = search(propose=lambda: PROPOSALS, critique=critique,
                evaluate_deterministic=lambda n: ({}, {"high_risk_work": True}),
                expand=expand, depth=2)
    print("winner:", r2["winner"], "| llm_calls:", r2["llm_calls"])
    assert r2["winner"] is None, "all-gated must yield no winner"
    assert r2["llm_calls"] == 1, "must not pay a critic to rank vetoed options"

    print("\n=== budget exhaustion degrades, and says so ===")
    r3 = search(propose=lambda: PROPOSALS, critique=critique,
                evaluate_deterministic=det, expand=expand, depth=2,
                budget=AdvisorBudget(max_llm_calls=2))
    print(f"depth_reached={r3['depth_reached']}  truncated={r3['truncated']}"
          f"  because={r3['truncated_because']}")
    assert r3["truncated"] and r3["depth_reached"] == 1

    print("\nall control-flow assertions passed")
