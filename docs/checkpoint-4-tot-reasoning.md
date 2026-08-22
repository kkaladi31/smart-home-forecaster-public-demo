# Checkpoint 4 — Tree-of-Thought reasoning design

**Project:** Smart-Home Forecaster — an agentic assistant that warns a homeowner or
renter about weather hazards to their property and helps them understand what they
are permitted to do to their home.

*All data used in this project is synthetic or drawn from public government APIs.
The saved homes, their associations, contractors and utilities are invented.*

---

## 1. Is Tree-of-Thought appropriate for this agent? (Required)

**Yes — for exactly one part of the system, and it would actively hurt the rest.**

The system routes each question to a specialist. Three of those specialists must
*not* use Tree-of-Thought:

- **Weather and hazard assessment** is deterministic. Whether 18 °F with 15 mph wind
  constitutes a severe freeze risk is computed, not deliberated. Generating
  competing "thoughts" about it would manufacture disagreement where a single
  correct answer exists, and would reintroduce exactly the failure the deterministic
  assessors were built to remove — a language model judging a temperature.
- **Policy and rule questions** are retrieval problems with a cite-or-refuse gate.
  The rule is in the homeowner's documents or it is not. Branching would generate
  several *plausible* rules, and a plausible fabricated covenant is the single most
  dangerous output this system could produce, because it is indistinguishable from a
  real one.
- **Cost analysis** is arithmetic over live utility rates. Branching over arithmetic
  produces variance, not insight.

Tree-of-Thought **does** improve the fourth specialist — the **Advisor**, which
handles open-ended DIY, maintenance, repair and installation decisions. That is
where the problem genuinely has multiple valid solutions whose constraints interact,
and where a linear chain measurably fails.

The design decision, then, is not "use ToT" but **"use ToT precisely where the
answer is a choice, and refuse to use it where the answer is a fact."**

---

## 2. Where ToT reasoning is used in the architecture

### The workflow segment

The supervisor delegates to the Advisor via an `ask_advisor` tool call. Inside the
Advisor:

```
gather grounding  →  PROPOSE strategies  →  CRITIQUE  →  PRUNE
                  →  EXPAND survivors    →  CRITIQUE  →  SELECT (argmax)  →  compose
```

Grounding is collected first and in parallel — the home profile, jurisdiction-scoped
policy passages from the vector store, and web evidence — so that every candidate is
generated and scored against the same facts.

### Why a linear chain fails here

Take an ordinary question: *"How do I hang a 20 lb mirror on this wall?"* All three
of the failure modes appear at once.

**Premature commitment.** A linear chain names an approach in its opening sentence
and spends the remainder justifying it. Having opened with "locate a stud", it will
not later discover that the occupant is a renter whose lease forbids permanent
fixings. The constraint arrives *after* the commitment, and nothing in a chain goes
back. In tree search the commitment happens last: every option is generated before
any is scored.

**High branching.** There are four to six genuinely defensible approaches — stud
mount, toggle bolt, drywall anchor, adhesive rail, French cleat, hire a
professional. These are not rephrasings of one answer; they are different answers
that win under different conditions.

**Constraint complexity.** The constraints interact rather than filtering
independently. Tenure rules out permanent fixings; wall construction rules out
toggles; an association covenant may govern anything visible from the street; weight
interacts with all three. A chain applies constraints in whatever order they happen
to surface, and a reasoner whose conclusion depends on the order its constraints
arrived in is not reliable.

---

## 3. ToT structure

**A "thought"** is a candidate approach to the task at a level of specification —
not a step of reasoning, but a proposed way of getting the job done.

**A node** is a thought plus its evaluation: `{id, parent_id, depth, name, summary,
rubric scores, gate results, weighted total, status, prune_reason}`.

**A branch** is the edge from a strategy to one concrete way of executing it.

**Depth is specificity, not time** — this tree is not a chain of reasoning steps:

| Depth | Node represents | Example |
|---|---|---|
| 0 | The task plus its grounding | *20 lb mirror, renter, drywall, HOA community* |
| 1 | **Strategy** | *Toggle bolt* · *Adhesive rail* · *Stud mount* · *Hire a pro* |
| 2 | **Execution plan** | *Two ¼" toggles at 16" spacing, ½" pilot, 50 lb rated* |

### Branching factor and depth limit

| Parameter | Value | Rationale |
|---|---|---|
| Branching factor at depth 1 | **4** | Below 3 there is nothing to compare; above 4 a small model emits near-duplicates and the extra scoring is wasted. |
| **Beam width** | **2** | Deliberately *less than* the branching factor, so pruning is structural — two of four strategies are always discarded before any effort is spent refining them. |
| Branching factor at depth 2 | **2** | Two executions per surviving strategy is enough to expose a materials/effort trade-off. |
| **Depth limit** | **2** | Depth 1 is the current flat implementation, the thing being fixed. Depth 3 would branch on brand of anchor — a shopping list, not a decision. |
| Maximum nodes | **8** | 4 at depth 1, 4 at depth 2. |

### Termination and final output

The search terminates when depth 2 has been scored. There is deliberately **no
convergence criterion**: a bounded, predictable tree is worth more than an adaptive
one here, because every expansion costs an LLM call while a person waits.

The final output is the **argmax leaf**, chosen in code, then composed into prose by
a single writing call that receives the already-chosen node and cannot substitute
another.

The search also terminates early, and correctly, when **every branch is pruned** —
see failure conditions below.

---

## 4. Search and evaluation strategy

### Primary search strategy: **beam search**

Beam search is BFS plus pruning, which is precisely the property this problem wants:
compare fairly at each level, then stop paying for what has already lost.

The three alternatives were considered and rejected for reasons specific to this
domain:

- **BFS** would score all twelve nodes, including refinements of strategies already
  known at depth 1 to violate a hard constraint. It would expand and cost a renter's
  "drill into the stud" branch before discarding it. Affordable, but it spends model
  calls to rediscover something already established.
- **DFS** drives one strategy to full specification before considering the second.
  That *is* premature commitment — the failure this design exists to remove — and it
  cannot produce the "options I compared" table that is a deliverable of the feature.
- **Monte-Carlo-style sampling** requires a cheap simulator to roll out against.
  Here there is no environment to simulate: one cannot "play out" hanging a mirror
  and observe a reward, and every rollout is a paid LLM call. The sample budget MCTS
  needs to beat a simple heuristic is unaffordable at this scale.

### Evaluation criteria and scoring rubric

The central decision is that **hard constraints are gates, not weights.** A weighted
average allows a cheap, easy, fast option to outvote *"the lease forbids this"* or
*"this is a gas line"*. Averaging a veto is how a scoring system produces confident,
catastrophic recommendations.

**Hard gates — veto, applied in Python, never averaged.** A gated node scores `-inf`
and cannot enter the beam regardless of any other criterion:

| Gate | Determined by |
|---|---|
| Occupant not permitted (renter + permanent modification) | persona + safety module |
| Explicitly prohibited by a retrieved rule | a cited passage from the vector store |
| High-risk work (gas, service panel, structural, roof, asbestos) | deterministic regex classifier |

The third gate matters for consistency: the system already **refuses to describe**
high-risk work. Allowing the Advisor to rank it would route around an existing
guardrail.

**Weighted rubric — criteria that genuinely trade off.** Each scored 0–10, weights
summing to 1.0:

| Criterion | Weight | Evaluated by |
|---|---|---|
| Safety risk | **0.30** | Deterministic hazard table |
| Permission / policy fit | **0.25** | RAG-grounded — must cite a passage, else neutral |
| Task suitability | 0.20 | Critic agent |
| Reversibility | 0.10 | Critic agent |
| Cost | 0.08 | Deterministic — materials + contractor rate |
| Effort / skill | 0.07 | Critic agent |

**63% of the score comes from criteria the model cannot author** (safety 0.30,
policy fit 0.25, cost 0.08). The Critic's judgement is confined to the genuinely
subjective 37% — suitability, reversibility and effort.

### Who performs the evaluation — a combination

- **A critic agent** — a separate module with its own system prompt and its own LLM
  call. It scores each node per criterion with a one-line justification. It is never
  shown the previous winner, and **it never selects**.
- **Heuristic checks** — the safety and cost criteria, computed in Python.
- **Tool calls** — policy fit is grounded in retrieval; cost calls the contractor
  directory.

Separating *scoring* from *selecting* is what makes the outcome auditable: the critic
produces numbers, the controller applies a fixed published policy to them.

### Pruning thresholds and failure conditions

Applied in order, in code:

1. **Gate veto** — any hard gate fires → pruned, `-inf`.
2. **Absolute floor** — weighted total **< 4.0 / 10** → pruned. Absolute rather than
   relative because "keep the top half" always keeps something, even when everything
   is bad.
3. **Relative floor** — total **< best − 3.0** → pruned. A dominated branch is not
   worth an expansion call.
4. **Beam width** — of the survivors, keep the top **2**.

**Failure condition.** If every branch is pruned, the system must **not** relax a
threshold and return the least-bad option. It returns *"no safe DIY approach found
for this job"*, names the gate that fired, and escalates to the professional
referral path. **An empty tree is a valid and correct output** — and one the
evaluation suite asserts on directly.

### Tie resolution

Deterministic and ordered, so that a re-run reproduces the decision:

1. Fewer gates near-missed → 2. Higher safety sub-score → 3. Higher policy-fit
sub-score (grounded beats ungrounded) → 4. Lower cost → 5. Stable proposal order.

No voting and no additional model call. Ties are resolved by **selection policy**,
not by judgement, because a tie-break that consults a model is a tie-break that
cannot be reproduced or explained.

### Constraint by compute, latency and cost

A budget object caps each run at **5 LLM calls, 45 seconds, 8 nodes**:

```
1. propose    4 strategies
2. critique   all 4, batched into ONE call
3. expand     2 each for the 2 survivors, batched
4. critique   all 4 leaves, batched
5. compose    write up the argmax leaf
```

Batching every critique into a single call is what holds this at 5 rather than 13.
On exhaustion the search degrades to depth 1 and the result is flagged
`budget_truncated`, surfaced in the trace rather than hidden.

This project also runs on a free model tier for demonstration, where the demo
configuration defaults to depth 1 — roughly three calls. A ninety-second turn ruins
a live demonstration, and the reasoning *architecture* is what is being assessed,
not its wall-clock depth.

---

## 5. ToT roles mapped to implementation tools

| Role | Tool | Why this tool |
|---|---|---|
| **Thought generator** | **LangChain** `ChatOpenAI`, called from the beam controller | Generation is one structured-output call. It needs no agent framework; a framework here would wrap a role abstraction around a single prompt. |
| **Critic / evaluator** | **LangChain** for the critic's own call, **plus** in-process Python guardrails and the RAG grounding gate | The hybrid is the design. A pure-LLM critic can be argued out of a safety score; a pure-heuristic critic cannot judge suitability. Each mechanism does what it is reliable at. |
| **Decision maker / controller** | **Plain Python** — a beam module, no framework | Selection must be deterministic and unit-testable without a model. This is the role most often handed to an LLM and the one that benefits least from one. |
| **Memory / state manager** | **LangGraph** — `SqliteSaver` checkpointer for conversation state, plus a Chroma vector store for semantic and episodic memory, plus the run configuration carrying home and persona identity | Branch state lives in the controller's node list within one run; anything crossing turns is LangGraph's checkpointer. |

### Frameworks considered and rejected

**CrewAI.** Its value is role definition and task routing among agents. This project
already has role separation (separate modules, separate system prompts) and already
has routing (a supervisor). Adopting CrewAI would mean maintaining a second agent
abstraction over the same functions, and its delegation model is less explicit than
the argmax policy this design specifically wants to keep in readable Python.

**MCP.** MCP earns its place when tools sit behind a process boundary or are shared
across clients. Every tool here is in-process Python over local files and public
HTTP APIs. Adding MCP would introduce a transport hop, a serialisation boundary and
an additional failure mode for no capability gain. It becomes the right answer the
moment these tools are shared with a second application — which is a reason to keep
the tool layer clean, not a reason to adopt it now.

---

## 6. One risk and one mitigation

### Risk: weak evaluation signals prune the best branch

This is the risk that matters, because it is **silent**. A small model produces noisy
0–10 judgements. A genuinely good option scored 5 instead of 8 is discarded at the
beam boundary; the user never sees it; the answer that *is* returned looks entirely
reasonable. There is no exception, no error, and nothing downstream can detect the
loss. Unmitigated, beam search degrades into an expensive way of choosing
arbitrarily — and it degrades *invisibly*, which is worse than degrading loudly.

### Mitigation: put the load-bearing criteria beyond the model's reach, and make every prune inspectable

Three concrete measures:

1. **63% of the rubric weight is not authored by the model** (safety 0.30, policy fit 0.25, cost 0.08). The
   criteria that decide whether an option is *acceptable* come from regex guardrails
   and retrieved passages. Critic noise therefore perturbs the ranking *among
   acceptable options* — where being wrong costs a slightly worse recommendation —
   rather than deciding acceptability itself.
2. **Gates are structurally separate from scores.** A vetoed branch is not "scored
   low"; it is removed for a stated reason. No amount of critic noise can promote it,
   and none can quietly demote a safe option into the same bucket.
3. **Every pruned node is returned with its prune reason**, and the interface renders
   pruned branches struck through beside the winner. This converts the silent failure
   into an observable one: a wrongly-pruned branch becomes visible to the user and
   assertable by the evaluation suite, which already tests that a renter's drilling
   branch is pruned *and that the system says why*.

Supporting measure: the absolute floor of 4.0 is deliberately permissive. It exists
to catch genuinely bad branches, not to act as a second beam width — the beam does
the narrowing, and the floor only catches the case where everything is bad.

---

## Addendum — implementation status as of 2026-08-17

*Added after submission. The design above is unchanged; this records what was
subsequently built and what measurement revealed, so the document does not
describe intent where evidence now exists.*

**Everything above is implemented and verified.** `agents/beam.py` runs the
search, `agents/rubric.py` holds the weights, gates and pruning rules, and
`agents/critic.py` scores. The whole loop — expansion, gating, pruning,
tie-breaking, selection, budget exhaustion and the all-pruned path — is testable
with **no model, no network and no cost**: `python -m agents.beam` exercises it
with injected callables.

**Parameters as shipped:** branch 4 then 2, beam width 2, depth ≤ 2, at most 4
model calls inside the budget (the composing call belongs to the caller), 45 s
wall clock, 8 nodes. Demo builds default to depth 1 for latency on the free
model; the evaluation pins depth 2 regardless of profile, so the capability is
always exercised.

**Selection is `argmax` in Python**, as designed. This closed a real defect: a
third model call previously re-picked the winner, so the option scores displayed
in the UI need not have corresponded to the recommendation.

### Three over-gating defects found after submission

Each removed the **correct** answer, and none was visible from the final text
alone — which is precisely why the tree is rendered with reasons.

1. A **single shared word** pruned all four branches, including the adhesive rail
   that was right for a renter. The gate now requires more than one shared term.
2. **Hiring a professional was hard-gated**, and the model then asserted that
   *"the home documents prohibit hiring a professional"* — a claim the gate had
   invented and no document contained.
3. A **clearly reversible option was marked restricted**, handing the
   recommendation to a paid handyman for a task a renter could do with adhesive
   strips.

The generalisable lesson: **a gate that fires too eagerly does not fail safe.** It
fails toward whatever survives, and what survives is often more expensive, less
reversible, or simply wrong.

### Measured

- **42/42** evaluation cases pass on the free model, including a case asserting
  that a renter's drilling branch is pruned *and that the reason is stated*.
- Depth 2 costs ≈ 140 s versus ≈ 110 s at depth 1 on `openai/gpt-oss-20b:free`,
  the model pinned when this was measured (withdrawn from OpenRouter's free tier
  on 2026-08-21; the demo now runs on `nvidia/nemotron-3-super-120b-a12b:free`,
  which is roughly an order of magnitude faster — the ratio holds, the absolute
  seconds do not) —
  materially cheaper than the ~190 s originally estimated.
- A three-criterion critique averages 21.7 s against 60.7 s for six, which is why
  the Critic scores only the subjective criteria and the deterministic ones are
  computed in code.
