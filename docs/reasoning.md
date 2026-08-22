# Reasoning architecture: Tree-of-Thought by beam search

> **Status.** This document specifies the target design for `agents/advisor.py`.
> The current implementation is the earlier depth-1 form described under
> [What changes from the current implementation](#what-changes-from-the-current-implementation).
> Anything below marked *designed* is not yet in the code.

## 1. Where Tree-of-Thought is used, and where it is not

The system is a supervisor routing to specialists. Only **one** of them reasons
over a tree: the **Advisor**, which handles open-ended DIY, maintenance, repair
and installation decisions.

Everywhere else, ToT would be wrong and is deliberately not used:

| Path | Reasoning | Why not ToT |
|---|---|---|
| Weather / hazards | Deterministic pipeline, no LLM | There is one correct answer. A tree over "is 18°F freezing" invents disagreement where none exists. |
| Policy / RAG | Single retrieval + cite-or-refuse | The answer is in a document or it is not. Branching would generate plausible rules, which is the exact failure the grounding gate exists to prevent. |
| Cost | Deterministic arithmetic, LLM writes up fixed figures | Branching over arithmetic produces variance, not insight. |
| Advisor | **Tree-of-Thought, beam search** | Genuinely several valid approaches, with constraints that interact. |

### Why a linear chain fails specifically here

Three of the failure modes the module names all show up in one ordinary question —
*"How do I hang a 20 lb mirror on this wall?"*:

- **Premature commitment.** A linear chain names an approach in its first sentence
  and spends the rest of the answer justifying it. If it opens with "use a stud
  finder", it will not later discover that the occupant is a renter whose lease
  forbids drilling. The constraint arrives *after* the commitment, and nothing in a
  chain revisits it.
- **High branching.** There are genuinely four to six defensible approaches — stud
  mount, toggle bolt, drywall anchor, adhesive rail, French cleat, hire a pro — and
  which one wins depends on weight, wall construction, tenure and reversibility.
  These are not variations on one answer; they are different answers.
- **Constraint complexity.** The constraints are not independent. Renter status
  rules out permanent fixings; a masonry wall rules out toggles; an HOA covenant may
  govern anything visible from the street; weight interacts with all of it. A chain
  evaluates constraints in whatever order they surface, and the order changes the
  conclusion — which is the definition of an unreliable reasoner.

Tree search fixes this by making the commitment *last* instead of first: options are
generated before any of them is scored, and the constraints are applied to all of
them uniformly.

## 2. The tree

**A thought** is a candidate approach to the task at a level of specification —
not a sentence of reasoning, but a proposed way of accomplishing the job.

**A node** is one thought plus its evaluation:

```
{ id, parent_id, depth, name, summary,
  rubric: {criterion: score, ...}, gates: {...},
  score: float, status: "live" | "pruned", prune_reason: str | None }
```

**A branch** is the edge from a strategy to one concrete way of executing it.
**Depth** is how specified the thought is:

| Depth | What a node is | Example |
|---|---|---|
| 0 | The task plus its grounding (home profile, retrieved policy, evidence) | *"Hang a 20 lb mirror, renter, drywall, Lakeshore Commons"* |
| 1 | **Strategy** — a class of approach | *"Toggle-bolt into drywall"*, *"Adhesive rail system"*, *"Hire a handyman"* |
| 2 | **Execution plan** — materials, sequence, checkpoints | *"Two 1/4" toggles at 16" spacing, pilot 1/2", verify 50 lb rating"* |

The tree is **not** a chain of reasoning steps. Depth is specificity, not time.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Branching factor, depth 1 | **b₁ = 4** | Below 3 there is nothing to compare; above 4 a small model starts emitting near-duplicates and the extra scoring is wasted. |
| Beam width | **k = 2** | Strictly less than b₁, so pruning is structural rather than incidental — two of four strategies are always discarded before any effort is spent refining them. |
| Branching factor, depth 2 | **b₂ = 2** | Two concrete executions per surviving strategy is enough to expose a materials/effort trade-off. |
| Depth limit | **D = 2** | D=1 is the current flat implementation — the thing being fixed. D=3 would branch on brand of anchor, which is a shopping list, not a decision. |
| Max nodes | **8** | 4 at depth 1 + 4 at depth 2. |

**Termination.** The search stops when depth 2 is scored — there is no convergence
criterion, deliberately. A bounded, predictable tree is worth more here than an
adaptive one, because the cost of each expansion is an LLM call and the user is
waiting. The final output is the argmax leaf, composed into prose by one last call.

## 3. Search strategy: beam search

Beam search, chosen over the three alternatives for reasons specific to this problem:

| Strategy | Why not |
|---|---|
| **BFS** | Scores all 12 nodes including refinements of strategies already known to violate a hard constraint. A renter's "drill into the stud" branch would be expanded and costed before being discarded. Affordable, but it spends LLM calls to learn something already known at depth 1. |
| **DFS** | Commits to one strategy and drives it to full specification before considering the second. That is *premature commitment* — the exact failure this design exists to remove — and it cannot produce the "options I compared" table that is the deliverable. |
| **Monte-Carlo sampling** | Needs a cheap simulator to roll out from. Here every rollout is an LLM call and there is no environment to simulate against — no way to "play out" hanging a mirror and observe a reward. The sample budget MCTS needs to beat a heuristic is unaffordable. |
| **Beam** ✅ | BFS's uniform comparison at each level, minus the branches that cannot win. Bounded cost, predictable latency, and it produces a comparison table as a by-product. |

Beam is BFS plus pruning. That is precisely the property wanted: compare fairly at
each level, then stop paying for what has already lost.

## 4. Evaluation

### Two mechanisms, deliberately separate

The most important design decision here is that **hard constraints are gates, not
weights**. A weighted average lets a cheap, easy, fast option outvote "the lease
forbids this" or "this is a gas line". Averaging a veto is how a scoring system
produces confident, catastrophic answers.

**Hard gates — veto, evaluated in Python, never averaged.** A gated node is scored
`-inf` and cannot enter the beam regardless of every other criterion:

| Gate | Source | Rationale |
|---|---|---|
| Occupant not permitted | persona + `tools/safety.py` | A renter cannot make permanent modifications. |
| Prohibited by a retrieved rule | `search_home_policies` passage | An explicit CC&R or lease prohibition is not a preference. |
| High-risk work | `tools/safety.check_high_risk()` | Gas, service panel, structural, roof, asbestos. The system already refuses to describe this work; it must not rank it either. |

**Weighted rubric — the criteria that genuinely trade off.** Each scored 0–10;
weights sum to 1.0:

| Criterion | Weight | Who scores it |
|---|---|---|
| Safety risk | **0.30** | **Deterministic** — hazard table + `check_high_risk`. The LLM cannot raise a safety score. |
| Permission / policy fit | **0.25** | **RAG-grounded** — must cite a retrieved passage, else scores neutral. Never inferred. |
| Task suitability | 0.20 | Critic LLM — load, wall type, climate zone |
| Reversibility | 0.10 | Critic LLM — damage on removal |
| Cost | 0.08 | **Deterministic** — materials estimate + pro hourly rate |
| Effort / skill | 0.07 | Critic LLM |

Note the weighting: **63% of the score is not authored by the model** (safety 0.30,
policy fit 0.25, cost 0.08). The Critic's judgement is confined to the genuinely
subjective 37%.

This is also why the Critic is asked for only three criteria rather than six.
Measured on `openai/gpt-oss-20b:free`, the six-criterion form took **60.7s** per
call against **21.7s** for the three-criterion form — a 2.8× difference that
decides whether a depth-2 search is affordable at all.

Those two numbers are kept as the pair they were measured as, on the model pinned
at the time, which OpenRouter withdrew from its free tier on 2026-08-21. The
six-criterion form was deleted when the split was made, so the comparison cannot
be re-run without rebuilding the thing it argued against. What *is* current: on
`nvidia/nemotron-3-super-120b-a12b:free` the surviving three-criterion critique
measures **14.4s** (median of 5), a propose call **4.6s**, and a full depth-2 run
**53.8s** against **25.9s** at depth 1. The ratio the design rests on survived the
model change; the absolute seconds did not.

### Who evaluates — a combination, by design

- **Critic agent** (`agents/critic.py`, *designed*) — its own system prompt and LLM
  call. Scores each node per criterion with a one-line justification. It is shown
  the nodes but **never the previous winner**, and it **never selects**.
- **Heuristic checks** — the safety and cost criteria, in Python.
- **Tool calls** — the policy-fit criterion is grounded in RAG retrieval; the cost
  criterion calls the contractor lookup.

Separating *scoring* from *selecting* is what makes the result auditable: the Critic
produces numbers, the controller applies a fixed policy to them.

### Pruning thresholds and failure conditions

Applied in order, in code:

1. **Gate veto** — any hard gate fires → pruned, `score = -inf`.
2. **Absolute floor** — weighted total `< 4.0 / 10` → pruned. Absolute rather than
   relative, because "keep the top half" always keeps something, even when
   everything is bad.
3. **Relative floor** — total `< best − 3.0` → pruned. A dominated branch is never
   worth an expansion call.
4. **Beam width** — of whatever survives, keep the top `k = 2`.

**Failure condition — the important one.** If *every* branch is pruned, the system
must **not** relax a threshold and return the least-bad option. It returns "no safe
DIY approach found for this job", states which gate fired, and escalates to the
professional referral path. An empty tree is a valid, correct output.

### Tie-breaking

Deterministic and ordered, so a re-run reproduces the decision:

1. Fewer gates near-missed
2. Higher safety sub-score
3. Higher policy-fit sub-score (grounded beats ungrounded)
4. Lower cost
5. Stable proposal order

No voting, no second LLM call, no sampling. Ties are broken by policy, not judgement.

### Selection

`beam.select(nodes) → node` is **pure-Python argmax** over the weighted total, then
the tie-break ladder. The composition call receives the *already chosen* node and
writes it up; it cannot substitute a different one.

This closes a real defect in the current implementation, where a third LLM call is
handed all the evaluations and asked to "choose the best (or top two if close)" —
so the scores the UI displays may not correspond to the recommendation, with nothing
detecting the mismatch.

### Compute, latency and cost constraints

`AdvisorBudget(max_llm_calls=5, max_wall_seconds=45, max_nodes=8)`.

Worst case is exactly **5 LLM calls**:

```
1. propose      b₁ = 4 strategies
2. critique     all 4, batched in one call
3. expand       b₂ = 2 each for the k = 2 survivors, batched
4. critique     all 4 leaves, batched
5. compose      write up the argmax leaf
```

Batching every critique into a single call is what keeps this at 5 rather than 13.
On exhaustion the search degrades to D = 1 and the result is flagged
`budget_truncated: true`, surfaced in the trace rather than hidden.

**The demo profile defaults to D = 1** (≈3 calls, today's cost) unless deep
reasoning is requested. A 90-second advisor turn ruins a recorded demonstration,
and the architecture is what is being assessed, not the wall-clock depth.

### The cheapest search is the one that is never run

`AdvisorBudget` bounds a search that has a decision to make. One class of
question does not: if `tools.safety` classifies the *question* as high-risk work,
every do-it-yourself branch is gated by definition, and the only survivor is
"hire a professional" — which is what the hazard table already said, in well
under a millisecond, before a single token was generated.

So that case never enters the beam. `run_advisor` returns the referral directly:
**0 LLM calls, no web fetch, an empty tree, and a strategy string that says the
search was skipped** rather than a manufactured tree of branches nobody proposed.

Two measurements motivated this. First, cost: the pre-existing path spent a live
web search and up to five model calls to rediscover a fixed verdict — on a
free-tier model, minutes of a user watching a spinner *after* the guardrail had
visibly fired. Second, and more seriously, correctness. The gate used to read
only the node's own text, and the node's text is the model's paraphrase of an
approach, not the user's question. Asked "how do I replace my breaker box
myself", a model proposes branches like *"Install a subpanel instead"* and
*"Swap individual breakers only"* — both unmistakably service-panel work, neither
matching a phrase in the hazard table. Measured: both scored 8.0 on safety and
entered the beam, on a question the orchestrator had already refused to answer.
The refusal and the ranking disagreed about the same job.

The question's verdict is therefore authoritative and the node's can only add to
it (`evaluate_deterministic(..., question_high_risk=...)`). A branch cannot
escape a gate by being described in safer words than the thing it is a branch of.
This is the same escalation ladder the orchestrator already climbs — an emergency
bypasses the model entirely, an ordinary question gets the full loop — with
high-risk work correctly placed between the two instead of lumped in with
"ordinary".

## 5. ToT roles mapped to implementation

| Role | Implementation | Why |
|---|---|---|
| **Thought generator** | LangChain `ChatOpenAI` via `agents/llm.py`, called from the beam controller | Generation is a single structured-output call. It needs no agent framework — a framework here would add a role abstraction over one prompt. |
| **Critic / evaluator** | `agents/critic.py` (*designed*) — separate module, own system prompt, own call — **plus** `tools/safety.py` and the RAG grounding gate | The hybrid is the point. A pure-LLM critic can be argued out of a safety score; a pure-heuristic critic cannot judge suitability. Each does what it is reliable at. |
| **Decision maker / controller** | `agents/beam.py` — plain Python | Selection must be deterministic and unit-testable without an LLM. This is the role most often handed to a model, and it is the one that least benefits from one. |
| **Memory / state manager** | LangGraph `SqliteSaver` checkpointer (conversation), `memory/episodic.py` (past runs), `memory/rag_store.py` (semantic), and the run config carrying `home_id`/`persona` | Branch state lives in the controller's node list within a single run; cross-turn state is LangGraph's. |

### Framework choices, including the rejections

- **LangChain / LangGraph — used.** LangGraph supplies the supervisor loop, tool
  binding, the SQLite checkpointer and the callback interface the telemetry hangs
  off. The beam controller is a plain function invoked as a tool, not a graph node,
  because the tree is bounded and internal — expressing four sibling nodes as
  graph state would add ceremony without adding capability.
- **CrewAI — considered and rejected.** Its value is role definition and task
  routing between agents. This project already has role separation (separate
  modules, separate prompts) and already has routing (the supervisor). Adding CrewAI
  would mean maintaining a second agent abstraction over the same functions, and its
  delegation model is less explicit than the argmax policy that this design
  specifically wants to keep in Python.
- **MCP — considered and rejected.** MCP earns its keep when tools live behind a
  process boundary or are shared across clients. Every tool here is in-process
  Python operating on local files and public HTTP APIs. Adding MCP would introduce a
  transport hop, a serialisation boundary and a second failure mode for no
  capability gain. It would become the right answer the moment these tools were
  shared with another application.

## 6. Risk and mitigation

**Risk: weak evaluation signals prune the best branch.** This is the failure mode
that matters, because it is silent. A small model produces noisy 0–10 judgements; a
genuinely good option scored 5 instead of 8 is discarded at the beam boundary, the
user never sees it, and the answer that *is* returned looks entirely reasonable.
There is no error, no exception, and nothing downstream can detect it. Left
unmitigated, beam search degrades into an expensive way of picking arbitrarily.

**Mitigation: put the load-bearing criteria beyond the model's reach, and make
pruning inspectable.** Concretely:

1. **63% of the weight is not authored by the model** (safety 0.30, policy fit 0.25, cost 0.08). The
   criteria that decide whether a branch is *acceptable* come from regex guardrails
   and retrieved passages, so evaluation noise moves the ranking among acceptable
   options — where the cost of being wrong is a slightly worse recommendation —
   rather than deciding acceptability itself.
2. **Gates are separate from scores.** A vetoed branch is not "scored low"; it is
   removed for a stated reason. No amount of critic noise can promote it, and no
   amount can silently demote a safe one to the same place.
3. **Every pruned node is returned, with its reason.** `run_advisor` returns the
   full tree including `status: "pruned"` and `prune_reason`. The UI renders them
   struck through. A wrongly-pruned branch becomes visible rather than invisible,
   which converts a silent failure into an observable one — the eval suite asserts
   on it (a renter question must prune the drilling branch *and say so*).

Secondary mitigation: the absolute floor of 4.0 is tuned to be permissive. It exists
to catch genuinely bad branches, not to be a second beam width. The beam does the
narrowing; the floor only catches the case where everything is bad.

## What changed — before and after

**This has been implemented.** The "After" column is the code as it stands
(`agents/{advisor,beam,rubric,critic}.py`); the "Before" column is what
`agents/advisor.py` was when this design was written — a genuine multi-step
Tree-of-Thought, three separate LLM calls, gather → branch → evaluate → select,
but depth-1 and unpruned. The comparison is kept because the *reasons* for each
change are the substance of the design, and because one row of it describes a
real defect rather than merely a limitation: a third LLM call re-picked the
winner in prose, so the option scores the UI displayed need not have matched the
recommendation the user was given.

| Aspect | Before | After (implemented) |
|---|---|---|
| Structure | Flat `list[dict]`, depth 1 | Tree, depth 2, parent/child |
| Branching | 3–4, truncated to 4 | b₁ = 4, b₂ = 2 |
| Pruning | **None** | 4 rules: gates, absolute floor, relative floor, beam width |
| Scoring | One unweighted integer 1–10, one batched call | 6 weighted criteria, 63% not model-authored |
| Evaluator | Same model, inline prompt | Separate Critic module + heuristics + tool calls |
| Selection | **A third LLM call re-picks the winner in prose** | Pure-Python argmax + ordered tie-break |
| Ties | Stable sort only, undocumented | Explicit 5-step policy |
| Budget | Unbounded (always 3 calls) | `AdvisorBudget`, 5 calls, 45 s, 8 nodes |
| Failure | Falls back to neutral score 5 for all | Empty tree → escalate to a professional |
| Observability | Sub-agent calls untraced | Tracer attached in `agents/llm.py` |

Two of these are correctness defects rather than missing features: the
select-step mismatch (displayed scores need not match the recommendation) and the
neutral-5 fallback (a JSON parse failure silently defeats ranking entirely).
