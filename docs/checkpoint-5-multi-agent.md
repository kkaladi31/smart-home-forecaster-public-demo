# Checkpoint 5 — Multi-agent architecture design

**Project:** Smart-Home Forecaster — an agentic assistant that warns a homeowner or
renter about weather hazards to their property and helps them understand what they
are permitted to do to their home.

*All data is synthetic or from public government APIs. Saved homes, associations,
contractors and utilities are invented.*

---

## The problem, and why one agent is not enough

A single question from a homeowner routinely spans four incompatible kinds of work.
*"It's going to freeze this week — can I wrap my outdoor spigots myself, and what
would a plumber cost?"* requires, at once:

1. a **live hazard assessment** that must be numerically correct,
2. a **rule lookup** that must be grounded in that specific home's documents and
   must refuse rather than guess,
3. an **open-ended decision** among several valid approaches under interacting
   constraints,
4. a **cost estimate** from live utility and contractor data.

These have opposite failure characteristics. The hazard assessment must never
improvise — a language model judging a temperature is precisely the bug the
deterministic assessors exist to prevent. The rule lookup must refuse when
ungrounded, because a plausible fabricated covenant is indistinguishable from a real
one. The DIY decision must *deliberately* generate alternatives. The cost estimate
must not invent numbers.

A single agent cannot hold four contradictory dispositions at once. Attempting it in
one prompt produced the failure that motivated this architecture: the model, told to
be helpful and thorough, would answer a cost question in prose and lose the dollar
figures entirely, or judge a forecast itself instead of calling the assessor.

**The organising principle is therefore: one agent per distinct failure mode.** Each
agent owns a way the system can be wrong, and is built with whatever disposition —
generative, deterministic, or refusing — makes that particular failure least likely.

---

## How many agents, and how that number was determined

**Seven**, plus a deterministic tool layer that is deliberately *not* agents.

The count comes from enumerating distinct failure modes, not from decomposing the
workflow into steps. A step only becomes an agent when it can be wrong in a way no
other component can catch:

| # | Agent | The failure it owns |
|---|---|---|
| 1 | **Supervisor / Orchestrator** | the wrong specialist is consulted |
| 2 | **Router** *(deterministic)* | the intent, complexity or risk of the turn is misread |
| 3 | **Researcher** | external evidence is missing, stale, or untrustworthy |
| 4 | **Advisor** (ToT controller) | the wrong option is chosen |
| 5 | **Critic** | an option is *evaluated* wrongly |
| 6 | **Cost** | the numbers are wrong |
| 7 | **Pro Finder** | an unlicensed or unsuitable professional is recommended |

### What is deliberately *not* an agent

Weather assessment, policy retrieval and safety screening are **tools**, not agents,
and describing them otherwise would inflate the count dishonestly. Each is fully
deterministic — regex guardrails, a scored vector search with a grounding gate,
arithmetic over a forecast. Giving them agency would mean giving a model discretion
over exactly the decisions that must not be discretionary. Safety in particular runs
*outside* the agent boundary entirely: an emergency phrase bypasses the language
model altogether rather than being routed to it.

### Diminishing returns — the agent that was rejected

An eighth agent, a dedicated **Writer** to compose final prose, was considered and
rejected. It would add a full round trip of latency and a serialisation boundary
while isolating no new failure mode: the Supervisor already composes, and a
composition error is not a distinct way of being wrong — it is the Supervisor being
wrong. Adding it would have grown the diagram without growing the system's
reliability, which is the definition of the returns having diminished.

The Router, by contrast, earns its place despite being trivial, because "misread the
turn" is a failure that occurs *before* any specialist is consulted and that no
specialist can detect from inside its own task.

---

## Roles and responsibilities

**Supervisor / Orchestrator** — owns the conversation and the user-facing answer.
Routes each turn to one or more specialists, combines their structured returns into
a single response, and holds conversation memory. It is the only component that
talks to the user.

**Router** — a deterministic pre-model node. Labels each turn with intent,
complexity, and whether it touches high-risk work, using a keyword table plus
embedding similarity against intent exemplars. It **advises and never overrides**, so
a misclassification degrades to the Supervisor's own judgement rather than
misdirecting the turn. It is deterministic on purpose: it runs on every single turn,
so it must be free and instant, and "the router chose wrong" has to be debuggable
from a log rather than re-derived from a model.

**Researcher** — plans two to four targeted sub-queries, fans out across search
providers in parallel, fetches and extracts page content, deduplicates, reranks with
a cross-encoder against a domain-authority table, and returns a bounded evidence
pack with citations. It also screens fetched content for prompt injection, and
fetched text is structurally confined to user-role messages so it can never be read
as instruction.

**Advisor** — the Tree-of-Thought controller. Runs a bounded beam search over
candidate approaches (branching factor 4, beam width 2, depth 2) and returns the
full tree, including pruned branches and their prune reasons. Detailed in
`docs/reasoning.md` and Checkpoint 4.

**Critic** — scores each candidate against a weighted six-criterion rubric with a
one-line justification per criterion. It is never shown the previous winner, and
**it never selects** — selection is a pure-Python argmax in the Advisor. Separating
scoring from selecting is what makes the decision auditable and reproducible.

**Cost** — gathers live state energy prices and the home's utility records, computes
itemised savings deterministically, and uses a language model only to write up
figures it did not compute. Its prompt forbids introducing new numbers.

**Pro Finder** — matches a task to licensed professionals. **License status is a
gate, not a ranking signal**: a contractor whose registration is not active is
withheld with a stated reason rather than ranked lower.

---

## Coordination strategy: hybrid

Three topologies, each chosen for the sub-problem it fits:

**A supervisor star at the top.** The Supervisor delegates to specialists and
combines results. Chosen because delegation here is genuinely hierarchical — the
specialists have no reason to know about one another, and a star keeps the execution
trace linear and readable, which is what the interface renders and the evaluation
suite asserts against.

**A cyclic sub-graph inside the Advisor.** Propose → critique → prune → expand →
critique → select is genuinely iterative: the second expansion depends on the first
critique. This is the only cycle in the system, and its iteration count is a
**constant, not a convergence condition** — an unbounded critic loop is how a turn
becomes a four-minute turn.

**Parallel fan-out inside the Researcher.** Independent sub-queries against
independent providers, gathered with per-provider timeouts so one slow source
degrades the evidence pack rather than the answer.

One coordination decision is a safety control rather than a performance one: the
active home and the occupant's role travel on the **run configuration**, never as
model-chosen arguments. A model that selects its own jurisdiction filter will
sometimes select the wrong one, and a wrong-jurisdiction answer is indistinguishable
from a right one. The interface knows which home the user is looking at, so it passes
that down and the model cannot get it wrong.

---

## Communication

**One-way, typed dictionaries by default.** A specialist receives a task and returns
a structured result. No negotiation, no shared scratchpad. This is cheap, and it
keeps every hand-off inspectable in the trace.

**Two-way in exactly two places**, both bounded:

- **Advisor ↔ Critic** — a genuine validation loop. The Advisor proposes, the Critic
  scores, the Advisor prunes and expands the survivors, the Critic scores again.
  Fixed at two rounds by the depth limit.
- **Advisor ↔ Researcher** — gap-fill. If the Critic cannot score a branch for lack
  of evidence, the Advisor may re-query the Researcher **once**. The cap is enforced
  by the budget object, not by convention.

**Brainstorming was considered and rejected.** N-way agent-to-agent discussion
multiplies token cost roughly linearly with participants and produces no measurable
benefit on a bounded decision problem — there are four to six candidate approaches,
not an open design space. It would also make the execution trace unreadable, and that
trace is not a debugging convenience here: it is what the interface shows the user
and what the evaluation suite asserts on. An architecture whose reasoning cannot be
displayed is worse for this product than one that explores slightly less.

---

## Trade-offs considered

**Reliability against latency.** Feedback loops improve reliability and cost time.
The Advisor's critic loop is the clearest case: it turns three model calls into five,
roughly twelve seconds of additional latency, in exchange for pruning that a
single-pass evaluator cannot do. The resolution was not to avoid the loop but to
**bound it explicitly** — five calls, forty-five seconds, eight nodes — and to
degrade to a shallower search when the budget is exhausted, flagging the result as
truncated rather than hiding it.

**Reliability against complexity.** Every agent added is a component that can fail
and a hand-off that can drop context. This is why the count is tied to failure modes
rather than to workflow steps, and why the deterministic layers were kept as tools:
they add reliability without adding coordination surface.

**Coordination overhead against correctness.** The costliest coordination decision
is passing home and persona out-of-band on the run configuration rather than letting
the model supply them. It is more plumbing — every sub-agent must thread the value,
and it does not cross thread-pool boundaries automatically, which has caused real
bugs. It is worth it because the alternative failure is silent and unfalsifiable.

**Determinism against flexibility.** The Router, the pruning thresholds, the
tie-break ladder and the final selection are all plain Python. Each could have been a
model call and each would have been more flexible. They are deterministic because
their failures must be reproducible, and because a system that cannot explain why it
discarded an option cannot be trusted to have discarded the right one.

---

## How the architecture supports effective and scalable problem solving

**Effective**, because responsibility and disposition are matched. The components
that must never improvise cannot: they are code. The component that must generate
alternatives does so under a controller that bounds its cost. The component that
evaluates is separate from the one that decides, so scores can be shown next to a
recommendation and be genuinely independent of it.

**Scalable along the axis that matters** — adding a capability. A new specialist
requires one module, one tool appended to the registry, one interface label and one
router row. Nothing existing changes, because the tool registry is append-only and
the supervisor's routing is expressed as instructions rather than as a hard-coded
switch. The same property makes the system scalable across *homes*: a home is a
directory of documents plus a profile, and the scope filter is metadata, so a third
or thirtieth property adds data rather than code.

The architecture also degrades well, which is a form of scalability under load: any
specialist can fail and the Supervisor still answers from what returned. Search
providers fall back, weather sources fall back, energy prices fall back to published
averages flagged as not-live, and the Advisor falls back to a shallower tree. The
only components with no fallback are the safety guardrails, which fail closed.

---

## Addendum — implementation status as of 2026-08-17

*Added after submission. The topology above is unchanged; this records which
agents now exist in code and what building them revealed.*

**All seven agents are implemented.** The census in this document was written
ahead of the code, and every row now corresponds to a module:

| # | Agent | Module |
|---|---|---|
| 1 | Supervisor / Orchestrator | `agents/orchestrator.py` |
| 2 | Router *(deterministic)* | `agents/router.py` |
| 3 | Researcher | `agents/researcher.py` |
| 4 | Advisor | `agents/advisor.py` + `agents/beam.py` |
| 5 | Critic | `agents/critic.py` |
| 6 | Cost | `agents/cost.py` |
| 7 | Pro Finder | `tools/pros/` |

The claim that Weather, Policy and Safety are **tools, not agents** still holds,
and safety still runs outside the agent boundary entirely.

### The Router, as built

Deterministic as designed, and measurably cheap: **~0.13 ms per turn** on the
table-only path, which 21 of 25 labelled cases take. It combines a
token-boundary phrase table with MiniLM similarity against intent exemplars,
reusing the embedder already loaded for retrieval — no new model, no new
dependency, no key.

Two properties were tightened during implementation:

- **The phrase table outranks the embedder on disagreement.** `matched:
  ["freezing", "tonight"]` is an explanation a person can act on; a cosine of
  0.41 is not. "The router picked wrong" has to be diagnosable from the Logs tab.
- **It advises and never overrides.** A hint is emitted only at medium confidence
  or better, and a low-confidence label produces no hint at all — a guess stated
  as fact would push the model toward a tool the router only suspected, and the
  model has the whole conversation where the router has one sentence.

### The Pro Finder, and why its data was harder than its design

The licence gate works as specified — non-current registrations are **withheld
with a stated reason**, never ranked lower. Two findings from the live registry
changed the implementation, not the design:

1. **The registry has two trade axes, not one.** Regulated trades live in
   `contractorlicensetypecodedesc`; everything else in `specialtycode1desc`. The
   obvious query — filtering `specialtycode1desc` for plumbing — returns **zero
   active firms**, because that value is a retired taxonomy. A product built on it
   would tell a user there are no licensed plumbers in their city while 182 sit
   in the same five cities under the other axis.
2. **Status alone is insufficient.** 28 rows report ACTIVE with an expiry date
   already past, so both are checked and the stricter answer wins.

Eligibility is an **allowlist** of statuses. The live vocabulary has ten values;
a denylist would fail open the moment an eleventh appeared, and invisibly.

### Communication, as built

The bounded feedback loops held. Advisor ↔ Critic is capped by the call budget,
Advisor ↔ Researcher at one gap-fill round. Neither is a convergence condition —
an unbounded critic loop is how a turn becomes a bill.

`home_id` and `persona` ride the run configuration and are never model-authored
arguments, which is what prevents a model filtering itself into an empty result
set that looks identical to "no rule exists".

### Measured

**44/44** evaluation cases pass on the free model, covering cross-home
jurisdiction isolation, audience filtering, the grounding gate, injection
screening, and that the published build cannot reach real data or a paid model.
