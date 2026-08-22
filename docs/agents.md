# Agents — who exists, what they own, and what is deliberately not one

This is the census the code supports, kept separate from the Checkpoint 5 design
document so the two cannot quietly disagree. Checkpoint 5 describes the target
topology; **this file describes what is built**, and the "Status" column is the
only place the difference is recorded.

The organising principle is **one agent per distinct failure mode**. A step
becomes an agent only when it can be wrong in a way no other component can
catch. Decomposing the workflow into steps and calling each one an agent would
produce a bigger number and a worse system.

| # | Agent | Module | Status | The failure it owns |
|---|---|---|---|---|
| 1 | Supervisor / Orchestrator | `agents/orchestrator.py` | **built** | the wrong specialist is consulted |
| 2 | Router *(deterministic)* | `agents/router.py` | **built** | the intent, complexity or risk of a turn is misread |
| 3 | Researcher | `agents/researcher.py` | **built** | external evidence is missing, stale, or untrustworthy |
| 4 | Advisor (ToT controller) | `agents/advisor.py`, `agents/beam.py` | **built** | the wrong option is chosen |
| 5 | Critic | `agents/critic.py`, `agents/rubric.py` | **built** | an option is *evaluated* wrongly |
| 6 | Cost | `agents/cost.py` | **built** | the numbers are wrong |
| 7 | Pro Finder | `tools/pros/` | **built** | an unlicensed or unsuitable professional is recommended |

All seven exist. The census and the code agree.

## What is deliberately not an agent

Weather assessment, policy retrieval and safety screening are **tools**. Each is
fully deterministic — regex guardrails, a scored vector search behind a grounding
gate, arithmetic over a forecast — and giving any of them agency would hand a
model discretion over exactly the decisions that must not be discretionary.

Safety runs *outside* the agent boundary entirely: an emergency phrase bypasses
the language model rather than being routed to it. Calling these agents would
inflate the count from seven to ten and make the system worse in the process.

---

## The Router

The newest agent, and the one whose design is most constrained by where it sits.

### Why it is deterministic

It runs on **every turn**, including the ones that hit the answer cache and never
reach a model at all. A router that costs a model call is a router you cannot
afford to run *before* deciding whether to spend a model call. Beyond cost, "the
router picked wrong" has to be diagnosable from the Logs tab, and a verdict that
reports `matched: ["freezing", "tonight"]` is an explanation where `cosine 0.41`
is not.

### Two signals, in order of authority

1. **A phrase table**, matched on token boundaries. Terms are written as plain
   words and compiled to `\b`-anchored patterns, so no entry can repeat the
   `tools/contractors.py` bug where the alias `"ac"` matched "repl**ac**e my
   lawn". T24 proves the property across all 160 phrases rather than by example.
2. **MiniLM cosine** against exemplar questions, reusing the embedding function
   already loaded for RAG — no new model, dependency or key. Scored on the
   *margin* between the top two intents, because short-sentence MiniLM cosines
   sit in a narrow band where an absolute threshold means nothing.

On disagreement the table wins the primary slot and the embedder's pick is kept
as a secondary intent, since disagreement usually means the turn spans both.

### The embedding pass is skipped when the table is decisive

Measured: ~1.56 s on the first route (ONNX load plus the exemplar corpus), ~300 ms
on each one after, against ~0.16 ms for a table lookup. A quarter-second on every
turn is most of the latency of a *cached* answer.

Skipping is safe when the table clears the field by a full strong term, because
the embedder cannot change the outcome there — it either confirms, or disagrees
and loses. **20 of the 23 labelled cases take the table-only path.**

The tie case is the subtle one: two intents scoring equally is the *most*
ambiguous input there is, and an early version computed the runner-up as
"highest score strictly below the top", which read a tie as a margin over nothing
and sent the hardest turns down the fast path claiming high confidence. The
latency report caught it by printing `via keyword` on a question written to be
ambiguous.

### It advises and never overrides

Nothing in the Router blocks a tool, vetoes an answer, or narrows retrieval:

- **Beam depth** — `agents.beam.depth_for_complexity` maps a complexity label to
  a depth with `max(base, ...)`, so a label can only raise depth above the
  profile default, never lower it. A demo build asked something genuinely complex
  spends the extra ~28 s (depth 1 measures 25.9 s against depth 2 at 53.8 s); "simple" never buys a cheaper answer. A
  router miss costs model calls, never answer quality. T24 asserts this floor.
- **Tool hints** — a single prompt line, placed after the safety guidance and
  before the question so it cannot read as outranking a guardrail, and explicitly
  labelled as ignorable. Emitted only when something matched *by name*: a
  cosine-only verdict produces no hint at all, because a guess stated as fact
  would push the model toward a tool the router invented.
- **`high_risk`** is not decided here. It is delegated to
  `tools.safety.check_high_risk` so there is one definition of dangerous work in
  the codebase rather than two that can drift. T24 asserts they agree.

A router failure is caught and swallowed at the call site: an unlabelled turn is
exactly the turn this system had before the Router existed.

### Where the verdict travels

On the run config — `configurable.routing` — beside `persona` and `home_id`, and
for the same reason. It is turn metadata the orchestrator computed, not something
the model should be able to assert about its own turn. Consumers read it through
`tools.agent_tools._current_routing()`, which returns `{}` outside a graph run and
requires every caller to treat that as "no opinion".

### Verifying it

```powershell
python -m agents.router              # accuracy on a labelled set, plus latency
python eval/run_eval.py --only T24   # the properties that make a wrong label safe
```

`python -m agents.router` reports intent and complexity accuracy twice — table
alone, then table plus embeddings — so a change to either signal shows up as a
number. The table currently carries 22 of 23 cases on its own; the case it misses
("What year was the house built?") is caught by the embedder, which is the
demonstration that the second signal earns its place.

---

## The Pro Finder

### The licence is a gate, not a score

A contractor whose registration is not current is **withheld with a stated
reason**, never ranked lower. Ranking still puts them in front of the user, and
"we showed them, just further down" is not a defence when the user hires one.

The demo fixture makes the difference testable rather than rhetorical: the
highest-rated plumber for the primary home — 4.9 stars, 388 reviews — is the one
refused. T25 asserts precisely that, because a gate that only ever rejects the
worst option is indistinguishable from a ranking.

Eligibility is an **allowlist** of one value, `ACTIVE`. The live vocabulary has
ten values, and a denylist of the bad ones fails open the day L&I adds an
eleventh — invisibly, since an unrecognised status would sail through as
recommendable. `RE-LICENSED` (9,415 rows) is excluded deliberately: it plausibly
means a lapsed registration since renewed, and "plausibly" is not the standard
for telling someone a contractor is licensed.

### L&I has two trade axes, and assuming one is a wrong answer

| Axis | Column | Trades |
|---|---|---|
| Regulated | `contractorlicensetypecodedesc` | PLUMBING / ELECTRICAL / ELEVATOR CONTRACTOR |
| Everything else | `specialtycode1desc` on CONSTRUCTION CONTRACTOR rows | ROOFING, LANDSCAPING, HVAC, HANDYMAN, … |

The obvious query — `specialtycode1desc LIKE '%PLUMB%'` — returns **zero active
firms**. That specialty is a retired taxonomy surviving only on EXPIRED (662),
RE-LICENSED (537), OUT OF BUSINESS (41) and INACTIVE (1) records. A product built
on it tells the user there are no licensed plumbers in their city while 182 sit
in the same five cities under the other axis: authoritative, specific, and wrong.

Licence type also beats matching the business name. `A B CONTRACTING AND DEV LLC`
is a licensed plumbing contractor whose name never says plumbing.

Restricted trades never fall back to a general contractor — there is no such
thing as a general contractor who may legally do plumbing. Unrestricted trades
do fall back, visibly, with a note: in a small city there may be no registered
roofer, and a general contractor is a real answer there.

### The timezone bug, which is the one to remember

Licence expiry is published as a Washington local date. `date.today()` is the
*process's* date, which in a container is UTC. Comparing against it on 2026-08-16
marked 28 contractors whose registration runs through today as expired — telling
prospective customers that named, properly registered businesses had lapsed.

It surfaced only because two measurements disagreed: a spike using a UTC date
found 28 stale-ACTIVE rows and the contract check using a local date found zero.
Chasing the discrepancy instead of picking the convenient number is what found
it. Each provider now supplies `registry_today()` in its own jurisdiction's
timezone.

### Contract tests are separate from the eval suite

`scripts/contract_check.py` asks *does the world still look how we assumed?* —
columns present, status vocabulary unchanged, both licence-type axes alive, and
**every trade still selecting a non-zero count**. A trade that silently goes to
zero is the retired-taxonomy failure happening again.

It is deliberately **not** in `run_eval.py`. That suite is offline and
reproducible, and folding a live call into it would let a state government's
maintenance window turn the project's regression gate red.

---

## Coordination

**A supervisor star at the top.** The Supervisor delegates to specialists and
owns the user-facing answer. Specialists do not call each other.

**A cyclic sub-graph inside the Advisor.** propose → critique → prune → expand →
critique → select, bounded by `AdvisorBudget` — five calls, forty-five seconds,
eight nodes. The bound is a **constant, not a convergence condition**: an
unbounded critic loop is how a turn becomes a bill.

**Parallel fan-out inside the Researcher**, over independent sub-queries.

**The Router is not in the star at all.** It is a pre-model node that writes a
label into the turn's config and returns. It has no inbound edges and sends no
messages, which is what makes it removable without touching anything else.

Communication is one-way typed dictionaries by default. Two-way in exactly two
places, both bounded: Advisor ↔ Critic, and Advisor ↔ Researcher (gap-fill,
capped at one round).

**Brainstorming was considered and rejected.** N-way agent-to-agent discussion
multiplies token cost linearly with no measurable benefit on a bounded decision
problem, and it makes the tool trace — the thing the UI renders and the eval
asserts on — unreadable.

## Adding a specialist

One module, one `@tool` **appended to the end** of `AGENT_TOOLS` (inserting
mid-list invalidates prompt caching), one `TOOL_LABELS` entry in `Chat.jsx`, and
one row here. Nothing else changes.
