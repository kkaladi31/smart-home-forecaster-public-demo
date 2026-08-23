# Smart-Home Forecaster

An agentic assistant that warns a homeowner or renter about weather hazards to
their property, and tells them what they are actually allowed to do about it.

Ask it *"can I replace my backyard grass with stones?"* and it retrieves the
home's own HOA covenants and city permit checklist, works out whether the
occupant's role permits the change, and answers with the clause it relied on.
Ask *"how do I hang a 20 lb mirror?"* and it searches several approaches, scores
each against the home's rules and the occupant's permissions, discards the ones
that violate a rule — **showing you which, and why** — and recommends what
survives.

Built for the **Carnegie Mellon Agentic AI Program** capstone.

> ### Every home in this demo is invented
>
> The addresses, builders, HOAs, covenants, contractors and utility rates are all
> **synthetic** — written for this project. No real property is described.
>
> Only the **city and ZIP are real public geography**, so that free weather APIs
> return genuine forecasts for a plausible point. The weather is real. The house
> is not.
>
> Values carrying invented provenance are marked on hover rather than shouted in
> the interface.

---

## Run it

Requires **Python 3.11+** and **Node 18+**.

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your OpenRouter key
python ingest.py              # builds the vector index — takes about a minute
python -m uvicorn api.main:app --port 8000
```

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Open <http://localhost:5173> and sign in with **`demo` / `forecaster`**.

The only key you need is `OPENROUTER_API_KEY`. Everything else this demo touches
is free and keyless — weather, geocoding, elevation, official advisories, web
search. The language model is a **free** model
(`nvidia/nemotron-3-super-120b-a12b:free`), and the build refuses to run on a
paid one.

> Free model slugs on OpenRouter **rotate, and get withdrawn without notice** —
> the previously pinned one stopped existing mid-project and every model-backed
> answer began returning `404 This model is unavailable for free`. If that
> happens to you, run `python scripts/check_free_model.py`: it probes the pinned
> slug, probes the known-good alternatives, and tells you which to switch to.
> You can repoint without editing code by setting `FREE_LLM_MODEL=<slug>` in
> `.env`.

> **Windows PowerShell:** `&&` is a parser error in PowerShell 5.1 — run each
> line separately.

---

## What it demonstrates

Six agentic concepts, each with an evaluation case that proves it:

| Concept | How it shows up |
|---|---|
| **Tool calling** | 13 tools — weather, hazards, elevation, geocoding, policy retrieval, cost analysis, contractor lookup, memory recall |
| **ReAct + recovery** | Every external call has a fallback chain; a failed source costs a citation, never the answer |
| **RAG + memory** | Section-aware chunking, hybrid retrieval, cross-encoder reranking, a grounding gate that refuses rather than guesses; four memory layers |
| **Tree-of-Thought** | A real beam search — branch 4, then 2; beam width 2; explicit pruning thresholds; selection is `argmax` in Python, not a model call |
| **Multi-agent** | Seven agents, each owning one distinct failure mode, under a supervisor |
| **Safety** | Emergencies bypass the model entirely; high-risk work is refused with a referral; PII is redacted before anything stores it |

---

## The parts worth looking at

**It speaks first.** A forecaster that only answers when asked is a search box.
When freeze or heat reaches moderate, or any official advisory is active for the
location, a notification appears with the level, the headline, and what to do
about it — without being asked.

The text is **not written by the model**, and that is what makes it safe to
appear unprompted. It comes from the same deterministic assessors the agent
itself may not second-guess, plus a table mapping advisory type to home actions.
So it is instant (a model call on the free tier measured 6–100 seconds), it costs
nothing, and it cannot invent a precaution on the one surface that speaks without
being asked. The tailored answer is one click away instead: *Ask the assistant*
puts the question in the chat box and focuses it, but never sends it — the
notification decides something is worth asking; you decide to ask.

Two details worth noticing. **Dismissing is not silencing**: dismissals are keyed
to the *condition*, so dismissing "freeze: moderate" cannot mute the "freeze:
severe" that arrives an hour later, and a dismissed alert collapses to a pill
showing how many are still active. And **official guidance and home actions are
never merged** — the National Weather Service tells people how to stay safe and
stops there, because protecting the building is not its job. That gap is this
product's whole subject: relaying "Heat Advisory — stay hydrated" just repeats
the radio.

---

**The reasoning tree.** Ask a DIY question and open the reasoning panel. Pruned
branches are struck through **with the reason they were discarded**. An option
ruled out by a rule shows *"ruled out"* rather than a low score — "a rule forbade
this" and "we considered it and it was poor" are different claims, and conflating
them is how a system launders a constraint into an opinion.

**Licensed professionals.** Registration status is a **gate, not a ranking
signal**. A contractor whose registration is not current is never recommended —
they appear in a separate *"not recommended"* section with the reason stated.
Ranking an expired licensee lower still puts them in front of you, and "we showed
them, just further down" is no defence once you have hired one. Note that the
highest-rated plumber in the demo directory is the one the gate refuses.

**The Logs tab.** Every tool call, retrieval, model round-trip, cache hit and
guardrail decision, with timings. The router's verdict for each turn appears
here too, so "it picked the wrong tool" is diagnosable rather than mysterious.

**The grounding gate.** Ask something the documents genuinely do not cover — *"can
I rent my roof for a billboard?"* — and it says it has no source for that, rather
than inventing a plausible-sounding rule. This is the behaviour the project cares
most about.

---

## Documentation

| | |
|---|---|
| [`docs/architecture-review.md`](docs/architecture-review.md) | How the system is built and why — diagrams, decision history, known limitations |
| [`docs/how-it-works.md`](docs/how-it-works.md) | A gentler walkthrough of one full request |
| [`docs/reasoning.md`](docs/reasoning.md) | The Tree-of-Thought design: parameters, rubric, pruning rules |
| [`docs/agents.md`](docs/agents.md) | The seven agents and the coordination topology |
| [`docs/safety.md`](docs/safety.md) | The risk register, including the ones **not** fully mitigated |

---

## Evaluation

```bash
python eval/run_eval.py --tools-only   # 25 deterministic cases, no model, ~2 min
python eval/run_eval.py                # adds 17 end-to-end agent cases, ~20 min
python -m eval.ledger                  # coverage at the current commit
```

The deterministic half needs no API key at all.

Two things about the suite are worth knowing, because both were learned the hard
way:

**Evidence is recorded per case, keyed to a commit and a model.** A pass counts
only at the revision it was earned at, on the model the artifact actually ships
with. Anything else is reported as stale and named. The free tier rate-limits,
and an all-or-nothing suite meant one 429 in case seventeen discarded sixteen
genuine results.

**Provider failures are not test failures.** A model that returns an empty
response, or collapses into repeating one character, is recorded as *not run* —
never as a failed assertion. Both shapes occurred, and both were briefly reported
as a headline feature regressing when the truth was a struggling free-tier
provider.

---

## Stack

FastAPI · LangGraph · LangChain · ChromaDB · ONNX Runtime (cross-encoder
reranking) · React · Vite · Tailwind · OpenRouter.

Data sources are free and keyless: Open-Meteo, the US National Weather Service,
the US Census geocoder, the EIA, and DuckDuckGo.

---

## Licence and intent

Coursework, published so the design can be read and criticised. The home data is
invented; the weather is real; the advice is informational and is not a
substitute for a licensed professional. For gas, electrical, flooding or medical
emergencies, contact emergency services or your utility.
