# Evaluation

**43 of 43 cases pass**, on the free model this project ships with
(`nvidia/nemotron-3-super-120b-a12b:free`).

Two layers:

| Layer | Cases | Needs a model? | Runtime |
|---|---|---|---|
| Deterministic tool cases | 26 | no — **no API key required** | ~2 min |
| End-to-end agent cases | 17 | yes, live | ~20 min |

```bash
python eval/run_eval.py --tools-only   # the 26 deterministic cases
python eval/run_eval.py                # adds the 17 agent cases
python -m eval.ledger                  # coverage at the current commit
```

---

## Two design decisions, both learned the hard way

**Evidence accrues per case, keyed to a commit *and* a model.** A pass counts only
at the revision it was earned at, on the model the artifact actually ships with.
The free tier rate-limits, and an all-or-nothing suite meant one 429 in case
seventeen discarded sixteen genuine results. The ledger also refuses to attribute
a pass to a commit when the working tree is dirty — it records `dirty:<sha>`
instead, which is not a pass at anything.

**Provider failures are not test failures.** A model that returns an empty
response, or collapses into repeating one character, is recorded as *not run* —
never as a failed assertion. Both shapes occurred, and both were briefly reported
as a headline feature regressing when the truth was a struggling free-tier
provider.

Agent-case assertions are **behavioural** — which tools were called, whether a
citation or a guardrail appeared — rather than assertions about live weather
values, so the suite stays reproducible as conditions change. Web research is
served from a frozen snapshot by default during evaluation, so a grader running
this months from now gets the same evidence rather than whatever the web returned
that day.

---

## Deterministic cases (26) — no model, no network, no API key

| id | what it holds | concept |
|---|---|---|
| T1–T3 | freeze and heat-index maths, and no false freeze on a warm forecast | safety / determinism |
| T4–T6 | geocoding, weather and energy-price fallbacks, each flagged not-live | ReAct recovery |
| T7 | retrieval returns the correct HOA section | RAG |
| T8–T10 | emergencies detected, prevention questions *not* misread as emergencies, high-risk work flagged | safety |
| T11 | PII detected and redacted | privacy |
| T12 | the semantic cache matches paraphrases and **not** other homes or locations | correctness |
| T13–T15 | hybrid retrieval on exact identifiers, audience filtering, reranker separates grounded from ungrounded | RAG |
| T16 | each home retrieves **only** its own jurisdiction's rules | jurisdiction isolation |
| T17–T18 | the demo build cannot reach real data, and its index contains no real-world strings | data separation |
| T19 | episodic memory stays inside its home | memory isolation |
| T20 | PII is redacted before the model **or any store** sees it | privacy |
| T21 | ingest is incremental and non-destructive | reliability |
| T22 | the beam prunes with stated reasons and still answers | Tree-of-Thought |
| T23 | research screens hostile and junk sources | prompt injection |
| T24 | the router advises without overriding | multi-agent |
| T25 | a licence is a **gate**, not a score | safety / Pro Finder |
| T26 | research refuses irrelevant evidence, and no one source owns a pack | research quality |

Two of these are worth singling out, because they assert the opposite of what a
system usually claims:

**T25** asserts that the *highest-rated* plumber in the directory is the one the
system refuses to recommend, because their registration is not current. Ranking an
expired licensee lower still puts them in front of you.

**T26** asserts that a pack of irrelevant sources comes back **empty**. Retrieval
already refused below a relevance threshold; research did not, and the asymmetry
was invisible because scores were normalised before ranking — so a pack in which
nothing was relevant scored identically to one in which everything was.

## End-to-end agent cases (17) — live model

| id | what it holds |
|---|---|
| A1–A2 | freeze risk for the saved home; heat plus an official advisory |
| A3–A4 | HOA landscaping and short-term-rental questions are grounded **and cited** |
| A5 | refuses to invent a rule the documents do not contain |
| A6 | a DIY decision runs the Tree-of-Thought advisor and returns a scored tree |
| A7 | utility savings go to the Cost specialist |
| A8 | a gas-leak emergency **bypasses the model entirely** |
| A9 | refuses step-by-step electrical panel work |
| A10 | an outward action is drafted, never executed |
| A11 | a prevention question is *not* treated as an emergency |
| A12–A14 | multi-turn context, an in-conversation correction, and recall across sessions |
| A15 | a renter is grounded on tenant rules, not owner covenants |
| A16–A17 | each home is answered from its own jurisdiction's rules |

---

## What is deliberately **not** in this repository

**`eval/ledger.json`.** It records the commit each pass was earned at, and this
repository is published with fresh history — so every recorded revision is
unreachable here and a fully green suite would render as entirely stale. The
tables above are the same evidence in a form that travels. Run the suite yourself
and the ledger rebuilds against commits that do exist.

**`eval/transcripts/`.** Full agent transcripts from development runs. They are
excluded because they are development artifacts of the private working tree, not
because anything in the suite is hidden — every case is in `eval/cases.py`, and
every assertion is readable there.

---

## Known limitations, stated rather than discovered

None of these are hidden by the suite; two of them are things the suite found.

- **A model can weaken its own grounding gate by drifting its search query toward
  the vocabulary of the corpus.** Asked whether a roof may be rented for a
  billboard — a question the documents do not cover — the *user's* phrasing scores
  far below the grounding threshold and is correctly refused, while a
  corpus-flavoured paraphrase the model wrote for itself once lifted an irrelevant
  permit document above it. The answer stayed safe both times: no billboard rule
  was invented. This is imprecision, not unsafety, and the fix is to rerank against
  the user's original question rather than the model's search string.
- **The authority table cannot rate what it does not know.** It lists roughly forty
  domains; the open web has millions, and everything else defaults to a neutral
  weight. On queries where every result is an unrated domain, the authority term
  contributes an identical constant and ranking is effectively relevance-only.
- **Web research is reachable only through the Advisor.** A question that does not
  route to a DIY decision cannot reach the web, even when the web is where the
  answer is.
- **Mid-conversation location correction is not closed.** It passes consistently on
  the current model and held roughly one run in four on a previous one. That is
  recorded as a change in the model, not a fix to the system — nothing about the
  rule was edited, and three passing samples is not a measurement.
- **The demo's search provider returns snippets, not page text.** Evidence
  passages are a few hundred characters each. The passage-splitting and
  per-passage screening machinery is largely idle as a result.

For the full risk register, including the risks that are **not** fully mitigated,
see [`docs/safety.md`](docs/safety.md).
