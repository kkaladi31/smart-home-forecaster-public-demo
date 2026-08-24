# Safety Analysis: Potential Unintended Actions and Mitigations

**Smart-Home Forecaster Agent** — Carnegie Mellon Agentic AI Program capstone.

This document is the project's response to the capstone's Safety requirement:
*"Statement of potential unintended actions the system may execute and the steps taken
to address this."* It also backs Section 8 of the final report.

The guiding principle throughout: **safety decisions are made in deterministic code, not
delegated to the language model.** Anything that could hurt someone — hazard ratings,
emergency detection, refusals — is enforced by code that executes regardless of what the
model decides to say.

That principle is what makes this build safe to run on a **small free model**.

It runs on `nvidia/nemotron-3-super-120b-a12b:free` over 100% synthetic property data, and
**refuses to run on a paid one** — an evaluation case asserts it.

Model quality matters for prose, and **not at all** for the guarantees below. The
emergency screen runs before the model is invoked and bypasses it entirely; hazard
ratings come from deterministic assessors; the high-risk refusal is a classifier.
A weaker model changes how an answer reads, not what the system will and will not
do.

**The safety argument is written against the weakest model the system supports**,
because that is the one it actually runs on.

There is a second safety property in how the data is arranged: this build has **no
real property document on disk** to retrieve even by accident. The filesystem is
the primary control and the runtime guard is the backup, not the reverse.

---

## 1. What the system can actually do (action surface)

Scoping the action surface is itself the largest safety control, because it eliminates
entire categories of risk rather than trying to detect them.

**Every tool is read-only.** The agent can fetch weather, alerts, coordinates, elevation,
energy prices, retrieved documents, contractor records, and web results. It computes risk
ratings and savings estimates.

**The agent has no ability to affect the physical or transactional world.** There is no
tool that sends email or SMS, books an appointment, makes a purchase, controls a
thermostat/valve/smart device, or writes to any external system. A request to do those
things can only ever produce *drafted text* for the user to act on themselves.

The only persistent writes are the local vector store (`memory/chroma/`) built from the
project's own document corpus, and local caches — none of which leave the user's machine.

---

## 2. Risk register

| # | Potential unintended action | How it could happen | Mitigation | Where |
|---|---|---|---|---|
| R1 | **Gives dangerous DIY instructions** (electrocution, gas explosion, structural collapse, falls) | User asks how to replace a service panel, move a gas line, remove a load-bearing wall, or work on the roof | Deterministic high-risk classifier refuses step-by-step procedures and redirects to a licensed professional; the model receives a hard override it cannot negotiate. **The refusal and the ranking now use the same verdict.** The Advisor's gate previously classified only the *node's* text — the model's paraphrase of an approach, not the user's question — so a branch could escape the gate by being worded more safely than the job it belonged to: asked how to replace a breaker box, *"Install a subpanel instead"* and *"Swap individual breakers only"* both scored 8.0 on safety and entered the beam, on a question the orchestrator had already refused. The question's verdict is now authoritative and the node's can only add to it, and a high-risk question skips the search entirely and returns the referral (0 model calls), because its outcome was never in doubt. *(Closed 2026-08-19.)* | `tools/safety.py: check_high_risk`, `guidance_for_prompt`, `agents/critic.py: evaluate_deterministic(question_high_risk=…)`, `agents/advisor.py: run_advisor` |
| R2 | **Fails to recognize a life-safety emergency** and answers conversationally while someone is in danger | User reports a gas smell, CO alarm, fire, flooding, or heat illness | Input is screened *before* the model runs; on a match the LLM is **bypassed entirely** and vetted emergency instructions are returned | `tools/safety.py: check_emergency`; `agents/orchestrator.py: answer_with_trace` |
| R3 | **False emergency alarm** — tells a user to evacuate over a planning question | Preventive questions contain emergency words ("how do I prevent a **burst pipe**") — this is the product's flagship use case | Preventive/hypothetical phrasing suppresses emergency classification; regression-tested against all core flows (0 false positives) | `tools/safety.py: _PREVENTIVE_CONTEXT` |
| R4 | **Under-rates a dangerous hazard** — calls a sub-freezing or extreme-heat forecast "no risk" | LLM eyeballs a temperature instead of using the tool (observed in real testing before the fix) | Freeze and heat levels are computed by deterministic assessors; the system prompt forbids the model from judging the numbers itself | `tools/freeze_risk.py`, `tools/heat_risk.py`, `agents/orchestrator.py: SYSTEM_PROMPT` |
| R5 | **Invents rules** — states an HOA/permit/tenant requirement that does not exist | RAG returns weakly-related passages and the model fills the gap | Cite-or-refuse: answers must come from retrieved passages and cite them; when nothing relevant is retrieved the agent says it has no source and refers the user to their HOA/city/lease. **The gate now judges the user's question, not the model's search string.** Previously the model authored the text its own gate was scored on, so it could drift toward corpus vocabulary and pass: asked whether a roof could be rented for a billboard, the user's phrasing scored -10.96 against an irrelevant permit passage and refused, while the model's "rent roof for commercial billboard permit HOA Minneapolis" scored -0.16 and cleared the -4.0 floor. Retrieval still uses the model's phrasing, which is what makes an elided follow-up searchable; only the verdict moved. **This is the same shape as R1** - a gate judging a model-authored paraphrase rather than the user - and the third instance found. *(Closed 2026-08-24.)* | `tools/agent_tools.py: search_home_policies`, `SYSTEM_PROMPT` |
| R6 | **Acts in the real world without consent** — sends a message, books a visit, buys something | User says "email my landlord" / "schedule a repair" | No side-effecting tool exists (§1), *and* a confirmation gate instructs the agent to draft only and hand control back to the user. Because answers are cached, the gate is also part of a cached answer's identity: the screen's verdict is fingerprinted into both cache keys, so an answer written under a confirmation gate — or under a high-risk override — can only ever be served back to a question that earns the same gate, and the guardrail is emitted *before* the cache is consulted rather than after | `tools/safety.py: needs_confirmation`, `agents/orchestrator.py: _safety_fingerprint`, `memory/semantic_cache.py: _context_key` |
| R7 | **Assesses the wrong property** — advice based on a mis-resolved address | Geocoder falls back to a city-level match, or the user's address is ambiguous | The resolved address and data source are reported back on every weather answer; place-level matches are flagged `approximate` | `tools/geocode.py`, `SYSTEM_PROMPT` |
| R8 | **Over-trusts a single data source** and fails or misleads when it is wrong/down | An upstream API outage or bad record | Layered fallbacks with the answering source always disclosed: NWS→Open-Meteo, Census→Open-Meteo, Open-Meteo→USGS, EIA→documented published averages (`live: false`) | `tools/weather.py`, `geocode.py`, `elevation.py`, `energy.py` |
| R9 | **Over-promises financial savings** the user then relies on | Savings presented as precise entitlements rather than estimates | Savings are computed deterministically from published DOE/EIA-RECS bases; every result carries a `basis` per measure plus an assumptions block stating these are estimates; the rate used and whether it was live are always reported | `tools/savings.py`, `agents/cost.py` |
| R10 | **Exposes personal data** | User pastes an SSN, card number, email, or account number | Redaction happens **once, at the top of every turn**, upstream of the model, the conversation checkpointer, episodic memory and the telemetry log — so no store ever receives the raw value. The payment-card pattern is Luhn-validated, because it is otherwise a bare digit run that also matches parcel numbers and meter serials, which this product handles legitimately. Because the text is clean from that point on, PII no longer suppresses memory: the turn is remembered in redacted form rather than silently dropped. *(Corrected 2026-08-15: `redact_pii` previously had no production caller at all — its only callers were in the eval suite, so the test passed while the product redacted nothing and the checkpointer stored raw messages.)* | `agents/orchestrator.py: _sanitize_input`, `tools/safety.py: redact_pii`, eval **T20** |
| R11 | **Leaks secrets into the public repository** | API keys committed to git | Keys live only in `.env`, which is gitignored; `.env.example` ships placeholders | `.gitignore`, `.env.example` |
| R12 | **Presents fabricated citations** | Model invents a plausible source name or URL | Citations are not authored by the model — they are passed through from tool outputs (retrieved passage metadata, web-result URLs, named APIs). For web evidence the model writes `[E3]` and Python resolves it, so a fabricated `[E9]` renders as `[unknown source]` rather than a plausible link | `memory/rag_store.py`, `tools/research/evidence.py` (`resolve_citations`) |
| R13 | **Cites rules from the wrong jurisdiction or an outdated code version** | The user has **two homes in different states**, so both corpora are present at once; a Minneapolis question retrieving the Dallas HOA's covenants would read as authoritative and be wrong | Every chunk carries `home_scope` + `jurisdiction`; retrieval applies a hard metadata filter to the active home plus the shared bucket. The active home comes from the run config the UI sets, **never** from a model-chosen argument, so the model cannot filter itself into the wrong state. Both answer caches include the home in their identity. Documents summarizing real law record their source URL and retrieval date, and say so in-text. Regression-tested in both directions. **Episodic memory is scoped the same way** — recall filters on `home_id`, closing the one layer that previously crossed homes while retrieval did not; a recalled answer is especially dangerous because it reads as something the user was already told | `memory/rag_store.py: _build_where`, `tools/homes.py: current_home_id`, `memory/semantic_cache.py: _context_key`, `memory/episodic.py: recall`, eval **T16** / **T19** / **A16** / **A17** |
| R14 | **Follows instructions embedded in fetched content** (prompt injection) | A retrieved web page or document contains adversarial text | **Five layers, ordered by how much each actually protects.** (1) Fetched text never enters a `SystemMessage` — structural, enforced by keeping grounding and evidence in separate functions, and the only layer that is a control rather than a request. (2) Every passage is delimited and labelled as quoted data, with delimiter lookalikes escaped so a page cannot forge a closing marker. (3) A detector **drops** passages carrying injection markers, invisible/bidi characters or encoded blobs, records the reason, and logs `safety.injection_dropped` at warn — visible, not silent. (4) Citations are not model-authored: the model writes `[E3]` and Python resolves it, so a fabricated reference renders as `[unknown source]` rather than a plausible link. (5) No side-effecting tool exists, so a successful injection can at worst produce bad prose, which the deterministic screens and hazard assessors still override. Pattern detection is deliberately the *least* load-bearing layer — it is an arms race that cannot be won, and treating it as primary is how a system ends up trusting a page because it did not say the magic words. *(Was listed as unmitigated "Phase 6a hardening" until 2026-08-16.)* | `tools/research/untrusted.py`, `tools/research/evidence.py`, `agents/advisor.py: _evidence_block`, eval **T23** |
| R15 | **Answers about the wrong location after the user corrects it** | The user says "actually I'm in X, not the saved address" mid-conversation, and the agent keeps using the saved home | The active home rides the run config from the **UI home switcher**, which is authoritative and cannot be overridden by chat text — so the documented, supported way to change homes is a control, not a sentence. A stated correction is additionally honoured by a dedicated prompt rule placed beside the weather rules. **Residual risk, and an honest note about how it changed.** On the previously pinned free model this rule held in roughly **1 run in 4**, and A13 was at that time recorded as failing rather than skipped. On the free model shipping today it passes consistently — **five runs across three revisions**. That is recorded as **a change in the model, not a fix to the system**: nothing about the rule was edited, and a handful of passing samples is not a measurement. It is treated as open. Consequence is a forecast for a property the user is not in, so the answer always names the resolved address it used — the user can see the mismatch | `agents/orchestrator.py: SYSTEM_PROMPT`, `tools/homes.py: current_home_id`, eval **A13** |
| R16 | **Unprompted output is wrong, or invents a precaution** — the system now SPEAKS FIRST, so one surface produces text nobody asked for and nobody is waiting to judge | A hazard notification fires on freeze/heat at moderate+ or on an official advisory | The notification text is **never model-written**. Levels and actions come from the same deterministic assessors the agent may not second-guess (`assess_freeze_risk`, `assess_heat_risk`) and from a fixed advisory-to-home-actions table. It therefore cannot hallucinate, appears instantly rather than after the 6–100 s a free-tier model call measured, and costs nothing to run on every load. An unrecognised advisory yields **no actions** rather than generic filler. Official guidance and home actions are rendered separately and never merged, so ours is never attributed to the weather service. **Dismissal keys on the condition, not the notification**, so dismissing a moderate warning cannot silence the severe one that follows — and a dismissed alert collapses to a visible pill rather than disappearing | `tools/home_precautions.py`, `tools/alerts.py`, `web/src/components/HazardAlert.jsx` |
| R17 | **Cites sources that do not support the answer** — evidence is attached that is about something else | A search returns results the ranker can order but none of which answer the question; the pack is filled anyway because something must go in the top six | Retrieved passages are dropped below the **same** cross-encoder threshold retrieval refuses at, applied to the raw score rather than the normalised one — normalising first was what hid the problem, since min-max maps the best of six irrelevant passages to 1.0. A pack may therefore come back **empty**, and the answer then cites nothing rather than citing the least-bad five. Separately, no single domain may hold more than two passages, so four extracts from one marketing page cannot present as four sources agreeing. Both drops record a reason and are shown in the interface, and an emptied pack logs `research.no_relevant_evidence` at warn — a search that found nothing and a search whose results were all irrelevant are different faults that look identical from outside | `tools/research/evidence.py: rank_passages`, `agents/researcher.py`, `web/src/components/EvidencePanel.jsx`, eval **T26** |

---

## 3. Human oversight and intervention

The system is advisory. It is designed so a human decides and acts at every consequential
point:

- **Emergencies** hand off immediately to 911, the utility's emergency line, or a licensed
  professional — the agent explicitly stops being the primary resource.
- **High-risk work** is routed to licensed trades, with permit and inspection expectations
  stated so the user knows what a legitimate job involves.
- **Outward actions** are drafted for user review; the user sends, books, or buys.
- **Every answer is source-attributed**, so the user can verify the underlying data rather
  than trusting the model.
- **Policy answers direct users to authoritative confirmation** (their HOA board, city
  permitting office, or lease) before they act.

---

## 4. Data protection and compliance with the program's data rule

The capstone requires *"only publicly available, synthetic, or anonymized information."*

- **Live data** comes from public government/open APIs: NWS (forecast + alerts), US Census,
  USGS, Open-Meteo, EIA.
- **All documents in the RAG corpus are synthetic**, authored for this project and labeled
  "SYNTHETIC DOCUMENT" in-file. They imitate real document types without reproducing any
  real HOA's, city's, or landlord's proprietary text.
- **The sample home, contractors, and utility records are synthetic.** Real builder and
  brand names appear only as factual labels; no copyrighted builder floor plan is copied.
  Floor-plan images are either synthetic or public domain, with licenses recorded.
- **No real user PII is collected or persisted.** Planned authentication (Phase 6b) is a
  lightweight demo login specifically to avoid storing real accounts or locations.
- **Phase 6a caching policy:** fetched public documents will be cached locally and
  gitignored. The system cites source URLs rather than redistributing document text —
  US law is not copyrightable, but code-hosting platforms' terms restrict bulk republication.
- Full provenance is tracked in [`data/SOURCES.md`](../data/SOURCES.md).

---

## 5. Residual risks and limitations

Stated plainly, because pretending these are solved would itself be a safety problem:

1. **Pattern-based screens can miss novel phrasings.** The emergency and high-risk
   classifiers use curated patterns; an unusual description of a gas leak could slip past
   them. They reduce risk substantially but are not exhaustive.
2. **The model can still err in free-form prose.** Guardrails constrain the dangerous
   decisions, not every sentence. In *Demo* mode a free-tier model is used by design (program
   constraint), and its reasoning quality is then the system's weakest component. This limit is
   stated against that weaker model deliberately: it is the floor the guarantees must hold at.
3. **The synthetic corpus is not real law.** Until Phase 6a, policy answers are grounded in
   invented documents. They demonstrate the retrieval mechanism, not actual local rules —
   and the documents say so.
4. **Official alerts are US-only.** `api.weather.gov` returns nothing outside the US; the
   system reports alerts as unavailable rather than implying "no hazards."
5. **Savings figures are estimates**, derived from typical-usage models, not an audit of
   the user's actual consumption.
6. **No professional review.** Nothing here has been reviewed by a licensed electrician,
   plumber, attorney, or insurance professional. It is explicitly not professional advice.

---

## 6. How these controls are verified

Guardrails are covered by the evaluation suite rather than asserted:

- Standalone screening tests (`python -m tools.safety`) cover emergencies, high-risk work,
  outward actions, PII, and a benign control.
- A **regression check against all core user flows** guards against over-triggering — the
  false-alarm failure mode found and fixed during development (R3).
- End-to-end agent tests confirm the emergency bypass, high-risk refusal, confirmation
  gate, and that benign questions are unaffected.
- Results and saved transcripts live in [`eval/`](../eval/).
