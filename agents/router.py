"""Deterministic turn router — the label a turn carries before the model sees it.

The Router owns exactly one failure mode: *the intent, complexity or risk of a
turn is misread*. It runs at the front of every turn and emits
`{intents, complexity, high_risk, hints}` for the rest of the turn to consult.

Three properties are constraints, not preferences:

**It contains no language model.** It runs on every turn — including the ones
that hit the answer cache and never reach a model at all — so it has to be free
and instant. A router that costs a model call is one you cannot afford to run
*before* deciding whether to spend a model call.

**It advises and never overrides.** Nothing here blocks a tool, vetoes an
answer, or narrows retrieval. The worst a wrong label can do is suggest a search
depth or a tool hint the model is free to ignore, so a router miss degrades to
the behaviour this system had before the router existed. Contrast the
`home_scope` filter in `memory.rag_store`, which *is* a hard control precisely
because a wrong home is unrecoverable — the two are different kinds of thing and
are deliberately built differently.

**Every decision is nameable.** "The router picked wrong" has to be diagnosable
from the Logs tab, so the verdict carries the literal terms that matched. That is
also why the keyword table outranks the embedder on disagreement: `matched:
["freeze", "pipes"]` is an explanation, and `cosine 0.41` is not.

Two signals, in that order of authority:

1. A **phrase table**, matched on token boundaries. Terms are written as plain
   words and compiled to `\\b`-anchored patterns here, so no entry can repeat the
   substring bug in `tools/contractors.py`, where the alias `"ac"` matched
   "repl**ac**e my lawn" and "surf**ac**e".
2. **MiniLM cosine** against a handful of exemplar questions per intent, using
   the embedding function already loaded for RAG — no new model, no new
   dependency, no key. It exists to catch the phrasings the table misses, and it
   is scored on the *margin* between the top two intents rather than an absolute
   threshold, because short-sentence cosines from MiniLM sit in a narrow band and
   an absolute cut on them is meaningless. It is **skipped entirely** when the
   table already clears the field — see DECISIVE_MARGIN, which is what keeps the
   common turn at roughly 0.1ms instead of 256ms.

`high_risk` is not decided here. It is delegated to `tools.safety.check_high_risk`
so there is one definition of dangerous work in the codebase rather than two that
can drift.

Run `python -m agents.router` for the labelled self-test — it reports per-intent
accuracy over a fixed question set and exits non-zero if any case regresses.
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field

from tools.phrases import compile_phrases, find_terms
from tools.safety import check_high_risk

# ---------------------------------------------------------------------------
# Intents
#
# One per genuinely distinct capability, matched to a tool the system actually
# has. Adding an aspirational intent here would put a label into telemetry that
# nothing can act on, which is the documentation-ahead-of-code failure this
# project has already had to correct twice.
# ---------------------------------------------------------------------------
INTENTS = ("weather", "policy", "advice", "cost", "property", "memory", "meta")
GENERAL = "general"

# Which tools an intent suggests. Consumed as a prompt *hint*; the model chooses.
TOOL_HINTS = {
    "weather": ("check_weather_hazards", "get_weather_forecast", "get_weather_alerts"),
    "policy": ("search_home_policies",),
    "advice": ("ask_advisor",),
    "cost": ("analyze_utility_costs",),
    "property": ("get_home_profile",),
    "memory": ("recall_memory",),
    "meta": (),
}

# The tool a high-risk turn actually needs, and the one it must not be pushed
# toward. See `_referral_hints`.
REFERRAL_TOOL = "find_licensed_pros"
ADVISOR_TOOL = "ask_advisor"


def _referral_hints(hints: list[str]) -> list[str]:
    """Rewrite the hint list for a turn the hazard screen has already decided.

    A high-risk turn ends in a referral: the assistant will not describe the work,
    so the useful tools are the ones that answer "who does this, and what does it
    take" — the pro directory first, then any policy lookup for permits.

    `ask_advisor` is REMOVED rather than merely demoted, and that is the point of
    this function. It is the most expensive tool in the product — a live web
    search plus a five-call beam search — and on a high-risk question its own
    deterministic gate rules out every do-it-yourself branch before any of that
    work can change the outcome. Hinting it spent minutes rediscovering a verdict
    a regex had already reached in well under a millisecond.

    The hint stays advisory: the model may still call the advisor, and
    `run_advisor` short-circuits safely when it does. This only stops the router
    from actively recommending it.
    """
    out = [h for h in hints if h != ADVISOR_TOOL]
    out.insert(0, REFERRAL_TOOL)
    return out

# Phrase table. `strong` terms are unambiguous inside this domain; `weak` terms
# corroborate but never carry an intent alone (see KEYWORD_FLOOR). Multi-word
# entries match across any whitespace, so "how   do  I" still hits.
KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "weather": {
        "strong": (
            "weather", "forecast", "freeze", "freezing", "frozen", "snow", "storm",
            "wind", "hail", "heat wave", "cold snap", "blizzard", "ice storm",
            "hurricane", "tornado", "flood", "flooding", "temperature", "degrees",
            "humidity", "rainfall", "wind chill", "frost",
            # Weather-caused damage to the house. Measured gap: "what should I do
            # to prevent a burst pipe?" matched NO weather term, so weather
            # reached the intent list only via the embedder and ranked below
            # advice. For a cold-climate home that is the flagship weather
            # question, and answering it without consulting the forecast is
            # exactly the product this is supposed to not be.
            "burst pipe", "burst pipes", "pipes freeze", "frozen pipe",
            "frozen pipes", "ice dam", "ice dams", "pipe burst",
        ),
        "weak": ("tonight", "tomorrow", "this week", "outside", "rain", "ice", "cold"),
    },
    "policy": {
        "strong": (
            "allowed", "permitted", "permission", "hoa", "covenant", "covenants",
            "ccrs", "bylaw", "bylaws", "ordinance", "permit", "permits", "building code",
            "lease", "landlord", "tenant", "regulation", "restriction", "restrictions",
            "prohibited", "zoning", "homeowners association", "violation", "variance",
            "am i allowed", "can i legally",
        ),
        "weak": ("rules", "approval", "legal", "may i", "code", "association"),
    },
    "advice": {
        "strong": (
            "how do i", "how can i", "how should i", "best way", "what is the best way",
            "install", "installing", "replace", "replacing", "repair", "repairing",
            "fix", "mount", "mounting", "hang", "hanging", "insulate", "insulating",
            "seal", "sealing", "diy", "should i", "recommend", "step by step",
        ),
        "weak": ("advice", "options", "help me", "maintain", "clean", "build", "worth doing"),
    },
    "cost": {
        "strong": (
            "bill", "bills", "cost", "costs", "price", "prices", "rate", "rates",
            "kwh", "utility", "utilities", "electricity bill", "gas bill",
            "save money", "cheaper", "expensive", "budget", "energy cost",
            "how much will it cost", "spend",
        ),
        "weak": ("worth it", "afford", "savings", "dollars"),
    },
    "property": {
        "strong": (
            "my home", "my house", "square feet", "square footage", "year built",
            "floor plan", "roof", "attic", "basement", "crawl space", "foundation",
            "hvac", "furnace", "water heater", "bedrooms", "bathrooms", "garage",
            "home profile",
        ),
        "weak": ("the house", "property", "my place"),
    },
    "memory": {
        "strong": (
            "did i ask", "we discussed", "you said", "you told me", "last time",
            "remember", "previously", "recall", "earlier you",
        ),
        "weak": ("earlier", "again", "before that"),
    },
    "meta": {
        "strong": (
            "what can you do", "what can you help with", "who are you", "what are you",
            "your capabilities", "how do you work",
        ),
        "weak": ("hello", "hi there", "thanks", "thank you"),
    },
}

# Exemplar questions for the embedding pass. Deliberately phrased the way a user
# would ask rather than the way the table is written, so the two signals fail
# independently — an exemplar that echoes its own keywords tests nothing.
EXEMPLARS: dict[str, tuple[str, ...]] = {
    "weather": (
        "Are my pipes going to be at risk this weekend?",
        "What is coming in over the next couple of days?",
        "Should I be worried about the conditions tonight?",
        "Is it going to get bad enough to damage anything outside?",
    ),
    "policy": (
        "Am I actually allowed to do that here?",
        "Does my agreement let me put something on the exterior?",
        "Do I need to ask anyone before making that change?",
        "Is there a rule against this where I live?",
    ),
    "advice": (
        "What is the right approach for putting this up?",
        "I want to get this done myself — where do I start?",
        "Which method would hold up best over time?",
        "Talk me through getting that mounted properly.",
    ),
    "cost": (
        "Why is my monthly statement so high?",
        "How much am I spending to keep this place warm?",
        "Where could I cut back to pay less each month?",
        "Is that going to be worth what it charges me?",
    ),
    "property": (
        "How big is this place again?",
        "What kind of heating system does it have?",
        "When was it originally put up?",
        "Tell me about the layout here.",
    ),
    "memory": (
        "What did we go over the other day?",
        "Bring back what I mentioned about that.",
        "You mentioned something about this already.",
        "Remind me what I asked you about that.",
    ),
    "meta": (
        "What sort of things are you able to handle?",
        "Explain what you are for.",
        "How does this whole thing work?",
        "Hey there.",
    ),
}

# ---------------------------------------------------------------------------
# Decision constants. Named and module-level so a wrong label is tunable from one
# place and the eval can assert against the same numbers the product uses.
# ---------------------------------------------------------------------------
STRONG_WEIGHT = 1.0
WEAK_WEIGHT = 0.5
# A keyword verdict needs one strong hit, or two weak ones. A lone weak term —
# "cold", "again", "rules" — is not evidence of anything.
KEYWORD_FLOOR = 1.0
# Keep a runner-up intent when it scores within this fraction of the winner.
# Multi-intent turns are the common case here ("can I install X, and what will
# it cost?"), and dropping the second intent is what makes a router feel wrong.
#
# The ratio is NOT the only way in, and that matters. Measured on the self-test:
# a ratio alone drops the second intent exactly when the first one stacks several
# terms — "how do I seal the windows and will that lower my bill?" scores advice
# 2.0 against cost 1.0, so the cost half of a two-part question disappears at any
# ratio above 0.5. A second intent that clears KEYWORD_FLOOR on its own has its
# own strong evidence and is kept regardless of how loud the winner was.
SECONDARY_RATIO = 0.6
# Cosine gap the embedder needs between its top two intents before it gets a
# say on its own. Measured over the self-test set: agreeing pairs sit at a gap
# of ~0.05-0.15, and genuinely ambiguous ones below 0.03.
EMBED_MARGIN = 0.04
# Keyword margin over the runner-up above which the table is treated as decisive
# and the embedding pass is SKIPPED entirely.
#
# This is a latency control, and it is worth naming why it is safe. Measured
# here: the first route costs ~1.56s (ONNX load plus the exemplar corpus) and
# every one after it ~256ms, against a ~250ms table lookup of ~0.1ms. The
# Router runs on every turn including the ones that hit the answer cache and
# never reach a model at all, so a quarter-second is real — it is most of the
# latency of a cached answer.
#
# Skipping is safe when the table is decisive because the embedder cannot change
# the outcome there: on agreement it confirms, and on disagreement the keyword
# already wins the primary slot by design. All it can add is a tertiary hint,
# which is the least valuable thing this module produces. When the table is
# ambiguous or silent — which is exactly where the embedder earns its place —
# it still runs.
DECISIVE_MARGIN = 1.0
# Word count above which a single-intent turn stops counting as simple.
SIMPLE_MAX_WORDS = 12

# Comparison and trade-off language. These make a turn complex regardless of
# intent, because they ask for options to be weighed against each other — which
# is the beam search's entire job.
#
# `\w+er than` catches the comparative form generically — cheaper/safer/warmer/
# faster than — rather than enumerating adjectives that will never be complete.
# "other than" is excluded because it is a preposition, not a comparison; "rather
# than" is deliberately left in, since it does state a trade-off.
_TRADEOFF = re.compile(
    r"\b(versus|vs\.?|compare|comparison|trade[\s-]?offs?|better|best|instead of|"
    r"rather than|pros and cons|either|which one|cheaper|cheapest|worth it)\b"
    r"|\b(?!other\b)\w+er\s+than\b",
    re.I,
)

# DIY-versus-hire language. Structurally the same claim as a trade-off — "should
# I do this myself or pay someone" is a two-option comparison written without a
# comparative — and it is the single most common shape of question this product
# receives, so it gets its own pattern rather than being missed by the one above.
_DIY_VS_HIRE = re.compile(
    r"\b(myself|my own|on my own|hire|hiring|contractor|professional|a pro|"
    r"call someone|pay someone)\b",
    re.I,
)


_PATTERNS = {
    intent: {tier: compile_phrases(terms) for tier, terms in tiers.items()}
    for intent, tiers in KEYWORDS.items()
}


@dataclass
class Verdict:
    """What the router concluded, and why.

    `intents` is ordered most-confident first and may be empty — an empty list is
    a real answer meaning "nothing here is nameable", not a failure. `primary` is
    then `general`, and every consumer falls back to its own default.
    """
    primary: str = GENERAL
    intents: list[str] = field(default_factory=list)
    complexity: str = "standard"          # simple | standard | complex
    high_risk: bool = False
    confidence: str = "none"              # high | medium | low | none
    hints: list[str] = field(default_factory=list)
    matched: dict[str, list[str]] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    method: str = "none"                  # keyword+embedding | keyword | embedding | none

    def as_dict(self) -> dict:
        return {
            "primary": self.primary,
            "intents": list(self.intents),
            "complexity": self.complexity,
            "high_risk": self.high_risk,
            "confidence": self.confidence,
            "hints": list(self.hints),
            "matched": {k: list(v) for k, v in self.matched.items()},
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "method": self.method,
        }

    def summary(self) -> str:
        """One line for the Logs tab."""
        if not self.intents:
            return "no intent matched — routing left to the model"
        terms = ", ".join(sorted({t for v in self.matched.values() for t in v})[:4])
        return (f"{'+'.join(self.intents)} · {self.complexity} · "
                f"{self.confidence} confidence"
                + (f" · matched: {terms}" if terms else ""))


# ---------------------------------------------------------------------------
# Signal 1 — the phrase table
# ---------------------------------------------------------------------------
def _keyword_scores(text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for intent, tiers in _PATTERNS.items():
        strong = find_terms(text, tiers["strong"])
        weak = find_terms(text, tiers["weak"])
        hits = strong + weak
        total = len(strong) * STRONG_WEIGHT + len(weak) * WEAK_WEIGHT
        if total:
            scores[intent] = total
            matched[intent] = hits
    return scores, matched


# ---------------------------------------------------------------------------
# Signal 2 — MiniLM exemplar similarity
#
# Availability is a sticky flag rather than a per-call try: if the ONNX model is
# missing or fails to load once, it will fail every time, and retrying it on
# every turn would put a multi-second stall in front of every question. Same
# reasoning, same shape, as `_UNAVAILABLE` in memory/rerank.py.
# ---------------------------------------------------------------------------
_EMBED_LOCK = threading.Lock()
_EXEMPLAR_VECTORS: dict[str, list[list[float]]] | None = None
_EMBED_UNAVAILABLE = False


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _exemplar_vectors():
    """Embed every exemplar once, on first use.

    Under the lock in full, not double-checked: this runs once per process and
    the cost of holding it is one model load, whereas two threads racing into
    `DefaultEmbeddingFunction()` on a cold cache both try to download the same
    ONNX file.
    """
    global _EXEMPLAR_VECTORS, _EMBED_UNAVAILABLE
    with _EMBED_LOCK:
        if _EXEMPLAR_VECTORS is not None or _EMBED_UNAVAILABLE:
            return _EXEMPLAR_VECTORS
        try:
            from memory.rag_store import _embed_fn

            fn = _embed_fn()
            flat = [q for intent in INTENTS for q in EXEMPLARS[intent]]
            vectors = [[float(x) for x in v] for v in fn(flat)]
            out, i = {}, 0
            for intent in INTENTS:
                n = len(EXEMPLARS[intent])
                out[intent] = vectors[i:i + n]
                i += n
            _EXEMPLAR_VECTORS = out
        except Exception:
            _EMBED_UNAVAILABLE = True
            return None
    return _EXEMPLAR_VECTORS


def _embedding_pick(text: str) -> tuple[str | None, dict[str, float]]:
    """Best intent by cosine, or None when the top two are too close to call.

    Scored on max-over-exemplars rather than a centroid: several of these intents
    are genuinely multi-modal — "what does this cost" and "how do I spend less"
    are one intent and not one direction — and averaging them produces a centroid
    that sits near neither.
    """
    vectors = _exemplar_vectors()
    if not vectors:
        return None, {}
    try:
        from memory.rag_store import _embed_fn

        query = [float(x) for x in _embed_fn()([text])[0]]
    except Exception:
        return None, {}

    sims = {intent: max(_cosine(query, v) for v in vecs)
            for intent, vecs in vectors.items()}
    ranked = sorted(sims.items(), key=lambda kv: -kv[1])
    if len(ranked) < 2 or ranked[0][1] - ranked[1][1] < EMBED_MARGIN:
        return None, sims
    return ranked[0][0], sims


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------
def _complexity(text: str, intents: list[str], high_risk: bool) -> str:
    """How much reasoning this turn is likely to need.

    Consumed only to pick a search depth, so the asymmetry is deliberate:
    over-calling complexity costs one extra model round-trip, while under-calling
    it on a question that needed options compared costs the user a worse answer.
    Ties break toward complex.
    """
    words = len(text.split())
    if high_risk:
        # Not because dangerous work needs deep search — the system refuses to
        # describe it either way — but because these turns end in a referral that
        # has to weigh permits, cost and who to call, and a shallow search there
        # reads as a brush-off.
        return "complex"
    if _TRADEOFF.search(text) or _DIY_VS_HIRE.search(text):
        return "complex"
    if len(intents) >= 2 and "advice" in intents:
        return "complex"
    if len(intents) >= 3:
        return "complex"
    if not intents:
        return "standard"
    if (len(intents) == 1 and words <= SIMPLE_MAX_WORDS
            and intents[0] in ("weather", "property", "memory", "meta")):
        return "simple"
    return "standard"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def route(text: str, *, high_risk: bool | None = None, use_embeddings: bool = True) -> Verdict:
    """Label one turn.

    Args:
        text: the user's message, already PII-redacted by `_sanitize_input`. The
            router never needs the raw text and must not be the reason a raw
            message travels further than it has to.
        high_risk: pass the value the safety screen already computed, so a turn
            runs the hazard regexes once. Left None, it is computed here.
        use_embeddings: off in the self-test's keyword-only pass, so the table can
            be measured on its own.

    Never raises. A router that can fail a turn is a router that has become a
    dependency, and this one is meant to be an opinion.
    """
    text = (text or "").strip()
    if high_risk is None:
        try:
            high_risk = bool(check_high_risk(text)["high_risk"])
        except Exception:
            high_risk = False
    if not text:
        return Verdict(high_risk=high_risk)

    kw_scores, matched = _keyword_scores(text)
    kw_top = max(kw_scores.values()) if kw_scores else 0.0
    # Second-highest INCLUDING ties. Filtering to `s < kw_top` instead reads two
    # intents tied at the top — the most ambiguous input there is — as a margin
    # over nothing, and sends the hardest turns down the fast path claiming high
    # confidence. Caught by the latency check reporting `via keyword` on the
    # question written specifically to be ambiguous.
    _ranked = sorted(kw_scores.values(), reverse=True)
    runner_up = _ranked[1] if len(_ranked) > 1 else 0.0
    kw_pick = None
    if kw_top >= KEYWORD_FLOOR:
        # Ties break by declaration order in INTENTS, so the result is stable
        # across runs and across dict orderings.
        kw_pick = min((i for i, s in kw_scores.items() if s == kw_top),
                      key=INTENTS.index)
    decisive = bool(kw_pick) and (kw_top - runner_up) >= DECISIVE_MARGIN

    emb_pick, sims = (None, {})
    if use_embeddings and not decisive:
        emb_pick, sims = _embedding_pick(text)

    # Authority order: a nameable match beats an opaque one. When the two
    # disagree the keyword wins the primary slot and the embedder's pick is kept
    # as a secondary intent rather than discarded — disagreement usually means
    # the turn genuinely spans both.
    #
    # `high` means strong evidence, not "two signals voted". A table hit that
    # clears the field on its own is strong evidence, and saying otherwise would
    # make the fast path report itself as less certain than it is.
    if decisive:
        primary, confidence, method = kw_pick, "high", "keyword"
    elif kw_pick and emb_pick == kw_pick:
        primary, confidence, method = kw_pick, "high", "keyword+embedding"
    elif kw_pick and emb_pick:
        primary, confidence, method = kw_pick, "medium", "keyword+embedding"
    elif kw_pick:
        primary, confidence, method = kw_pick, "medium", "keyword"
    elif emb_pick:
        primary, confidence, method = emb_pick, "low", "embedding"
    else:
        return Verdict(complexity=_complexity(text, [], high_risk),
                       high_risk=high_risk, scores={k: v for k, v in sims.items()},
                       matched=matched)

    intents = [primary]
    for intent, score in sorted(kw_scores.items(), key=lambda kv: -kv[1]):
        if intent == primary:
            continue
        # Either it stands on its own evidence, or it is close enough to the
        # winner to be part of the same question. See SECONDARY_RATIO.
        if score >= KEYWORD_FLOOR or score >= SECONDARY_RATIO * kw_top:
            intents.append(intent)
    if emb_pick and emb_pick not in intents:
        intents.append(emb_pick)

    hints: list[str] = []
    for intent in intents:
        for tool in TOOL_HINTS.get(intent, ()):
            if tool not in hints:
                hints.append(tool)
    if high_risk:
        hints = _referral_hints(hints)

    return Verdict(
        primary=primary,
        intents=intents,
        complexity=_complexity(text, intents, high_risk),
        high_risk=high_risk,
        confidence=confidence,
        hints=hints,
        matched={k: v for k, v in matched.items() if k in intents},
        scores={**{k: v for k, v in kw_scores.items()}},
        method=method,
    )


def hint_for_prompt(verdict: Verdict) -> str:
    """The advisory line added to the model's message, or "" for no hint.

    Worded as a hint on purpose, and omitted entirely when the only evidence was
    a cosine. The line is drawn at *nameable*: high and medium both mean the
    phrase table matched a literal domain term the user typed, which is defensible
    evidence; `low` means the embedder guessed and nothing was matched by name. A
    guess stated as fact is worse than no hint at all — it would push the model
    toward a tool the router invented, and the model has the whole conversation
    where the router has one sentence.
    """
    if verdict.confidence not in ("high", "medium") or not verdict.hints:
        return ""
    tools = ", ".join(verdict.hints[:3])
    return ("[Routing hint (advisory — ignore it if the question needs something "
            f"else): this reads as {'/'.join(verdict.intents[:2])}. "
            f"Likely useful tools: {tools}.]")


# ---------------------------------------------------------------------------
# Self-test: `python -m agents.router`
#
# Labelled by hand, phrased as a user would phrase it, and including the cases
# the table is expected to find hard. Prints per-intent accuracy for the table
# alone and for the table plus embeddings, so a change to either signal shows up
# as a number rather than a feeling.
# ---------------------------------------------------------------------------
_CASES: tuple[tuple[str, str, str], ...] = (
    # question, expected primary intent, expected complexity
    ("Are my pipes at risk of freezing tonight?", "weather", "simple"),
    ("What's the forecast for tomorrow?", "weather", "simple"),
    ("Is there a storm warning for my area?", "weather", "simple"),
    ("Am I allowed to install a satellite dish on the roof?", "policy", "complex"),
    ("Does the HOA permit a fence in the front yard?", "policy", "standard"),
    ("Do I need a permit to replace my water heater?", "policy", "complex"),
    ("How do I hang a 20 lb mirror on drywall?", "advice", "standard"),
    ("What's the best way to insulate an attic hatch?", "advice", "complex"),
    ("Should I replace the furnace filter myself?", "advice", "complex"),
    ("Why is my electricity bill so high this month?", "cost", "standard"),
    ("How much does it cost to run the heat all winter?", "cost", "standard"),
    ("What can I do to save money on utilities?", "cost", "standard"),
    ("How many square feet is my home?", "property", "simple"),
    ("What year was the house built?", "property", "simple"),
    ("What did I ask you about last time?", "memory", "simple"),
    ("Do you remember what I said about the deck?", "memory", "simple"),
    ("What can you do?", "meta", "simple"),
    ("Who are you?", "meta", "simple"),
    # Multi-intent — the case a single-label router gets wrong.
    #
    # "Can I install X" is labelled ADVICE, not policy, and that was a correction:
    # the first version of this set called it policy, the router disagreed, and
    # the router was right. "Can I" is ambiguous between *am I permitted* and *am
    # I able*, and a homeowner asking it alongside "what would it cost" is asking
    # the second. The permission reading needs the permission words, which is the
    # case on the line below it.
    ("Can I install a heat pump, and what would it cost?", "advice", "complex"),
    ("Am I allowed to install a heat pump, and what would it cost?", "policy", "complex"),
    ("How do I seal the windows and will that lower my bill?", "advice", "complex"),
    # High risk: complexity is forced regardless of how short the question is.
    ("How do I run a new gas line to the range?", "advice", "complex"),
    # Trade-off language forces complex even on one intent.
    ("Is a space heater cheaper than running the furnace?", "cost", "complex"),
    # Weather-caused damage. This was a measured gap: no weather term matched, so
    # a freeze-prevention question at a cold-climate home ranked as generic
    # advice and the forecast never entered the answer.
    ("What should I do to prevent a burst pipe at my home?", "weather", "complex"),
    ("How do I stop ice dams forming on the roof?", "weather", "complex"),
)


def _selftest() -> int:
    print("Router self-test\n" + "=" * 68)
    failures = []
    for use_embeddings in (False, True):
        label = "keyword + embedding" if use_embeddings else "keyword only"
        correct = complexity_ok = 0
        print(f"\n--- {label} ---")
        for question, want_intent, want_complexity in _CASES:
            v = route(question, use_embeddings=use_embeddings)
            intent_ok = v.primary == want_intent
            comp_ok = v.complexity == want_complexity
            correct += intent_ok
            complexity_ok += comp_ok
            if not (intent_ok and comp_ok):
                flag = "FAIL" if use_embeddings else "miss"
                print(f"  {flag}  {question[:52]:<52} "
                      f"got {v.primary}/{v.complexity}, want {want_intent}/{want_complexity}")
                if use_embeddings:
                    failures.append(question)
        n = len(_CASES)
        print(f"  intent {correct}/{n}   complexity {complexity_ok}/{n}")
        if not use_embeddings and correct == n:
            print("  (the table carries every case on its own; embeddings are"
                  " insurance, not load-bearing)")

    if _EMBED_UNAVAILABLE:
        print("\nNOTE: the embedder did not load — the second pass ran on keywords"
              " alone.\n      That is the intended degraded behaviour, not a failure.")

    # Latency is a stated property of this module, so it is measured rather than
    # asserted in a comment. The split matters more than the total: the decisive
    # path is what most turns take.
    print("\n--- latency ---")
    import time

    for label, question in (("decisive (table only)", "Are my pipes at risk of freezing tonight?"),
                            ("ambiguous (embeds)", "Can I install a heat pump, and what would it cost?")):
        route(question)  # warm
        t0 = time.perf_counter()
        for _ in range(20):
            route(question)
        per_call = (time.perf_counter() - t0) / 20 * 1000
        used = route(question).method
        print(f"  {label:<24} {per_call:7.2f} ms/turn   via {used}")

    fast = sum(route(q).method == "keyword" for q, _, _ in _CASES)
    print(f"  {fast}/{len(_CASES)} of the labelled set takes the table-only path")

    print("\n--- worked example ---")
    v = route("Can I install a heat pump, and what would it cost?")
    for key, value in v.as_dict().items():
        print(f"  {key:<11} {value}")
    print(f"\n  hint: {hint_for_prompt(v) or '(none — nothing matched by name)'}")

    print("\n" + "=" * 68)
    if failures:
        print(f"FAILED: {len(failures)} case(s)")
        return 1
    print("PASS: every case routed as labelled")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
