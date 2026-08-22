"""Semantic answer cache — reuse an answer when the *meaning* matches.

The exact-match cache in `tools/cache.py` only helps when someone retypes a
question character-for-character. In practice people paraphrase: "can I put
stones in my backyard" and "am I allowed to replace my grass with rocks" want the
same answer. Embedding the question and matching by similarity catches those.

Three rules keep this safe rather than merely fast:

1. **A high similarity floor.** Serving the wrong answer is far worse than being
   slow, so the threshold is deliberately conservative and near-misses fall
   through to a real run.
2. **Context is part of identity, not part of the similarity score.** Persona,
   location, and which saved home is active must match *exactly*; only the wording
   is matched fuzzily. The same question about a different place — or about the
   other house, with its own HOA and its own city code — is a different question.
3. **Never reuse an answer across a safety boundary.** The caller's safety-screen
   fingerprint is part of the exact-match context, so a guarded answer is only
   ever served back to a question that earns the same guard. This used to be
   enforced by refusing to cache guarded questions at all, which was heavier than
   the hazard warranted: the screen is deterministic and re-runs on live input
   every turn, so the guardrail is never skipped — it just had to be emitted
   before the cache could return. Excluding those questions outright meant the
   slowest class of question in the product was the one class that could never
   get faster, however many times it was asked.

Reuses the Chroma client and local embedding model already loaded for RAG, so
this costs no extra dependency and no API calls.
"""
from __future__ import annotations

import hashlib
import json
import time

COLLECTION = "answer_cache"

# Cosine similarity required to treat two questions as the same question.
#
# Calibrated by measurement, not taste. Against "Am I allowed to replace my
# backyard grass with stones?" the local MiniLM embeddings score:
#     0.997  same question, different casing/punctuation
#     0.796  "can i put rocks in my backyard instead of grass?"
#     0.717  "Can I replace my lawn with gravel?"
#     ------ decision boundary ------
#     0.454  "Is xeriscaping allowed in my yard?"   (related, not the same ask)
#     0.377  "Can I build a shed in my backyard?"
#     0.058  "Are my pipes at risk of freezing?"
# 0.65 sits in the empty band between the weakest true paraphrase and the
# strongest false match, with roughly 0.07 of margin on either side.
MIN_SIMILARITY = 0.65

# Answers that depended on live weather go stale quickly; policy answers do not.
TTL_DEFAULT = 30 * 60
TTL_TIME_SENSITIVE = 8 * 60

_TIME_SENSITIVE_TOOLS = {
    "get_weather_forecast", "get_weather_alerts",
    "assess_freeze_risk", "assess_heat_risk",
}


def _collection():
    from memory.rag_store import CACHE_STORE, _embed_fn, _get_client

    return _get_client(CACHE_STORE).get_or_create_collection(
        COLLECTION, embedding_function=_embed_fn(), metadata={"hnsw:space": "cosine"}
    )


def _context_key(persona: str | None, location: str | None,
                 home_id: str | None = None, safety: str = "clean") -> str:
    """Exact-match identity for everything that isn't the question's wording.

    `safety` is the caller's safety-screen fingerprint. It belongs in the exact
    part of the identity rather than the fuzzy part for the same reason persona
    does: two questions that read alike but earn different guardrails are not the
    same question, and similarity cannot be trusted to notice the difference.
    "How do I replace my breaker box?" and "How do I replace my air filter?" are
    far apart, but a paraphrase pair that straddles the hazard regex would not be
    — and serving the unguarded answer for the guarded question is precisely the
    failure this cache must not have.
    """
    return (f"{(persona or 'owner').lower()}|{(location or 'home').lower()}"
            f"|{(home_id or 'primary').lower()}|{(safety or 'clean').lower()}")


def ttl_for(trace: list[dict] | None) -> int:
    """Shorter lifetime when the answer leaned on live weather."""
    names = {s.get("name") for s in (trace or [])}
    return TTL_TIME_SENSITIVE if names & _TIME_SENSITIVE_TOOLS else TTL_DEFAULT


def lookup(
    question: str,
    persona: str | None = None,
    location: str | None = None,
    min_similarity: float = MIN_SIMILARITY,
    home_id: str | None = None,
    safety: str = "clean",
) -> dict | None:
    """Return a cached answer for a semantically equivalent question, or None."""
    question = (question or "").strip()
    if not question:
        return None

    try:
        col = _collection()
        res = col.query(
            query_texts=[question],
            n_results=3,
            where={"context": _context_key(persona, location, home_id, safety)},
        )
    except Exception:
        return None  # cache is an optimisation; never let it break a turn

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    now = time.time()

    for doc, meta, dist in zip(docs, metas, dists):
        similarity = 1 - dist
        if similarity < min_similarity:
            continue
        if float(meta.get("expires_at", 0)) < now:
            continue
        try:
            trace = json.loads(meta.get("trace") or "[]")
        except ValueError:
            trace = []
        return {
            "answer": meta.get("answer", ""),
            "trace": trace,
            "similarity": round(similarity, 3),
            "matched_question": doc,
            "age_seconds": int(now - float(meta.get("created_at", now))),
        }
    return None


def store(
    question: str,
    answer: str,
    trace: list[dict] | None = None,
    persona: str | None = None,
    location: str | None = None,
    ttl: int | None = None,
    home_id: str | None = None,
    safety: str = "clean",
) -> None:
    """Remember an answer for future paraphrases of the same question."""
    question = (question or "").strip()
    if not question or not answer:
        return

    now = time.time()
    ttl = ttl if ttl is not None else ttl_for(trace)
    # Keep the trace small — it is replayed for display, not re-executed.
    slim = [
        {"kind": s.get("kind"), "name": s.get("name"), "args": s.get("args", {})}
        for s in (trace or [])
        if s.get("kind") == "call"
    ]
    context = _context_key(persona, location, home_id, safety)
    entry_id = hashlib.sha1(f"{context}|{question.lower()}".encode()).hexdigest()

    try:
        _collection().upsert(
            ids=[entry_id],
            documents=[question],
            metadatas=[{
                "context": context,
                "answer": answer,
                "trace": json.dumps(slim),
                "created_at": now,
                "expires_at": now + ttl,
            }],
        )
    except Exception:
        pass  # best-effort


def clear() -> int:
    """Drop every cached answer. Returns how many were actually removed.

    Deletes the ENTRIES, not the collection — and that distinction is the fix.
    Dropping and recreating a Chroma collection under a shared client path is the
    exact operation this project already learned to avoid: it kills the segment
    reader for any collection queried earlier in the same process, which is why
    `ingest()` was moved to `get_or_create` plus targeted deletes. `clear()` had
    kept doing it, inside a bare `except: pass`, so when it failed it failed
    invisibly — measured 18 entries before and 18 after, with a supposedly cleared
    question still being served from cache.

    Returning a count rather than `None` is the other half. The admin endpoint
    used to report the *pre-clear* count as its result, so the UI's "Clear cache"
    button announced a number of dropped entries whether or not anything was
    dropped. A maintenance control that cannot fail visibly is one you cannot
    trust during a demo, which is precisely when it gets used.
    """
    try:
        col = _collection()
        before = col.count()
        ids = (col.get(include=[]) or {}).get("ids") or []
        if ids:
            col.delete(ids=ids)
        return max(0, before - col.count())
    except Exception:
        return 0


def count() -> int:
    try:
        return _collection().count()
    except Exception:
        return 0


if __name__ == "__main__":
    HOME = "demo-002"
    LOC = "Minneapolis, MN"
    clear()
    store("Am I allowed to replace my backyard grass with stones?",
          "Yes, with ARC approval per the Lakeshore Commons CC&Rs section 6.2.",
          trace=[{"kind": "call", "name": "search_home_policies"}],
          persona="owner", location=LOC, home_id=HOME)

    probes = [
        ("can i put rocks in my backyard instead of grass?", "paraphrase — should HIT"),
        ("Am I allowed to replace my backyard grass with stones?", "exact — should HIT"),
        ("How often should I change my HVAC filter?", "unrelated — should MISS"),
        ("Do I need a permit to build a pool?", "different topic — should MISS"),
    ]
    for q, note in probes:
        hit = lookup(q, persona="owner", location=LOC, home_id=HOME)
        status = f"HIT  sim={hit['similarity']}" if hit else "miss"
        print(f"  {status:<18} {note:<28} {q}")

    paraphrase = "can i put rocks in my backyard instead of grass?"
    print("\n  different location (must MISS):",
          bool(lookup(paraphrase, persona="owner", location="Seattle, WA", home_id=HOME)))
    print("  different persona  (must MISS):",
          bool(lookup(paraphrase, persona="renter", location=LOC, home_id=HOME)))
    # The one that matters most: the other house has a different HOA, so its
    # answer must never be served here.
    print("  different home     (must MISS):",
          bool(lookup(paraphrase, persona="owner", location=LOC, home_id="demo-001")))
