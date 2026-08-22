"""Build (or rebuild) the policy knowledge base from this profile's corpus root.

Run this once before using the policy Q&A features:
    python ingest.py

Safe to re-run any time you add or edit corpus documents.

It also warms the reranker, because this is the documented first command and a
fresh clone otherwise downloads a 23 MB cross-encoder on its first *question* —
see `_warm_reranker`.
"""
from memory.rag_store import ingest


def _warm_reranker() -> None:
    """Fetch the cross-encoder now rather than during the first user question.

    The weights are not in the repository — they are downloaded on first use into
    `memory/models/`, which is correct (23 MB of binary does not belong in git)
    but puts the download in front of whatever happens to touch retrieval first.

    That is not merely slow, it is silently WRONG. `rerank.score` returns None
    when the model is unavailable and callers fall back, so a clone whose
    download had not finished ranked evidence by a degraded path while every
    surface still reported success. It was caught because the public build's
    verify step failed T23 — a forum outranking a .gov source on equal relevance
    — in a tree where the model had not yet arrived, and passed in the same tree
    minutes later once it had.

    Best-effort by design: no network, no model, still a working knowledge base.
    Retrieval degrades, and `available()` reports it.
    """
    from memory import rerank

    print("\nWarming the reranker (one-time ~23 MB download on a fresh clone)…")
    if rerank.available():
        print(f"  ready: {rerank.MODEL_DIR}")
    else:
        print("  UNAVAILABLE — retrieval will fall back to vector similarity "
              "alone. Re-run once you have network access.")


if __name__ == "__main__":
    ingest()
    _warm_reranker()
