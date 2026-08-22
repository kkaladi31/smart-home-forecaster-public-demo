"""BM25 keyword search over the policy corpus — the sparse half of hybrid RAG.

Dense (embedding) retrieval understands *meaning*, which is why it finds the
xeriscaping section for "can I put rocks down". What it is bad at is rare exact
tokens, because an embedding blurs them into their neighbourhood. This corpus is
full of exactly those:

    "CC&R 4.2"   "ARC approval"   "48-inch barrier"   "24 inches"   "STR permit"

Those are also the tokens that matter most here, because every policy answer has
to cite the specific section it came from. BM25 scores on exact term overlap, so
it is strongest precisely where the embeddings are weakest. Running both and
fusing the results is what "hybrid" means.

BM25 is implemented here rather than pulled from a package for one concrete
reason: the whole benefit depends on the *tokenizer* keeping "4.2" and "48-inch"
intact, so the tokenizer had to be written either way. The scoring itself is the
standard Okapi BM25 formula and fits in a few lines.

The index is small (tens of chunks) and rebuilt from Chroma on first use, so
there is nothing extra to persist or keep in sync — re-running `ingest.py`
invalidates it via `reset()`.
"""
from __future__ import annotations

import math
import re
import threading

# Keep digits, dots and hyphens attached to their neighbours so section numbers
# ("4.2"), measurements ("48-inch") and hyphenated compounds survive as single
# terms. A plain \w+ split would shatter "4.2" into "4" and "2" and destroy the
# only signal this index exists to capture.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

# Standard Okapi BM25 parameters. k1 controls term-frequency saturation, b how
# strongly to normalise by document length.
_K1 = 1.5
_B = 0.75

_LOCK = threading.Lock()
_INDEX: "_Bm25Index | None" = None


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class _Bm25Index:
    def __init__(self, ids: list[str], documents: list[str]) -> None:
        self.ids = ids
        self.docs = [tokenize(d) for d in documents]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0

        self.freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for doc in self.docs:
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.freqs.append(counts)
            for term in counts:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        # Okapi IDF with the +1 inside the log, which keeps the value positive for
        # terms that appear in more than half the corpus. Without it, a common term
        # gets a negative weight and can push a genuinely matching passage *down*.
        self.idf = {
            term: math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms or not self.n:
            return []

        scored: list[tuple[str, float]] = []
        for i, counts in enumerate(self.freqs):
            length = len(self.docs[i])
            score = 0.0
            for term in terms:
                tf = counts.get(term)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + _K1 * (1 - _B + _B * length / (self.avgdl or 1))
                score += idf * (tf * (_K1 + 1)) / (denom or 1)
            if score > 0:
                scored.append((self.ids[i], score))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]


def _build() -> "_Bm25Index | None":
    """Read every chunk out of Chroma and index it. Returns None if unavailable."""
    try:
        from memory.rag_store import COLLECTION, _embed_fn, _get_client

        collection = _get_client().get_collection(COLLECTION, embedding_function=_embed_fn())
        data = collection.get(include=["documents"])
    except Exception:
        return None  # knowledge base not built yet; caller falls back to dense only

    ids = data.get("ids") or []
    documents = data.get("documents") or []
    if not ids or not documents:
        return None
    return _Bm25Index(ids, documents)


def search(query: str, k: int = 8) -> list[tuple[str, float]]:
    """Return [(chunk_id, bm25_score)] best-first. Empty if the index is unavailable."""
    global _INDEX
    with _LOCK:
        if _INDEX is None:
            _INDEX = _build()
        index = _INDEX
    if index is None:
        return []
    return index.search(query, k)


def reset() -> None:
    """Drop the cached index — call after re-ingesting the corpus."""
    global _INDEX
    with _LOCK:
        _INDEX = None


if __name__ == "__main__":
    for q in ["section 4.2 xeriscaping", "ARC approval", "48-inch barrier",
              "can I put rocks in my yard"]:
        print(f"\n{q!r}")
        for cid, score in search(q, k=3):
            print(f"   {score:6.2f}  {cid}")
