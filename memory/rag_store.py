"""Retrieval-Augmented Generation (RAG) store over the home-policy corpus.

This is the capstone's "Knowledge & Memory" concept: a persistent vector store
(Chroma) the agent queries to ground its answers in real documents instead of
guessing. Embeddings use Chroma's built-in MiniLM model — no API key required.

Chunking is *section-aware*: each Markdown `##` section becomes one chunk tagged
with its document title and section heading, so every retrieved passage carries a
precise, human-readable citation like "Lakeshore Commons HOA CC&Rs — 6. Landscaping
and Yard Surfaces".

The corpus is laid out one directory per scope:

    <corpus_root>/common/     rules that apply to every saved home
    <corpus_root>/demo-001/   the Dallas, TX home's HOA, permits, STR and tenant docs
    <corpus_root>/demo-002/   the Minneapolis, MN home's

where <corpus_root> is `data/<profile>/corpus` for the active build profile (see
`config.corpus_root`). The directory name *is* the `home_scope`
metadata value, so adding a home is adding a folder and there is no registry to
keep in step. See `_build_where` for why that filter is a safety control and not
just a relevance tweak.
"""
from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

import config
from tools.cache import TTL_RAG, cached
from tools.homes import COMMON_SCOPE, home_scope_ids

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = config.corpus_root()
CHROMA_DIR = config.chroma_dir()
COLLECTION = "home_policies"

# Built-in MiniLM embeddings (downloads a small ONNX model once; no key needed).
# Loaded lazily so that importing this module stays cheap for code paths that
# never touch RAG (e.g. the weather-only demo).
_EMBED_FN_CACHE = None

# Who each document actually applies to, used to narrow the search space per
# request (see `search_policies(audience=...)`).
#
# The rule is deliberately conservative: a document is tagged for one persona
# ONLY when it genuinely does not apply to the other. Everything that binds
# whoever is living in the house — the HOA covenants, the city code, the
# maintenance guides — stays "all", because a renter is bound by the CC&Rs
# through their lease just as the owner is bound directly. Filtering those out
# would hide a real rule, which is a far worse failure than showing one extra
# passage.
#
# So this excludes exactly two things: tenant-rights material from an owner, and
# owner-as-host short-term-rental rules from a renter.
#
# Listed per file rather than by pattern: this is a safety filter, and an
# explicit table is auditable in a way that a filename regex is not.
# Keyed on basename, not on scope+basename, so the same document TYPE keeps its
# audience across every home. demo-001 and demo-002 both carry a
# `renter_policy_summary.md`, and both are renter-only — which is the point: a new
# home added with the conventional filenames inherits the right audience instead
# of silently defaulting to "all" and leaking owner covenants to a tenant.
AUDIENCE_BY_FILE = {
    "renter_policy_summary.md": "renter",
    "wa_renter_policy_summary.md": "renter",
    "short_term_rental_policy.md": "owner",
    "bonney_lake_short_term_rental_policy.md": "owner",
}
DEFAULT_AUDIENCE = "all"

# Human-readable jurisdiction per scope, carried on every chunk so an answer can
# say *which* city's rule it just quoted. This is display and provenance only —
# the filtering is done on `home_scope`, which is the directory name.
#
# ONLY the demo profile's scopes live here. The full profile names its own in
# `<data_root>/jurisdictions.json`, overlaid by `_jurisdiction_labels()`.
#
# They shared this table until the public-build scanner refused a release over
# it: a real city name in shipped source is a leak, and it was right to say so.
# Moving the entry to the data tree also puts the label beside the corpus it
# describes, so adding a home stays "add a folder".
JURISDICTION_BY_SCOPE = {
    "demo-001": "Dallas, TX (synthetic)",
    "demo-002": "Minneapolis, MN (synthetic)",
    COMMON_SCOPE: "any home",
}


def _jurisdiction_labels() -> dict:
    """Scope -> human label, with per-profile entries overlaid from the data tree.

    The full profile's scopes are named in `<data_root>/jurisdictions.json`, not
    here, for two reasons. It keeps the public build free of any reference to the
    private one — the release scanner treats a real city name as a leak, and it is
    right to. And it puts the label beside the corpus it describes, so adding a
    home stays "add a folder" rather than "add a folder and edit a table".

    A missing file is normal: the demo profile's scopes are all listed above.
    """
    labels = dict(JURISDICTION_BY_SCOPE)
    path = config.data_root() / "jurisdictions.json"
    try:
        import json as _json

        labels.update(_json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return labels


def _embed_fn():
    global _EMBED_FN_CACHE
    if _EMBED_FN_CACHE is None:
        _EMBED_FN_CACHE = embedding_functions.DefaultEmbeddingFunction()
    return _EMBED_FN_CACHE


_CLIENTS: dict[str, chromadb.ClientAPI] = {}
_CLIENT_LOCK = threading.Lock()

# Each logical store gets its OWN directory, and therefore its own client.
#
# They used to share one. That is not a tidiness question — on chromadb 1.5.9 it
# is a correctness one. Once a collection has been QUERIED, a write to any OTHER
# collection under the same path invalidates the first one's segment reader for
# the rest of the process, and every subsequent query dies with
#
#     Error executing plan: Internal error: Error creating hnsw segment reader:
#     Nothing found on disk
#
# The on-disk index is fine — a fresh process reads it happily — and the order
# matters: write-then-query works, query-then-write-then-query does not. Since
# the agent queries the corpus and then records the turn to episodic memory,
# every turn was poisoning retrieval for the next one. The evaluation suite hid
# it because tool cases run before agent cases and episodic memory is off by
# default there.
#
# Separate paths sidestep it completely, and the three stores had no reason to
# share one: a policy corpus, an answer cache and a conversation history are
# unrelated.
POLICY_STORE = "policies"
EPISODIC_STORE = "episodic"
CACHE_STORE = "cache"


def _get_client(store: str = POLICY_STORE) -> chromadb.ClientAPI:
    """The one client for `store`, created on first use and reused thereafter.

    Cached per path: constructing a second PersistentClient over a path already
    open is its own source of segment-reader errors, independent of the
    cross-collection bug described above.
    """
    client = _CLIENTS.get(store)
    if client is None:
        with _CLIENT_LOCK:
            client = _CLIENTS.get(store)
            if client is None:
                path = CHROMA_DIR / store
                path.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(path))
                _CLIENTS[store] = client
    return client


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def chunk_markdown(text: str, filename: str) -> list[dict]:
    """Split a Markdown doc into one chunk per `##` section.

    The `#` title heads the document; `##` starts a new chunk; `###` subsections
    stay inside their parent `##` chunk. Preamble before the first `##` (title +
    disclaimer) is treated as boilerplate and skipped for retrieval.
    """
    title = filename
    chunks: list[dict] = []
    section: str | None = None
    body: list[str] = []

    def flush() -> None:
        if section and body:
            content = "\n".join(body).strip()
            if content:
                chunks.append({"title": title, "section": section, "body": content})

    for line in text.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section = line[3:].strip()
            body = []
        else:
            if section is not None:
                body.append(line)
    flush()
    return chunks


def _scope_for(path: Path) -> str:
    """The `home_scope` for a corpus file — its directory name under the corpus root.

    A file sitting loose at the top of the corpus is treated as common rather than
    skipped, so a document dropped in the wrong place is over-shared (visible to
    every home) rather than silently invisible. Being noisy is recoverable; a rule
    that never surfaces looks exactly like a rule that does not exist.
    """
    relative = path.relative_to(CORPUS_DIR)
    return relative.parts[0] if len(relative.parts) > 1 else COMMON_SCOPE


# Bump when chunk_markdown or the metadata shape changes, so existing chunks are
# treated as stale and re-embedded without anyone having to remember --rebuild.
CHUNKER_VERSION = "2"


def ingest(scopes: list[str] | None = None, rebuild: bool = False,
           verbose: bool = True) -> dict:
    """Bring the vector store in line with the corpus on disk. Incremental.

    Previously this dropped the collection and recreated it. That is fine for a
    one-shot script and wrong for a running system: a live reader holding the
    collection fails with "Error creating hnsw segment reader: nothing found on
    disk" the moment the drop lands, which takes down retrieval mid-request. It
    happened twice while this project was being built, both times because an
    ingest ran alongside something else.

    Now nothing is dropped. Each chunk carries a content hash; chunks whose file
    or text changed are replaced, chunks whose file disappeared are deleted, and
    everything else is left alone. Readers never see a missing segment, and a
    re-ingest of an unchanged corpus is a no-op.

    Args:
        scopes: limit the work to these scope directories. Required for
            per-location document acquisition, which must not re-embed every
            other home to add one city's ordinances.
        rebuild: drop and rebuild from scratch. The old behaviour, kept for the
            case where the embedding model itself changes.
    """
    client = _get_client()
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass  # collection may not exist yet
    collection = client.get_or_create_collection(
        COLLECTION, embedding_function=_embed_fn(), metadata={"hnsw:space": "cosine"}
    )

    ids, docs, metadatas = [], [], []
    files = [f for f in sorted(CORPUS_DIR.rglob("*.md"))
             if scopes is None or _scope_for(f) in scopes]
    for path in files:
        scope = _scope_for(path)
        raw = path.read_text(encoding="utf-8")
        for chunk in chunk_markdown(raw, path.stem):
            # The scope is part of the id so two homes can hold documents with the
            # same filename (both have a "permit checklist") without colliding.
            cid = f"{scope}/{path.stem}::{_slug(chunk['section'])}"
            # Embed the heading together with the body for stronger matching.
            document = f"{chunk['title']} — {chunk['section']}\n\n{chunk['body']}"
            # Identity of the chunk's CONTENT, so an unchanged file is skipped and
            # a changed one is replaced. CHUNKER_VERSION is folded in so altering
            # how documents are split invalidates every hash automatically.
            digest = hashlib.sha256(
                f"{CHUNKER_VERSION}\x00{document}".encode()).hexdigest()[:16]
            ids.append(cid)
            docs.append(document)
            metadatas.append({
                "source_title": chunk["title"],
                "section": chunk["section"],
                "file": path.name,
                # Chroma metadata values must be scalars, so audience is stored as
                # a single string and queried with `$in [persona, "all"]` rather
                # than stored as a list.
                "audience": AUDIENCE_BY_FILE.get(path.name, DEFAULT_AUDIENCE),
                "home_scope": scope,
                "jurisdiction": _jurisdiction_labels().get(scope, scope),
                "content_hash": digest,
            })

    # Diff against what is already indexed, within the scopes we just walked.
    where = {"home_scope": {"$in": scopes}} if scopes else None
    try:
        existing = collection.get(include=["metadatas"], **({"where": where} if where else {}))
        indexed = {cid: (m or {}).get("content_hash")
                   for cid, m in zip(existing.get("ids") or [],
                                     existing.get("metadatas") or [])}
    except Exception:
        indexed = {}

    wanted = {cid: m["content_hash"] for cid, m in zip(ids, metadatas)}
    stale = [cid for cid, h in indexed.items()
             if cid not in wanted or wanted[cid] != h]
    changed = [i for i, cid in enumerate(ids)
               if indexed.get(cid) != wanted[cid]]

    if stale:
        collection.delete(ids=stale)
    if changed:
        collection.upsert(
            ids=[ids[i] for i in changed],
            documents=[docs[i] for i in changed],
            metadatas=[metadatas[i] for i in changed],
        )
    unchanged = len(ids) - len(changed)

    # The BM25 index is built from these chunks and cached in-process, so it must
    # be dropped or a re-ingest would leave keyword search scoring stale text.
    try:
        from memory import lexical

        lexical.reset()
    except Exception:
        pass

    by_scope: dict[str, int] = {}
    for meta in metadatas:
        by_scope[meta["home_scope"]] = by_scope.get(meta["home_scope"], 0) + 1

    stats = {"files": len(files), "chunks": len(ids), "scopes": by_scope,
             "added_or_changed": len(changed), "unchanged": unchanged,
             "removed": len(stale), "total_indexed": collection.count(),
             "path": str(CHROMA_DIR)}
    if verbose:
        scope_note = f" [scopes: {', '.join(scopes)}]" if scopes else ""
        print(f"Walked {stats['files']} files -> {stats['chunks']} chunks{scope_note}")
        print(f"  re-embedded {stats['added_or_changed']}, unchanged {stats['unchanged']}, "
              f"removed {stats['removed']}  ->  {stats['total_indexed']} indexed in {CHROMA_DIR}")
        for scope, n in sorted(by_scope.items()):
            print(f"  {scope:<14} {n:>3} chunks  ({_jurisdiction_labels().get(scope, scope)})")
    return stats


def count() -> int:
    """Return how many chunks are in the store (0 if it has not been built)."""
    client = _get_client()
    try:
        return client.get_collection(COLLECTION, embedding_function=_embed_fn()).count()
    except Exception:
        return 0


def _build_where(audience: str | None, home_id: str | None) -> dict | None:
    """Turn persona/home into a Chroma metadata filter.

    Each clause is an `$in` that always includes the shared bucket, so narrowing
    the search space can never hide a rule that applies to everybody. A bare
    equality filter here would be the classic failure: filter to `renter`, miss
    the city ordinance, and refuse a question that had a perfectly good answer.

    The `home_scope` clause is the stronger of the two. Without it, a question
    about the Minneapolis home can retrieve the Dallas HOA's covenants and the
    answer will read as authoritative, because nothing downstream distinguishes a
    rule that applies from one that merely exists. That is the wrong-jurisdiction
    hazard in `docs/safety.md`, and this filter is the control for it.
    """
    clauses = []
    if audience:
        clauses.append({"audience": {"$in": [audience, DEFAULT_AUDIENCE]}})
    if home_id:
        clauses.append({"home_scope": {"$in": home_scope_ids(home_id)}})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _result(doc: str, meta: dict, dense_score: float | None) -> dict:
    return {
        "text": doc,
        "citation": f"{meta['source_title']} — {meta['section']}",
        "source_title": meta["source_title"],
        "section": meta["section"],
        "file": meta["file"],
        "audience": meta.get("audience", DEFAULT_AUDIENCE),
        "home_scope": meta.get("home_scope", COMMON_SCOPE),
        "jurisdiction": meta.get("jurisdiction", JURISDICTION_BY_SCOPE[COMMON_SCOPE]),
        # Cosine similarity from the dense leg, or None for a passage that only
        # BM25 found. Kept under its original name and meaning so existing
        # callers and thresholds are unaffected.
        "score": dense_score,
    }


# Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper and
# is standard; it damps the influence of the very top ranks just enough that one
# list cannot dominate the other.
_RRF_K = 60


@cached(TTL_RAG)
def search_policies(
    query: str,
    k: int = 4,
    audience: str | None = None,
    home_id: str | None = None,
    hybrid: bool = True,
    rerank: bool = True,
    gate_query: str | None = None,
) -> list[dict]:
    """Return the top-k relevant policy passages with citations.

    Three stages, each of which degrades independently:

    1. **Filter** — `audience`/`home_id` narrow the searchable corpus before any
       scoring, so a renter is never grounded on owner-only rules and the Bonney
       Lake home is never grounded on the Dallas home's HOA.
    2. **Hybrid retrieve** — dense (embedding) and BM25 (exact term) results are
       fused with Reciprocal Rank Fusion. Dense finds paraphrases; BM25 finds
       "CC&R 4.2" and "48-inch". See memory/lexical.py.
    3. **Rerank** — a cross-encoder rescores the survivors jointly against the
       query, which is what makes `rerank_score` a trustworthy relevance signal.
       See memory/rerank.py.

    `gate_query` adds a **second** scoring pass, stored as `gate_score`, and
    changes nothing else — not what is retrieved, not the order. It exists
    because the caller of this function is a model-authored search string, and a
    grounding gate that judges the model's own phrasing can be talked past.
    Measured: asked *"can I rent out my roof for a commercial billboard?"* — a
    question the corpus does not answer — the model searched for "rent roof for
    commercial billboard permit HOA Minneapolis", which scored **−0.16** against
    an irrelevant permit passage and cleared the −4.0 floor, while the user's own
    words scored **−10.96** and did not.

    Retrieval deliberately keeps using `query`. The model's phrasing is doing
    real work there: it resolves elided follow-ups like *"what about artificial
    turf instead?"* into something searchable. **The defect and the feature are
    the same mechanism**, so the fix separates them by role — the model chooses
    what to *fetch*, the user's question decides whether it *answers them*.

    Each result: {text, citation, source_title, section, file, audience,
    home_scope, jurisdiction, score, lexical_rank, rerank_score}. `score` remains the dense
    cosine similarity (None if only BM25 surfaced the passage); `rerank_score` is
    a cross-encoder logit and is None when the reranker is unavailable.

    Raises a clear error if the store has not been built yet.
    """
    client = _get_client()
    try:
        collection = client.get_collection(COLLECTION, embedding_function=_embed_fn())
    except Exception as exc:  # collection missing
        raise RuntimeError(
            "Policy knowledge base is empty. Build it first: python ingest.py"
        ) from exc

    where = _build_where(audience, home_id)
    # Retrieve a wider net than we return: fusion and reranking can only reorder
    # what they are given, so a passage the dense leg ranked 6th can still win.
    depth = max(k * 3, 10)

    dense_ids: list[str] = []
    by_id: dict[str, dict] = {}
    query_kwargs = {"query_texts": [query], "n_results": depth}
    if where:
        query_kwargs["where"] = where
    res = collection.query(**query_kwargs)

    for cid, doc, meta, dist in zip(
        (res.get("ids") or [[]])[0],
        (res.get("documents") or [[]])[0],
        (res.get("metadatas") or [[]])[0],
        (res.get("distances") or [[]])[0],
    ):
        dense_ids.append(cid)
        by_id[cid] = _result(doc, meta, round(1 - dist, 3))

    lexical_ids: list[str] = []
    if hybrid:
        from memory import lexical

        candidates = [cid for cid, _ in lexical.search(query, k=depth)]
        # BM25 scores the whole corpus, so its hits must be pushed back through
        # the same metadata filter — otherwise the keyword leg would quietly
        # reintroduce the passages the filter just excluded.
        unseen = [cid for cid in candidates if cid not in by_id]
        fetched: dict[str, dict] = {}
        if unseen:
            try:
                get_kwargs = {"ids": unseen, "include": ["documents", "metadatas"]}
                if where:
                    get_kwargs["where"] = where
                got = collection.get(**get_kwargs)
                for cid, doc, meta in zip(got.get("ids") or [],
                                          got.get("documents") or [],
                                          got.get("metadatas") or []):
                    fetched[cid] = _result(doc, meta, None)
            except Exception:
                fetched = {}  # keyword leg is optional; dense results still stand

        for cid in candidates:
            if cid in by_id:
                lexical_ids.append(cid)
            elif cid in fetched:
                by_id[cid] = fetched[cid]
                lexical_ids.append(cid)

    # Reciprocal Rank Fusion: rank position, not raw score, so the two legs'
    # incomparable scales (cosine similarity vs BM25 magnitude) never need to be
    # normalised against each other.
    fused: dict[str, float] = {}
    for ranked in (dense_ids, lexical_ids):
        for rank, cid in enumerate(ranked):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)

    order = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    for cid in order:
        by_id[cid]["lexical_rank"] = (
            lexical_ids.index(cid) + 1 if cid in lexical_ids else None
        )
        by_id[cid]["rerank_score"] = None

    shortlist = order[: max(k * 2, 8)]
    if rerank and shortlist:
        from memory import rerank as reranker

        scores = reranker.score(query, [by_id[cid]["text"] for cid in shortlist])
        if scores is not None:
            for cid, s in zip(shortlist, scores):
                by_id[cid]["rerank_score"] = round(s, 3)

            # The gate's own score, against the user's words rather than the
            # model's. Computed here so it rides with the passage instead of
            # forcing every caller to re-run the cross-encoder, and applied
            # AFTER the ordering pass above so it cannot change which passages
            # come back or in what order — only the verdict about them.
            if gate_query and gate_query.strip() and gate_query.strip() != query.strip():
                gate_scores = reranker.score(
                    gate_query, [by_id[cid]["text"] for cid in shortlist])
                if gate_scores is not None:
                    for cid, s in zip(shortlist, gate_scores):
                        by_id[cid]["gate_score"] = round(s, 3)

            shortlist = sorted(shortlist, key=lambda cid: by_id[cid]["rerank_score"],
                               reverse=True)

    return [by_id[cid] for cid in shortlist[:k]]


if __name__ == "__main__":
    import json

    ingest()

    PRIMARY, OTHER = "demo-002", "demo-001"

    def show(label: str, query: str, **kwargs) -> None:
        print(f"\n{label}: {query!r}")
        for r in search_policies(query, k=3, **kwargs):
            dense = f"{r['score']:.3f}" if r["score"] is not None else "  -  "
            cross = f"{r['rerank_score']:+.2f}" if r["rerank_score"] is not None else "  -  "
            lex = r["lexical_rank"] or "-"
            print(f"  dense={dense} bm25#{lex:<3} cross={cross}  [{r['home_scope']}] {r['citation']}")

    # Paraphrase: the dense leg carries this one.
    show("Semantic", "Can I replace my front lawn with gravel?", audience="owner", home_id=PRIMARY)
    # Exact identifiers: the BM25 leg carries this one.
    show("Identifier", "RCW 59.18.280 deposit 30 days", audience="renter", home_id=PRIMARY)
    # Same question, two search spaces — the owner-only STR document is filtered
    # out for a renter.
    show("As owner", "Can I list my house on Airbnb?", audience="owner", home_id=PRIMARY)
    show("As renter", "Can I list my house on Airbnb?", audience="renter", home_id=PRIMARY)
    # The headline jurisdiction check: the same question against each home must
    # return that home's own covenants, never the other's.
    show("Fences (WA home)", "How tall can my fence be?", audience="owner", home_id=PRIMARY)
    show("Fences (TX home)", "How tall can my fence be?", audience="owner", home_id=OTHER)
    # Nothing in the corpus answers this; the cross-encoder should score it far
    # below memory.rerank.MIN_RERANK_SCORE.
    show("Ungrounded", "Am I allowed to keep a pet tiger in my backyard?",
         audience="owner", home_id=PRIMARY)

    print("\nFull top hit:")
    print(json.dumps(search_policies("Can I list my house on Airbnb?", k=1,
                                     audience="owner", home_id=PRIMARY)[0], indent=2)[:900])
