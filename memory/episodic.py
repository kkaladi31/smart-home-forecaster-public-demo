"""Episodic memory: what the assistant has been asked and told before.

This is the second half of the capstone's Knowledge-and-Memory concept. The RAG
store (`rag_store.py`) is *semantic* memory — a body of knowledge. This module is
*episodic* memory — a record of past task executions, which is the other success
metric the rubric names ("the system can recall past task executions").

Two layers, kept in sync:

* **SQLite** (`state/<profile>/home.db`) is the source of truth: one row per interaction,
  queryable by recency and thread.
* **Chroma** (`interaction_history` collection) indexes a short summary of each
  interaction so past turns can be recalled *semantically* — "what did you tell
  me about my pipes?" finds the freeze conversation without keyword overlap.

Recording is automatic (the orchestrator calls `record_interaction` after every
turn), so memory does not depend on a weak model remembering to call a tool.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = config.episodic_db()
COLLECTION = "interaction_history"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    user_query  TEXT NOT NULL,
    answer      TEXT NOT NULL,
    summary     TEXT NOT NULL,
    tools_used  TEXT NOT NULL DEFAULT '[]',
    home_id     TEXT NOT NULL DEFAULT ''
);
"""

# Indexes are applied AFTER the additive migration below, not as part of the
# schema script. An index on a column the migration is about to add cannot be
# created before it exists, and `CREATE INDEX IF NOT EXISTS` does not save you:
# the index genuinely does not exist, so SQLite tries to build it and fails on
# the missing column. Ordering matters -- migrate, then index.
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_interactions_thread ON interactions(thread_id);
CREATE INDEX IF NOT EXISTS idx_interactions_time   ON interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_interactions_home   ON interactions(home_id);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # Additive migration for stores created before memory was home-scoped. The
    # state tree is gitignored and rebuildable, but a developer's local history
    # should not have to be thrown away to pick up a column.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(interactions)")}
    if "home_id" not in cols:
        conn.execute("ALTER TABLE interactions ADD COLUMN home_id TEXT NOT NULL DEFAULT ''")
        conn.commit()
    conn.executescript(_INDEXES)
    return conn


def _collection():
    """The Chroma collection used for semantic recall over past interactions."""
    from memory.rag_store import EPISODIC_STORE, _embed_fn, _get_client

    client = _get_client(EPISODIC_STORE)
    return client.get_or_create_collection(
        COLLECTION, embedding_function=_embed_fn(), metadata={"hnsw:space": "cosine"}
    )


def _summarize(user_query: str, answer: str, limit: int = 300) -> str:
    """Cheap extractive summary (no LLM call): the question plus the answer's opening."""
    flat = " ".join(answer.split())
    return f"Q: {user_query.strip()} | A: {flat[:limit]}"


def record_interaction(
    thread_id: str,
    user_query: str,
    answer: str,
    tools_used: list[str] | None = None,
    home_id: str | None = None,
) -> int:
    """Persist one completed interaction to SQLite and index it for semantic recall.

    Returns the new row id. Failures are swallowed deliberately — memory is an
    enhancement and must never break the user's answer.
    """
    tools_used = tools_used or []
    home_id = home_id or ""
    summary = _summarize(user_query, answer)
    created_at = datetime.now().isoformat(timespec="seconds")

    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO interactions (thread_id, created_at, user_query, answer, summary,"
            " tools_used, home_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (thread_id, created_at, user_query, answer, summary, json.dumps(tools_used),
             home_id),
        )
        conn.commit()
        row_id = int(cur.lastrowid)
    finally:
        conn.close()

    try:
        _collection().add(
            ids=[f"interaction-{row_id}"],
            documents=[summary],
            metadatas=[{"row_id": row_id, "thread_id": thread_id, "created_at": created_at,
                        "tools_used": ",".join(tools_used), "home_id": home_id}],
        )
    except Exception:
        pass  # semantic index is best-effort; SQLite remains the source of truth
    return row_id


# Below this cosine similarity a "memory" is noise rather than context. Injecting
# weakly-related history actively hurts: it wastes the free model's limited context
# and invites it to answer from stale memory instead of calling a tool.
#
# When recall is suppressed, and why that is the desired behaviour
# ---------------------------------------------------------------
# `recall()` returns [] — and `format_for_prompt([])` returns "" — whenever every
# candidate scores below MIN_RELEVANCE. In practice that means:
#
#   * **A genuinely new topic.** Asking about freeze risk when the only stored
#     interactions are about HOA landscaping returns nothing, so the agent works
#     from tools instead of being nudged by an unrelated past answer.
#   * **An empty or newly-cleared store.** After "Clear all memory" (or on a first
#     run) there is nothing to score, so recall is silently skipped.
#   * **A question whose PII was screened.** The orchestrator skips recall
#     entirely when `screen_input` flags PII, so nothing is recalled *or* recorded.
#
# Suppression is invisible to the user by design: the turn simply proceeds without
# a "RELEVANT MEMORY" block, and the model is never told that a weak memory
# existed. That matters because the alternative — passing along a 0.2-similarity
# memory with a caveat — reliably produces answers that drift toward the old topic.
#
# Note the asymmetry with the RAG store: retrieval there is now cross-encoder
# reranked (see memory/rerank.py), because a *wrong policy citation* is a
# correctness failure. Episodic recall stays on a plain cosine threshold because
# its failure mode is milder — a missed memory costs a follow-up question, not a
# fabricated rule — and because it runs on every single turn, where the
# reranker's extra ~80ms would be paid whether or not any memory exists.
#
# The threshold is deliberately conservative. If follow-ups start failing to
# resolve elided context ("what about tomorrow?"), lower it; if answers start
# drifting toward stale topics, raise it.
MIN_RELEVANCE = 0.30


def recall(
    query: str,
    limit: int = 3,
    max_age_days: int | None = None,
    min_score: float = MIN_RELEVANCE,
    home_id: str | None = None,
) -> list[dict]:
    """Semantically recall past interactions relevant to `query`, for ONE home.

    Only memories at or above `min_score` are returned, so an unrelated question
    yields nothing rather than dragging in irrelevant history.

    `home_id` scopes recall the same way `home_scope` scopes retrieval. Without
    it this was the one memory layer that crossed homes: RAG filtered hard on
    jurisdiction while episodic recall happily surfaced a Dallas conversation in
    the middle of a Minneapolis answer. That is the same wrong-jurisdiction
    hazard docs/safety.md R13 describes, arriving through memory instead of
    through retrieval, and it is harder to spot because a recalled answer reads
    as something the user was already told.

    Rows written before memory was scoped carry an empty home_id. They are
    included for the active home rather than orphaned — they predate the
    distinction, and dropping a user's history silently is worse than showing it.

    Returns [{row_id, when, user_query, answer, tools_used, score, age_days}].
    """
    where = None
    if home_id:
        where = {"home_id": {"$in": [home_id, ""]}}
    try:
        # Over-fetched well beyond `limit`, because a vector whose SQLite row has
        # been deleted still ranks and is only discovered at hydration, where it
        # is silently dropped. With a narrow fetch a handful of those orphans fill
        # every slot and recall returns NOTHING while a perfectly good memory sits
        # just below the cut — recall degrading to silence as the index ages, with
        # no error anywhere. Measured: 32 of 42 vectors orphaned, and the agent's
        # `recall_memory` answering "I don't have a record of that" about a
        # conversation that was in the database the whole time.
        #
        # `prune_orphans()` fixes the cause; this makes the read survive one.
        # Deliberately NOT self-healing: writing to a collection during a read is
        # the shape of the chromadb segment-reader bug this project already lost
        # two days to, and recall is on the critical path of every turn.
        res = _collection().query(query_texts=[query],
                                  n_results=max(limit * 4, 12),
                                  **({"where": where} if where else {}))
    except Exception:
        return []

    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    if not metas:
        return []

    now = datetime.now()
    candidates: list[dict] = []
    for meta, dist in zip(metas, dists):
        created = meta.get("created_at", "")
        try:
            age = (now - datetime.fromisoformat(created)).days
        except ValueError:
            age = 0
        if max_age_days is not None and age > max_age_days:
            continue
        score = round(1 - dist, 3)
        if score < min_score:
            continue
        candidates.append({"row_id": meta.get("row_id"), "when": created,
                           "age_days": age, "score": score,
                           "tools_used": meta.get("tools_used", "")})

    candidates.sort(key=lambda c: c["score"], reverse=True)
    candidates = candidates[:limit]
    if not candidates:
        return []

    # Hydrate full text from the source of truth.
    conn = _connect()
    try:
        out = []
        for c in candidates:
            row = conn.execute(
                "SELECT user_query, answer, created_at FROM interactions WHERE id = ?",
                (c["row_id"],),
            ).fetchone()
            if row:
                out.append({**c, "user_query": row["user_query"], "answer": row["answer"]})
        return out
    finally:
        conn.close()


def recent(limit: int = 5, thread_id: str | None = None) -> list[dict]:
    """Most recent interactions, optionally scoped to one conversation thread."""
    conn = _connect()
    try:
        if thread_id:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
                (thread_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_threads(limit: int = 30) -> list[dict]:
    """Every conversation, newest first, with a title taken from its first message.

    Backs the "past conversations" list in the UI — episodic memory already
    records each turn with its thread, so the history comes for free.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT thread_id,
                   COUNT(*)      AS turns,
                   MIN(id)       AS first_id,
                   MAX(created_at) AS last_at
            FROM interactions
            GROUP BY thread_id
            ORDER BY MAX(id) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            first = conn.execute(
                "SELECT user_query FROM interactions WHERE id = ?", (r["first_id"],)
            ).fetchone()
            title = (first["user_query"] if first else "Conversation").strip()
            out.append({
                "thread_id": r["thread_id"],
                "title": title[:80] + ("…" if len(title) > 80 else ""),
                "turns": r["turns"],
                "last_at": r["last_at"],
            })
        return out
    finally:
        conn.close()


def thread_messages(thread_id: str) -> list[dict]:
    """Replay one conversation as alternating user/assistant messages."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT user_query, answer, created_at, tools_used FROM interactions "
            "WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        ).fetchall()
    finally:
        conn.close()

    messages = []
    for r in rows:
        messages.append({"role": "user", "content": r["user_query"], "at": r["created_at"]})
        try:
            tools = json.loads(r["tools_used"] or "[]")
        except ValueError:
            tools = []
        messages.append({
            "role": "assistant", "content": r["answer"], "at": r["created_at"],
            "steps": [{"kind": "call", "name": t, "done": True} for t in tools],
        })
    return messages


def count() -> int:
    conn = _connect()
    try:
        return int(conn.execute("SELECT COUNT(*) AS n FROM interactions").fetchone()["n"])
    finally:
        conn.close()


def format_for_prompt(memories: list[dict]) -> str:
    """Render recalled interactions as context to inject into the agent's prompt."""
    if not memories:
        return ""
    lines = ["RELEVANT MEMORY from earlier conversations with this user:"]
    for m in memories:
        when = m["when"][:10] if m.get("when") else "earlier"
        ago = "today" if m.get("age_days") == 0 else f"{m.get('age_days')}d ago"
        answer = " ".join(m["answer"].split())[:220]
        lines.append(f'- On {when} ({ago}) they asked: "{m["user_query"]}" '
                     f"— you answered: {answer}…")
    lines.append("Use this only if it is relevant to the current question; do not repeat it "
                 "verbatim, and prefer fresh tool data for anything time-sensitive.")
    return "\n".join(lines)


def clear_thread(thread_id: str) -> int:
    """Forget one conversation: its SQLite rows and their semantic index entries.

    Used by the UI's "Clear this conversation" button. The Chroma ids are derived
    from the SQLite row ids, so the rows are read *before* they are deleted —
    otherwise the vector copies would be orphaned and still turn up in recall.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id FROM interactions WHERE thread_id = ?", (thread_id,)
        ).fetchall()
        row_ids = [int(r["id"]) for r in rows]
        conn.execute("DELETE FROM interactions WHERE thread_id = ?", (thread_id,))
        conn.commit()
    finally:
        conn.close()

    if row_ids:
        try:
            _collection().delete(ids=[f"interaction-{i}" for i in row_ids])
        except Exception:
            pass  # best-effort, same as when they were written
    return len(row_ids)


def prune_orphans() -> int:
    """Drop indexed vectors whose SQLite row no longer exists. Returns how many.

    An orphan is a memory that can be FOUND but not READ: `recall` ranks it, then
    hydration looks for its row, finds nothing, and drops it without a word. Enough
    of them and recall goes quiet even though the database is full of memories.

    They accumulate whenever rows leave SQLite without their vectors following —
    which is exactly what `clear_all` used to do, because it deleted the SQLite
    rows and then tried to drop the whole Chroma collection inside a bare
    `except: pass`. When that silently failed, every vector survived its row.
    Measured on this machine before the fix: 42 vectors, 10 rows, 32 orphans.
    """
    conn = _connect()
    try:
        live = {int(r["id"]) for r in conn.execute("SELECT id FROM interactions")}
    finally:
        conn.close()

    try:
        col = _collection()
        stored = col.get(include=["metadatas"]) or {}
        ids = stored.get("ids") or []
        metas = stored.get("metadatas") or []
        dead = [vid for vid, meta in zip(ids, metas)
                if int((meta or {}).get("row_id", -1)) not in live]
        if dead:
            col.delete(ids=dead)
        return len(dead)
    except Exception:
        return 0


def clear_all() -> None:
    """Wipe episodic memory (useful for tests and demos).

    Deletes the ENTRIES rather than the collection. Dropping the collection is the
    operation this project has now been bitten by twice — it fails silently under
    a bare `except`, and here that meant the SQLite rows went while every vector
    stayed, turning the whole index into orphans that quietly starve recall.
    """
    conn = _connect()
    try:
        conn.execute("DELETE FROM interactions")
        conn.commit()
    finally:
        conn.close()
    try:
        col = _collection()
        ids = (col.get(include=[]) or {}).get("ids") or []
        if ids:
            col.delete(ids=ids)
    except Exception:
        pass


if __name__ == "__main__":
    clear_all()
    record_interaction("demo", "Are my pipes at risk of freezing?",
                       "Freeze risk is HIGH; shut off and drain outdoor spigots.",
                       ["get_weather_forecast", "assess_freeze_risk"])
    record_interaction("demo", "Can I put stones in my backyard?",
                       "Yes, with ARC approval per the Maple Grove CC&Rs section 4.2.",
                       ["search_home_policies"])
    print("stored:", count())
    hits = recall("what did you say about my pipes freezing?")
    print("\nsemantic recall for a pipes question:")
    for h in hits:
        print(f"  [{h['score']}] {h['user_query']}")
    print("\nprompt injection preview:\n" + format_for_prompt(hits))
