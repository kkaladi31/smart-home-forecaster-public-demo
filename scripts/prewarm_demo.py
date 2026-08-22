"""Warm the answer cache before a demo, so the first ask is not the slow one.

WHY THIS EXISTS
---------------
The demo runs on a free model by design, and a cold turn on a free model is
slow in a way no amount of local optimisation fixes: on a measured high-risk
question, 99.4% of a 183-second turn was model time and 4 ms was tools. The
answer cache already removes that cost on a repeat ask — this just moves the
repeat ask to before the audience is watching.

Guarded questions can be warmed at all only since the safety fingerprint landed
in the cache key. They used to be excluded from the cache outright, which meant
the slowest and most demo-worthy questions in the product ("how do I replace my
breaker box myself") were precisely the ones that could never be warmed.

TIMING — READ THIS BEFORE RELYING ON IT
---------------------------------------
Warming is not permanent, and the windows are short:

    exact-match cache      15 min   (tools/cache.py TTL_ANSWER)
    semantic cache         30 min   (memory/semantic_cache.py TTL_DEFAULT)
    ...if weather was used  8 min   (TTL_TIME_SENSITIVE)

So: run this **within ~8 minutes** of presenting for full coverage, or within
~30 minutes if you are not demonstrating a weather question. Past 30 minutes
every entry has expired and you are cold again. Re-running is cheap and
idempotent — a question that is still warm costs one cache hit, not a model call.

The TTLs are deliberately NOT raised for demo purposes. A stale freeze warning
presented live is a worse failure than a slow one, and this product's flagship
claim is that its weather answers are current.

USAGE
-----
    python scripts/prewarm_demo.py                  # warm the default set
    python scripts/prewarm_demo.py --verify         # warm, then prove each is a hit
    python scripts/prewarm_demo.py --list           # show the questions, ask nothing
    python scripts/prewarm_demo.py -q "..." -q "..." # warm your own questions
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8000"

# The questions worth having warm, chosen to cover the demo's talking points
# rather than to be exhaustive. Each one is a different thing to say out loud.
DEMO_QUESTIONS: list[tuple[str, str]] = [
    ("How do I replace my breaker box myself?",
     "high-risk refusal + licensed referral (0 model calls in the Advisor)"),
    ("How do I run a new gas line to my range myself?",
     "a second hazard category, to show the table is not one rule"),
    ("How do I hang a 20 lb mirror on drywall?",
     "the beam search with a visible pruned branch"),
    ("Am I allowed to replace my backyard grass with stones?",
     "RAG grounding + citation from the home's own CC&Rs"),
    ("Do I need a permit to replace my water heater?",
     "policy retrieval against the city permit checklist"),
    ("Why is my electricity bill so high this month?",
     "the Cost specialist and real dollar figures"),
]


def login(session: requests.Session) -> dict:
    r = session.post(f"{BASE}/api/auth/login",
                     json={"username": "demo", "password": "forecaster"}, timeout=15)
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['token']}"}


def ask(session: requests.Session, headers: dict, question: str) -> dict:
    """Run one turn to completion. Returns timing and whether it was cached."""
    started = time.time()
    cached, answer_chars, error = False, 0, None
    thread = f"prewarm-{int(started * 1000)}"
    with session.post(f"{BASE}/api/chat/stream", headers=headers,
                      json={"message": question, "persona": "owner", "thread_id": thread},
                      stream=True, timeout=600) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            try:
                ev = json.loads(raw[5:].strip())
            except ValueError:
                continue
            kind = ev.get("type")
            if kind == "answer":
                cached = bool(ev.get("cached"))
                answer_chars = len(ev.get("content") or "")
            elif kind == "error":
                error = str(ev.get("content"))[:120]
    return {"seconds": time.time() - started, "cached": cached,
            "chars": answer_chars, "error": error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="ask everything a second time and require a cache hit")
    parser.add_argument("--list", action="store_true",
                        help="print the questions and exit without asking anything")
    parser.add_argument("-q", "--question", action="append", default=None,
                        help="warm this question instead of the default set (repeatable)")
    args = parser.parse_args()

    if args.question:
        questions = [(q, "(supplied on the command line)") for q in args.question]
    else:
        questions = DEMO_QUESTIONS

    if args.list:
        for q, why in questions:
            print(f"  {q}\n      {why}\n")
        return 0

    session = requests.Session()
    try:
        headers = login(session)
    except Exception as exc:
        print(f"Could not reach the API at {BASE}: {exc}")
        print("Start it first:  python -m uvicorn api.main:app --port 8000")
        return 2

    # Printed text stays ASCII: this is run live from a PowerShell console, which
    # is cp1252 here, and an em dash comes out as a replacement character.
    print(f"Warming {len(questions)} question(s). A cold answer on the free model "
          f"can take minutes - that is the point of doing it now.\n")

    results = []
    for i, (question, why) in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        print(f"          {why}")
        try:
            out = ask(session, headers, question)
        except Exception as exc:
            print(f"          FAILED: {type(exc).__name__}: {exc}\n")
            results.append({"question": question, "ok": False})
            continue
        if out["error"]:
            # A provider 429 is the expected failure here, and it is worth
            # saying plainly: nothing was cached, so this one is still cold.
            print(f"          NOT WARMED - {out['error']}\n")
            results.append({"question": question, "ok": False})
            continue
        state = "already warm" if out["cached"] else "warmed"
        print(f"          {state} in {out['seconds']:.1f}s ({out['chars']} chars)\n")
        results.append({"question": question, "ok": True})

    warmed = [r for r in results if r["ok"]]
    print(f"{len(warmed)}/{len(results)} warm.")

    if args.verify and warmed:
        print("\nVerifying - every question below must come back cached and instant:")
        problems = []
        for r in warmed:
            out = ask(session, headers, r["question"])
            mark = "hit " if out["cached"] else "MISS"
            print(f"  {mark}  {out['seconds']:5.1f}s  {r['question'][:58]}")
            if not out["cached"]:
                problems.append(r["question"])
        if problems:
            print(f"\n{len(problems)} question(s) did not come back cached. A cache is an "
                  f"optimisation and never blocks a turn, so the demo still works — it "
                  f"will just be slow on these.")
            return 1
        print("\nAll verified warm.")

    if warmed:
        print("\nRemember: 8 minutes if you are demoing a weather question, 30 otherwise. "
              "Re-run this right before you present.")
    return 0 if len(warmed) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
