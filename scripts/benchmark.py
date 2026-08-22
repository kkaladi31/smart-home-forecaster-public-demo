"""Re-measure every performance claim the documentation makes.

WHY THIS EXISTS
---------------
Every latency figure in `docs/architecture-review.md` was measured on
`openai/gpt-oss-20b:free`. OpenRouter withdrew that model from the free tier on
2026-08-21, so those numbers now describe a system nobody can run. Rather than
hand-patch twenty numbers once and be in the same position after the next model
change, the measurements live here and can be re-run.

    python scripts/benchmark.py                 # everything
    python scripts/benchmark.py --quick         # deterministic parts only, no LLM
    python scripts/benchmark.py --repeats 5     # more samples per figure
    python scripts/benchmark.py --markdown      # emit a table to paste into docs

MEDIAN, NOT MEAN. A free endpoint occasionally stalls for tens of seconds; one
such sample drags a mean somewhere the system never actually goes. The median of
three is what a person sitting in front of the app would call typical.

WHAT CANNOT BE RE-MEASURED, AND WHY THAT IS FINE
------------------------------------------------
Some documented figures are before/after pairs where the "before" is *deleted
code* — the six-tool weather path that `check_weather_hazards` replaced, for
instance. Those cannot be re-run without reverting the optimisation, and they
were measured like-for-like on one model at one commit, so the COMPARISON
remains valid evidence for the decision it justified. What changes is that the
absolute seconds no longer describe today's system. Such figures stay in the
docs labelled with the model and date they came from; this script supplies the
current number alongside.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import active_model  # noqa: E402

RESULTS: list[dict] = []


def record(name: str, unit: str, samples: list[float], note: str = "") -> None:
    if not samples:
        RESULTS.append({"name": name, "unit": unit, "median": None,
                        "min": None, "max": None, "n": 0, "note": note or "no samples"})
        return
    RESULTS.append({
        "name": name, "unit": unit,
        "median": statistics.median(samples),
        "min": min(samples), "max": max(samples),
        "n": len(samples), "note": note,
    })


def fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    if unit == "ms":
        return f"{value:.1f} ms" if value < 100 else f"{value:.0f} ms"
    return f"{value:.1f} s"


# ---------------------------------------------------------------------------
# Deterministic components — no model, so these are stable and cheap.
# ---------------------------------------------------------------------------
def bench_deterministic(repeats: int) -> None:
    print("\n=== deterministic components (no model) ===")

    from agents.router import route
    from memory import rerank
    from memory.lexical import search as lexical_search
    from memory.rag_store import search_policies

    route("warm up the embedder", high_risk=False)
    samples = []
    for _ in range(max(repeats * 10, 30)):
        t = time.perf_counter()
        route("Are my pipes at risk of freezing tonight?", high_risk=False)
        samples.append((time.perf_counter() - t) * 1000)
    record("Router verdict", "ms", samples, "table + embedding path")
    print(f"  Router verdict            {fmt(statistics.median(samples), 'ms')}")

    q = "can I replace my lawn with gravel"
    samples = []
    for _ in range(repeats * 3):
        t = time.perf_counter()
        lexical_search(q, k=5)
        samples.append((time.perf_counter() - t) * 1000)
    record("BM25 lexical search", "ms", samples)
    print(f"  BM25 lexical search       {fmt(statistics.median(samples), 'ms')}")

    passages = [p["text"] for p in search_policies(q, k=5, home_id="demo-002")]
    if passages:
        rerank.score(q, passages[:1])
        samples = []
        for _ in range(repeats * 3):
            t = time.perf_counter()
            rerank.score(q, passages)
            samples.append((time.perf_counter() - t) * 1000)
        record("Cross-encoder rerank", "ms", samples, f"{len(passages)} passages")
        print(f"  Cross-encoder rerank      {fmt(statistics.median(samples), 'ms')}"
              f"  ({len(passages)} passages)")

    # The tool cache has to be cleared between samples or this measures the cache,
    # not the search: an identical repeat returns in 0.0 ms against 484 ms cold.
    # A benchmark that silently measures its own memoisation is worse than none —
    # it reports a number the user will never experience and looks like a triumph.
    from tools import cache as tool_cache
    samples = []
    for _ in range(repeats * 3):
        tool_cache.clear()
        t = time.perf_counter()
        search_policies(q, k=3, home_id="demo-002")
        samples.append((time.perf_counter() - t) * 1000)
    record("Full RAG search (cold)", "ms", samples, "hybrid + rerank, cache cleared")
    print(f"  Full RAG search (cold)    {fmt(statistics.median(samples), 'ms')}"
          f"  (hybrid + rerank)")

    samples = []
    search_policies(q, k=3, home_id="demo-002")   # populate
    for _ in range(repeats * 3):
        t = time.perf_counter()
        search_policies(q, k=3, home_id="demo-002")
        samples.append((time.perf_counter() - t) * 1000)
    record("Full RAG search (cached)", "ms", samples, "repeat of an identical query")
    print(f"  Full RAG search (cached)  {fmt(statistics.median(samples), 'ms')}")

    from tools.contractors import find_contractors
    samples = []
    for _ in range(repeats * 3):
        t = time.perf_counter()
        find_contractors("run a gas line", limit=3, home_id="demo-002")
        samples.append((time.perf_counter() - t) * 1000)
    record("Pro directory lookup", "ms", samples)
    print(f"  Pro directory lookup      {fmt(statistics.median(samples), 'ms')}")

    from agents.advisor import run_advisor
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        run_advisor("How do I replace my breaker box myself?", persona="owner",
                    home_id="demo-002")
        samples.append((time.perf_counter() - t) * 1000)
    record("Advisor high-risk short-circuit", "ms", samples, "0 model calls")
    print(f"  Advisor short-circuit     {fmt(statistics.median(samples), 'ms')}"
          f"  (0 model calls)")


# ---------------------------------------------------------------------------
# End-to-end turns. These are what the documentation's headline numbers describe.
# ---------------------------------------------------------------------------
def _turn(question: str, home_id: str = "demo-002", cold: bool = True) -> dict:
    """Run one turn through the streaming path and return its own measurements."""
    from agents.orchestrator import stream_answer
    from memory import semantic_cache
    from tools import cache

    if cold:
        cache.clear()
        semantic_cache.clear()

    out = {"elapsed_s": None, "first_token_s": None, "tools": 0,
           "llm_turns": None, "llm_s": None, "tool_s": None,
           "tokens_in": None, "tokens_out": None, "cached": False, "chars": 0}
    started = time.perf_counter()
    for ev in stream_answer(question, persona="owner", thread_id=f"bench-{time.time()}",
                            use_memory=False, home_id=home_id):
        kind = ev.get("type")
        if kind == "tool_call":
            out["tools"] += 1
        elif kind == "answer":
            out["chars"] = len(ev.get("content") or "")
            out["cached"] = bool(ev.get("cached"))
        elif kind == "done":
            out["elapsed_s"] = (ev.get("elapsed_ms") or 0) / 1000
            for src, dst, div in (("llm_turns", "llm_turns", 1),
                                  ("llm_ms", "llm_s", 1000),
                                  ("tool_ms", "tool_s", 1000),
                                  ("first_token_ms", "first_token_s", 1000),
                                  ("tokens_in", "tokens_in", 1),
                                  ("tokens_out", "tokens_out", 1)):
                if ev.get(src) is not None:
                    out[dst] = ev[src] / div
    if out["elapsed_s"] is None:
        out["elapsed_s"] = time.perf_counter() - started
    return out


QUESTIONS = [
    ("Flagship weather answer", "Are my pipes at risk of freezing tonight?", "demo-002"),
    ("Policy answer (RAG + citation)",
     "Am I allowed to replace my backyard grass with stones?", "demo-002"),
    ("DIY answer (beam search)", "How do I hang a 20 lb mirror on drywall?", "demo-002"),
    ("High-risk referral", "How do I replace my breaker box myself?", "demo-002"),
    ("Cost answer", "Why is my electricity bill so high this month?", "demo-002"),
]


def bench_turns(repeats: int) -> None:
    print(f"\n=== end-to-end turns on {active_model()} ===")
    print(f"  {'question':<34}{'wall':>8}{'1st tok':>9}{'model':>8}"
          f"{'tools':>7}{'turns':>7}")
    for label, question, home in QUESTIONS:
        walls, firsts, llms, tools, turns = [], [], [], [], []
        for _ in range(repeats):
            try:
                r = _turn(question, home_id=home, cold=True)
            except Exception as exc:
                print(f"  {label:<34}  FAILED {type(exc).__name__}: {exc}"[:110])
                continue
            if not r["chars"]:
                continue
            walls.append(r["elapsed_s"])
            if r["first_token_s"]:
                firsts.append(r["first_token_s"])
            if r["llm_s"]:
                llms.append(r["llm_s"])
            tools.append(r["tools"])
            if r["llm_turns"]:
                turns.append(r["llm_turns"])
        if not walls:
            record(label, "s", [], "no successful sample")
            print(f"  {label:<34}  no successful sample")
            continue
        note = (f"{int(statistics.median(tools))} tool calls, "
                f"{int(statistics.median(turns)) if turns else '?'} model turns, "
                f"model time {fmt(statistics.median(llms) if llms else None, 's')}")
        record(label, "s", walls, note)
        print(f"  {label:<34}{statistics.median(walls):>7.1f}s"
              f"{(statistics.median(firsts) if firsts else 0):>8.1f}s"
              f"{(statistics.median(llms) if llms else 0):>7.1f}s"
              f"{int(statistics.median(tools)):>7}"
              f"{(int(statistics.median(turns)) if turns else 0):>7}")

    # The cached path, measured deliberately rather than inferred.
    warm = []
    q = QUESTIONS[1][1]
    _turn(q, cold=True)                       # populate
    for _ in range(max(repeats, 3)):
        r = _turn(q, cold=False)
        if r["cached"]:
            warm.append(r["elapsed_s"])
    # Labelled as in-process on purpose. `stream_answer` computes elapsed_ms
    # itself, so this excludes HTTP, SSE framing and the browser — it reads as
    # 0.00 s, which is true and is not a number any user can observe. Measured
    # end-to-end through the API the same path is ~0.25 s. Publishing the
    # in-process figure would overstate the cache by two orders of magnitude.
    record("Cached answer (repeat ask, in-process)", "s", warm,
           "no model call; ~0.25 s end-to-end over HTTP")
    if warm:
        print(f"  {'Cached answer (in-process)':<34}{statistics.median(warm):>7.2f}s"
              f"   (no model call; ~0.25 s over HTTP)")


def bench_advisor(repeats: int) -> None:
    """The beam's internals, which `docs/reasoning.md` and checkpoint-4 quote.

    Only the CURRENT shapes are measurable. The six-criterion critique those docs
    compare against was deleted when the evaluation was split, so its 60.7 s is a
    historical figure from a design that no longer exists — re-running it would
    mean rebuilding the thing the split replaced.
    """
    print(f"\n=== advisor internals on {active_model()} ===")
    from agents.advisor import _expand, _gather, _grounding_text, _propose, _evidence_block
    from agents.critic import critique
    from agents.llm import build_llm
    from agents.beam import Node

    question = "How do I hang a 20 lb mirror on drywall?"
    llm = build_llm()
    ctx = _gather(question, persona="renter", home_id="demo-002")
    grounding = _grounding_text(ctx, "renter")
    ev = _evidence_block(ctx)

    proposals, samples = [], []
    for _ in range(repeats):
        t = time.perf_counter()
        proposals = _propose(llm, question, grounding, ev) or proposals
        samples.append(time.perf_counter() - t)
    record("Advisor propose call", "s", samples, "b1 = 4 strategies, one call")
    print(f"  propose (b1=4)            {fmt(statistics.median(samples), 's')}")

    nodes = [Node(id=f"d1-{i}", name=str(p.get("name") or f"Option {i}"),
                  summary=str(p.get("summary") or ""), depth=1, order=i)
             for i, p in enumerate((proposals or [])[:4])]
    if nodes:
        samples = []
        for _ in range(repeats):
            t = time.perf_counter()
            critique(nodes, llm=llm, task=question, grounding=grounding)
            samples.append(time.perf_counter() - t)
        record("Advisor critique call", "s", samples,
               f"3 subjective criteria, {len(nodes)} nodes batched")
        print(f"  critique (3 criteria)     {fmt(statistics.median(samples), 's')}"
              f"  ({len(nodes)} nodes, batched)")

    from agents.advisor import run_advisor
    for depth in (1, 2):
        samples = []
        for _ in range(repeats):
            t = time.perf_counter()
            run_advisor(question, persona="renter", home_id="demo-002", depth=depth)
            samples.append(time.perf_counter() - t)
        record(f"Full advisor run, depth {depth}", "s", samples)
        print(f"  full run, depth {depth}         {fmt(statistics.median(samples), 's')}")


def emit_markdown() -> None:
    print("\n\n<!-- paste into docs -->")
    print(f"| Measurement | Median | Range | n | Notes |")
    print(f"|---|---|---|---|---|")
    for r in RESULTS:
        rng = ("—" if r["median"] is None
               else f"{fmt(r['min'], r['unit'])} – {fmt(r['max'], r['unit'])}")
        print(f"| {r['name']} | {fmt(r['median'], r['unit'])} | {rng} | "
              f"{r['n']} | {r['note']} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="deterministic measurements only; no model calls")
    parser.add_argument("--repeats", type=int, default=3,
                        help="samples per figure (default 3, median reported)")
    parser.add_argument("--markdown", action="store_true",
                        help="also print a markdown table for the docs")
    parser.add_argument("--advisor", action="store_true",
                        help="also measure the beam's internals (slow)")
    parser.add_argument("--json", default=None, help="write raw results to this path")
    args = parser.parse_args()

    print(f"model: {active_model()}")
    print(f"repeats: {args.repeats} (median reported)")

    bench_deterministic(args.repeats)
    if not args.quick:
        bench_turns(args.repeats)
        if args.advisor:
            bench_advisor(args.repeats)

    if args.markdown:
        emit_markdown()
    if args.json:
        Path(args.json).write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
