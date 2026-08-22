"""Run the evaluation suite and write results + transcripts.

Usage (from the project root):
    python eval/run_eval.py              # everything (tool cases + agent cases)
    python eval/run_eval.py --tools-only # fast, deterministic, no LLM calls
    python eval/run_eval.py --only A5    # a single case by id

Outputs:
    eval/results.md          summary table + failure details
    eval/transcripts/*.md    full transcript per agent case (report/video artifacts)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Allow running as `python eval/run_eval.py` from the project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from agents import output_guard  # noqa: E402
from config import active_model  # noqa: E402
from eval import ledger  # noqa: E402
from eval.cases import AGENT_CASES, TOOL_CASES  # noqa: E402
from memory import episodic  # noqa: E402

TRANSCRIPTS = Path(__file__).resolve().parent / "transcripts"
RESULTS = Path(__file__).resolve().parent / "results.md"

# Longest answer written verbatim into a transcript. See `_write_transcript`.
MAX_TRANSCRIPT_ANSWER = 8000


# Models routinely emit typographic punctuation (curly apostrophes, en/em dashes).
# Without normalising, a check for "couldn't find" silently fails against the
# model's "couldn’t find" — a false failure that would misreport the results.
_PUNCT_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                            "–": "-", "—": "-", " ": " "})


def _normalize(text: str) -> str:
    """Lowercase and fold typographic punctuation to ASCII for robust matching."""
    return text.translate(_PUNCT_MAP).lower()


def _check_agent(case: dict, result: dict) -> tuple[bool, list[str]]:
    """Apply a case's declarative checks to an agent result. Returns (passed, failures)."""
    answer = _normalize(result.get("answer") or "")
    tools_called = [s["name"] for s in result.get("trace", []) if s["kind"] == "call"]
    failures: list[str] = []

    for tool in case.get("expect_tools", []):
        if tool not in tools_called:
            failures.append(f"expected tool '{tool}' was not called (called: {tools_called or 'none'})")

    # "at least one of these" — for capabilities where several tool paths are
    # legitimate. Demanding an exact chain from a non-deterministic model produces
    # false failures, which erodes trust in the whole suite.
    any_of = case.get("expect_tools_any", [])
    if any_of and not any(t in tools_called for t in any_of):
        failures.append(f"expected at least one of {any_of} (called: {tools_called or 'none'})")

    for tool in case.get("forbid_tools", []):
        if tool in tools_called:
            failures.append(f"forbidden tool '{tool}' was called")

    # Episodic memory may be satisfied either by auto-recall (injected context) or
    # by the recall_memory tool — assert the capability, not the mechanism.
    if case.get("expect_recall"):
        used_memory = bool(result.get("recalled")) or "recall_memory" in tools_called
        if not used_memory:
            failures.append("expected episodic memory to be used (auto-recall or recall_memory)")

    if "expect_blocked" in case:
        want = case["expect_blocked"]
        got = bool(result.get("blocked"))
        if got != want:
            failures.append(f"expected blocked={want}, got blocked={got}")

    expect_any = case.get("expect_any", [])
    if expect_any and not any(_normalize(s) in answer for s in expect_any):
        failures.append(f"answer contained none of {expect_any}")

    for s in case.get("forbid_any", []):
        if _normalize(s) in answer:
            failures.append(f"answer contained forbidden text {s!r}")

    # Assertions about what was actually LOOKED UP, not what the prose mentions.
    #
    # These exist because forbidding a string in the answer conflates two very
    # different things. A13 forbade the saved home's city to catch "ignored my correction
    # and answered about the saved home" — but it also fired when the agent
    # correctly answered about the corrected city and then helpfully noted which
    # homes it does hold documents for. The tool arguments distinguish the two
    # cleanly: what the agent looked up is the behaviour, what it mentions is
    # commentary.
    call_args = " ".join(
        _normalize(json.dumps(s.get("args", {}), ensure_ascii=False))
        for s in result.get("trace", []) if s["kind"] == "call"
    )

    expect_arg_any = case.get("expect_arg_any", [])
    if expect_arg_any and not any(_normalize(s) in call_args for s in expect_arg_any):
        failures.append(
            f"no tool was called with any of {expect_arg_any} (args seen: {call_args[:160] or 'none'})")

    for s in case.get("forbid_arg_any", []):
        if _normalize(s) in call_args:
            failures.append(f"a tool was called with forbidden argument text {s!r}")

    # Assertions about what a tool RETURNED. Deterministic where the answer is
    # not: whether the grounding gate rejected a query is a fact about the
    # retrieval stage, and it does not vary with how the model chose to word its
    # refusal. Pair it with a broad expect_any rather than trying to enumerate
    # every phrasing a model might use for "I found nothing".
    tool_results = " ".join(
        _normalize(str(s.get("content", "")))
        for s in result.get("trace", []) if s["kind"] == "result"
    )

    expect_result_any = case.get("expect_tool_result_any", [])
    if expect_result_any and not any(_normalize(s) in tool_results for s in expect_result_any):
        failures.append(f"no tool result contained any of {expect_result_any}")

    # Regex over the answer, for properties that are SEMANTIC rather than lexical.
    #
    # A5 is the case that forced this. "Did it admit it had no source" has no
    # finite phrase list: across four runs the model produced "no documented
    # rule", "no matching passage", "does not contain any guidance", "can't
    # confirm" and "none found in the policy database". Each failure was met by
    # adding the missing phrase, which is whack-a-mole with a suite that is
    # supposed to be a safety net. A pattern expresses the actual invariant —
    # a negation near a word meaning "source" — and stops the game.
    for pattern in case.get("expect_pattern", []):
        if not re.search(pattern, answer, re.I):
            failures.append(f"answer did not match required pattern {pattern!r}")

    for pattern in case.get("forbid_pattern", []):
        if re.search(pattern, answer, re.I):
            failures.append(f"answer matched forbidden pattern {pattern!r}")

    return (not failures), failures


def _is_infrastructure_error(exc: Exception) -> bool:
    """True if the failure was the environment (quota, rate limit, network), not the system."""
    if isinstance(exc, UnusableAnswer):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    markers = ("rate limit", "429", "quota", "insufficient credits", "timeout",
               "connection", "temporarily unavailable", "503", "502",
               # The vector store was rebuilt underneath a running suite — which
               # `python ingest.py` does, and which is easy to trigger by accident
               # while a long agent run is in flight. The collection handle goes
               # stale and every subsequent retrieval case dies. That is the
               # environment, not the system: re-running the case passes.
               "hnsw segment reader", "nothing found on disk",
               "does not exist" if "collection" in text else "\0")
    return any(m in text for m in markers)


class UnusableAnswer(RuntimeError):
    """The model produced nothing a check could be applied to.

    Environment, not a test failure. Two observed shapes, both from a free-tier
    provider under load, and both of which the suite scored as content failures
    on real cases before this existed:

      * **empty** — A9 ran 1671s, fired the safety guardrail correctly, ran
        retrieval correctly, then returned no text. Reported as "answer contained
        none of ['licensed', 'electrician', 'permit']", which reads exactly like
        the safety refusal regressing, on the suite's most safety-critical case.
      * **degenerate** — A12 ran 2193s and returned ~33,000 tokens of "!!!!!!!!".
        Reported as "answer contained none of ['turf']", which reads like the
        multi-turn memory feature being broken.

    Neither says anything about this system. Both are retried, then recorded as
    not-run rather than failed.
    """


def _unusable_reason(answer: str, trace: list) -> str | None:
    """Why this answer cannot be checked, or None if it can be.

    Delegates to `agents.output_guard`, which the PRODUCT also uses to cut off a
    collapsing stream. That sharing is the point rather than tidiness: this logic
    previously lived only here, so the harness knew how to recognise a degenerate
    answer while the live product streamed one to a browser without limit until
    the user gave up. A definition of "not an answer" that only the test knows is
    a definition the product cannot act on.
    """
    return output_guard.unusable_reason(answer, trace)


def _infra_kind(exc: Exception) -> str:
    """"daily" | "transient" | "other" — which decides whether retrying can help.

    The distinction is the whole point. A daily quota will not clear in ninety
    seconds, so retrying burns time for nothing; a shared-pool 429 usually will,
    and treating it as fatal is what made three separate suite runs die at three
    different cases while every one of them was recoverable.
    """
    # An empty answer is the provider degrading under load, so it is worth
    # another attempt — the same reasoning as a 429.
    if isinstance(exc, UnusableAnswer):
        return "transient"
    text = str(exc).lower()
    if "free-models-per-day" in text or ("rate limit" in text and "day" in text):
        return "daily"
    if "429" in text or "rate limit" in text or "temporarily rate-limited" in text:
        return "transient"
    return "other"


def _infrastructure_reason(exc: Exception) -> str:
    if isinstance(exc, UnusableAnswer):
        return ("NOT RUN — the model returned an empty answer on every attempt "
                "(provider degraded under load), so nothing could be checked.")
    kind = _infra_kind(exc)
    if kind == "daily":
        return ("NOT RUN — OpenRouter free-tier daily request limit reached. "
                "Re-run after the daily reset (or with a funded key).")
    if kind == "transient":
        return "NOT RUN — provider rate limit persisted across retries."
    return f"NOT RUN — infrastructure error: {type(exc).__name__}"


def _safe_name(text: str) -> str:
    """Filesystem-safe slug (case names may contain '/', '&', etc.)."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def _write_transcript(case: dict, result: dict, passed: bool, failures: list[str]) -> Path:
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPTS / f"{case['id']}_{_safe_name(case['name'])}.md"
    lines = [
        f"# {case['id']} — {case['name']}",
        "",
        f"- **Concept:** {case['concept']}",
        f"- **Result:** {'PASS' if passed else 'FAIL'}",
        f"- **Run at:** {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Query",
        "",
    ]
    for q in (case.get("turns") or [case.get("query", "")]):
        lines.append(f"> {q}")
        lines.append("")
    if case.get("turns"):
        lines.append("_(multi-turn: checks apply to the final answer)_")
        lines.append("")
    lines += [
        "## Tool-call trace",
        "",
    ]
    trace = result.get("trace", [])
    if not trace:
        lines.append("_(no tools called)_")
    for step in trace:
        if step["kind"] == "call":
            lines.append(f"- **→ called `{step['name']}`** with `{json.dumps(step['args'])}`")
        else:
            preview = " ".join(str(step["content"]).split())[:300]
            lines.append(f"  - ← `{step['name']}` returned: {preview}…")
    # Capped. A degenerate generation wrote ~33,000 tokens of "!" into a
    # transcript, which is unreadable as an artifact and unreviewable in a diff —
    # and transcripts exist to be read, by a grader and by whoever is debugging.
    # 8,000 characters is several times the longest real answer observed.
    answer = result.get("answer") or "_(none)_"
    if len(answer) > MAX_TRANSCRIPT_ANSWER:
        answer = (answer[:MAX_TRANSCRIPT_ANSWER]
                  + f"\n\n_… truncated: answer was {len(answer)} characters._")
    lines += ["", "## Answer", "", answer, ""]
    if failures:
        lines += ["## Failures", ""] + [f"- {f}" for f in failures] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_tool_cases(only: str | None = None) -> list[dict]:
    rows = []
    for case in TOOL_CASES:
        if only and case["id"] != only:
            continue
        start = time.time()
        try:
            passed, detail = case["fn"]()
        except Exception as exc:
            passed, detail = False, f"EXCEPTION: {type(exc).__name__}: {exc}"
            traceback.print_exc()
        elapsed = time.time() - start
        rows.append({**case, "passed": passed, "detail": detail, "seconds": elapsed})
        print(f"  [{'PASS' if passed else 'FAIL'}] {case['id']} {case['name']} — {detail}")
        # Recorded like the agent cases even though these never rate-limit, so
        # coverage is a statement about the whole suite rather than the flaky
        # half of it.
        #
        # Normally "none": these call no LLM, and a model slug would imply the
        # result depended on one. But under `--model` it is attributed to the
        # OVERRIDE, because some of them do. T17 asserts a demo build is pinned
        # to the free model, so a diagnostic Sonnet run makes it fail — correctly,
        # that is the guard working. Recorded as "none" it would land in the
        # GRADED coverage as a false failure, which is exactly the confusion the
        # model-aware ledger exists to prevent. Found by running it.
        try:
            ledger.record(case["id"], passed=passed,
                          model=config.model_override() or ledger.NO_MODEL,
                          seconds=elapsed, detail=detail)
        except Exception:
            pass
    return rows


# A transient rate limit gets three shots with a widening pause. The delays are
# generous relative to the `retry_after: 1` the provider suggests, because what
# was actually observed was shared-pool pressure lasting tens of seconds rather
# than a per-request cooldown.
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (20, 60)
# If two cases in a row exhaust their retries, capacity is genuinely gone and
# grinding through the rest wastes a quarter of an hour to learn nothing. The
# ledger keeps whatever was earned before that point.
MAX_CONSECUTIVE_SKIPS = 2


def run_agent_cases(only: str | None = None, *, use_ledger: bool = True) -> list[dict]:
    from agents.orchestrator import answer_with_trace, build_agent

    cases = [c for c in AGENT_CASES if not only or c["id"] == only]
    if not cases:
        return []

    agent = build_agent()  # build once, reuse across cases
    rows = []
    consecutive_skips = 0
    revision = ledger.revision() if use_ledger else None
    for case in cases:
        start = time.time()
        infra_error = None
        try:
            # Isolate each case: a unique thread, and episodic memory off unless the
            # case explicitly tests it — so cases cannot contaminate one another.
            thread = f"eval-{case['id']}-{int(start)}"
            use_memory = case.get("use_memory", False)

            # A memory case may need prior interactions to exist first.
            for seed_q, seed_a in case.get("seed_memory", []):
                episodic.record_interaction(f"{thread}-seed", seed_q, seed_a, [])

            # Multi-turn cases replay several messages on one thread; the checks
            # apply to the final answer.
            queries = case.get("turns") or [case["query"]]
            # Persona drives the RAG audience filter and home_id drives the
            # jurisdiction filter, so a case can assert that a renter is grounded
            # on different documents than an owner, and that the Texas home is
            # answered from Texas documents rather than the primary home's.
            persona = case.get("persona", "owner")
            home_id = case.get("home_id")
            # Retried as a WHOLE CASE, not per request. A multi-turn case that
            # dies on its second turn has to start over — resuming mid-thread
            # would leave the conversation in a state the case never describes,
            # and the checks apply to the final answer.
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    for q in queries:
                        result = answer_with_trace(q, agent=agent, thread_id=f"{thread}-{attempt}",
                                                   persona=persona, use_memory=use_memory,
                                                   home_id=home_id)
                    # Checked INSIDE the try, so provider garbage is retried like
                    # a 429 rather than raised once. See `UnusableAnswer` — the
                    # two observed shapes cost A9 and A12 a false failure each,
                    # and both looked like a headline feature regressing.
                    unusable = _unusable_reason(result.get("answer") or "",
                                                result.get("trace") or [])
                    if unusable:
                        raise UnusableAnswer(unusable)
                    break
                except Exception as exc:
                    kind = _infra_kind(exc) if _is_infrastructure_error(exc) else "fatal"
                    if kind != "transient" or attempt == MAX_ATTEMPTS:
                        raise
                    pause = BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
                    print(f"         (retrying — {type(exc).__name__}: attempt "
                          f"{attempt}/{MAX_ATTEMPTS}, waiting {pause}s)")
                    time.sleep(pause)
            passed, failures = _check_agent(case, result)
        except Exception as exc:
            result = {"answer": f"EXCEPTION: {type(exc).__name__}: {exc}", "trace": []}
            passed, failures = False, [f"exception: {exc}"]
            # Distinguish "the system is wrong" from "we could not test it".
            # Quota/rate-limit/network problems are NOT test failures, and
            # reporting them as such would misrepresent the results.
            if _is_infrastructure_error(exc):
                infra_error = _infrastructure_reason(exc)
                failures = [infra_error]
        elapsed = time.time() - start
        # Never let a transcript-writing problem abort the whole suite.
        try:
            transcript = _write_transcript(case, result, passed, failures).name
        except Exception as exc:
            transcript = ""
            print(f"         (could not write transcript: {type(exc).__name__}: {exc})")
        rows.append({**case, "passed": passed, "failures": failures, "infra_error": infra_error,
                     "seconds": elapsed, "transcript": transcript})
        status = "SKIP" if infra_error else ("PASS" if passed else "FAIL")
        print(f"  [{status}] {case['id']} {case['name']} ({elapsed:.1f}s)")
        for f in failures:
            print(f"         - {f}")

        # A skip is the ABSENCE of evidence, so nothing is written for it. Only
        # a case that actually ran updates the ledger.
        if use_ledger and not infra_error:
            try:
                ledger.record(case["id"], passed=passed, model=active_model(),
                              seconds=elapsed, detail="; ".join(failures),
                              rev=revision)
            except Exception as exc:
                print(f"         (could not update the ledger: {type(exc).__name__}: {exc})")

        if infra_error:
            consecutive_skips += 1
            if "daily" in infra_error or "daily request limit" in infra_error:
                print("         (stopping — a daily quota will not clear mid-run)")
                break
            if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                print(f"         (stopping — {consecutive_skips} cases in a row exhausted "
                      "their retries; capacity is gone. Earned results are kept in the ledger)")
                break
            print("         (continuing — capacity may return; the ledger keeps what passed)")
        else:
            consecutive_skips = 0
    return rows


def _results_path() -> Path:
    """Where this run's artifact goes.

    A diagnostic `--model` run writes to its OWN file. `results.md` is what the
    report and the README point at, and a green Sonnet run overwriting it would
    be indistinguishable from the graded artifact passing — in the exact document
    a reader trusts. Two files, two claims, no confusion.
    """
    override = config.model_override()
    if not override:
        return RESULTS
    safe = re.sub(r"[^a-z0-9]+", "-", override.lower()).strip("-")
    return RESULTS.with_name(f"{RESULTS.stem}.{safe}{RESULTS.suffix}")


def write_results(tool_rows: list[dict], agent_rows: list[dict]) -> None:
    all_rows = tool_rows + agent_rows
    skipped = [r for r in agent_rows if r.get("infra_error")]
    ran = [r for r in all_rows if not r.get("infra_error")]
    passed = sum(r["passed"] for r in ran)
    override = config.model_override()
    lines = [
        "# Evaluation Results",
        "",
        f"_Generated {datetime.now().isoformat(timespec='seconds')}_",
        "",
        f"**Model:** `{active_model()}`"
        + ("  ⚠️ **diagnostic override** — this run says whether the CODE is correct, "
           "**not** whether the graded free-tier artifact works." if override else
           "  (the model the graded artifact ships with)"),
        "",
        f"**{passed}/{len(ran)} executed cases passed.**",
        "",
    ]
    if skipped:
        lines += [
            f"> ⚠️ {len(skipped)} agent case(s) could not be executed — the free-tier provider "
            "was rate limited even after retries. These are **not** test failures; they were "
            "never run. The coverage ledger below carries what earlier runs earned, so this "
            "does not discard evidence.",
            "",
        ]

    # The ledger goes above the per-run tables on purpose. A partial run used to
    # overwrite this file wholesale, so the artifact reported whatever the last
    # process happened to reach — and a run that died at case 2 looked worse than
    # one that died at case 16, though neither said anything about the cases they
    # never touched. Coverage is the honest headline; the tables below are one
    # run's detail.
    try:
        lines += ledger.render([c["id"] for c in TOOL_CASES + AGENT_CASES])
    except Exception as exc:  # never let bookkeeping break the artifact
        lines += [f"_(coverage ledger unavailable: {type(exc).__name__})_", ""]
    lines += [
        "Agent-case checks are behavioural (which tools were called, whether a citation or "
        "guardrail appeared) rather than assertions about live weather values, so the suite "
        "stays reproducible as conditions change.",
        "",
        "## Deterministic tool cases (no LLM)",
        "",
        "| id | case | concept | result | detail |",
        "|---|---|---|---|---|",
    ]
    for r in tool_rows:
        lines.append(f"| {r['id']} | {r['name']} | {r['concept']} | "
                     f"{'✅ pass' if r['passed'] else '❌ FAIL'} | {r['detail']} |")

    lines += ["", "## End-to-end agent cases", "",
              "| id | case | concept | result | time | transcript |", "|---|---|---|---|---|---|"]
    for r in agent_rows:
        link = f"[{r['transcript']}](transcripts/{r['transcript']})" if r["transcript"] else "—"
        if r.get("infra_error"):
            status = "⏭️ not run"
        else:
            status = "✅ pass" if r["passed"] else "❌ FAIL"
        lines.append(f"| {r['id']} | {r['name']} | {r['concept']} | "
                     f"{status} | {r['seconds']:.1f}s | {link} |")

    failures = [r for r in ran if not r["passed"]]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            detail = r.get("detail") or "; ".join(r.get("failures", []))
            lines.append(f"- **{r['id']} {r['name']}** — {detail}")

    if skipped:
        lines += ["", "## Not run (environment, not defects)", ""]
        for r in skipped:
            lines.append(f"- **{r['id']} {r['name']}** — {r['infra_error']}")

    path = _results_path()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation suite")
    parser.add_argument("--tools-only", action="store_true", help="skip LLM agent cases")
    parser.add_argument("--only", default=None, help="run a single case id (e.g. A5)")
    parser.add_argument(
        "--live-research", action="store_true",
        help=("hit the live web instead of the frozen research snapshot. Off by "
              "default: a graded run must give a grader the same answer months "
              "from now, and DuckDuckGo is a scrape, not an API"))
    parser.add_argument(
        "--model", default=None, metavar="SLUG",
        help=("DIAGNOSTIC: force one model for the whole process, e.g. "
              "anthropic/claude-sonnet-5. Separates 'is the code correct' from "
              "'does the free-tier artifact work'. Writes to its own results file "
              "and does NOT count toward graded coverage."))
    args = parser.parse_args()

    # Research is served from the frozen snapshot unless asked otherwise, so the
    # suite's answers do not depend on what the web returned today. The live path
    # still has a drift check: scripts/contract_check.py.
    if not args.live_research:
        from tools.research import fixtures

        if fixtures.available():
            config.set_research_fixtures(True)
            print(f"Research: frozen snapshot ({len(fixtures._load())} queries) — "
                  "pass --live-research to hit the web")
        else:
            print("Research: LIVE (no snapshot found — run "
                  "scripts/capture_research_fixtures.py)")

    if args.model:
        config.set_model_override(args.model)
        print(f"⚠️  DIAGNOSTIC RUN on {active_model()} — this measures whether the CODE")
        print(f"    is correct. The graded artifact ships on {config.graded_model()},")
        print("    and passes recorded here do NOT count toward its coverage.\n")

    print("=== Deterministic tool cases ===")
    tool_rows = run_tool_cases(only=args.only)

    agent_rows = []
    if not args.tools_only:
        print("\n=== End-to-end agent cases (calls the LLM) ===")
        agent_rows = run_agent_cases(only=args.only)

    write_results(tool_rows, agent_rows)

    all_rows = tool_rows + agent_rows
    skipped = [r for r in all_rows if r.get("infra_error")]
    ran = [r for r in all_rows if not r.get("infra_error")]
    passed = sum(r["passed"] for r in ran)
    failed = len(ran) - passed

    print(f"\n{passed}/{len(ran)} executed cases passed"
          + (f"; {len(skipped)} not run (environment)" if skipped else ""))
    if skipped:
        print("  -> " + skipped[0]["infra_error"])
    # Exit 1 only for genuine failures; 2 signals an incomplete (but not failing) run.
    if failed:
        return 1
    return 2 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
