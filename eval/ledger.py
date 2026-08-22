"""A per-case record of evaluation evidence, so a rate limit costs one case
instead of a whole suite.

**Why this exists.** The agent suite needs seventeen consecutive LLM turns
against a free-tier model on a shared provider pool. Measured across three
attempts on 2026-08-16, one died at the first case, one reached the last, and
one died at the twelfth. Every failure was a transient 429 rather than quota
exhaustion — but under the old all-or-nothing design each one discarded evidence
for the cases that had already passed, and overwrote `results.md` with a partial
record in the process.

The capstone's requirement is that the **artifact runs on free tokens**. It is
not that seventeen cases complete inside one process. Conflating those made a
provider's shared-pool pressure into a project blocker.

So evidence accumulates per case, and the gate becomes *"every case has a
passing record at the current commit"* rather than *"one process survived all of
them"*.

**The honesty constraint, which is the whole design.** A pass is evidence about
one revision of the code. A case green at commit X says nothing about commit Y
if the change touched that path, and a ledger that quietly counted old passes
would be worse than no ledger — it would manufacture a green suite out of stale
facts. So every entry records the commit it was earned at, and anything not
matching HEAD is reported as **stale**, named, with its commit shown. Deciding
whether a stale pass still applies is a judgement, and it stays with the person
reading it rather than being made silently here.

A run against a dirty working tree is attributed to no commit at all and is
always stale: there is no revision to name.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "ledger.json"

CURRENT = "current"   # earned at HEAD, with a clean tree
STALE = "stale"       # earned at a different commit, or with uncommitted changes
MISSING = "missing"   # never recorded


def _git(*args: str) -> str | None:
    """Raw stdout, with only the trailing newline removed.

    NOT `.strip()`. `git status --porcelain` encodes the status in the first two
    columns, so a modified-but-unstaged file is `" M path"` with a LEADING space.
    Stripping the whole output eats that space on the first line only, shifting it
    one character left — so `line[3:]` returned `"val/ledger.json"` instead of
    `"eval/ledger.json"`, the exclusion prefix never matched, and the first line
    always counted as a real change. Every run would have been recorded as
    `dirty:` forever, meaning every case was permanently stale and the ledger
    silently did nothing.
    """
    try:
        out = subprocess.run(("git", *args), cwd=LEDGER.parent.parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.rstrip("\n") if out.returncode == 0 else None
    except Exception:
        return None


def revision() -> str:
    """The commit this code is at, or `dirty:<sha>` / `unknown`.

    A dirty tree gets its own marker rather than borrowing the commit's name.
    Recording uncommitted work under the last commit's SHA is how a ledger
    starts lying: the next person sees evidence attributed to a revision that
    never contained the code that produced it.
    """
    sha = (_git("rev-parse", "--short", "HEAD") or "").strip()
    if not sha:
        return "unknown"
    dirty = _git("status", "--porcelain")
    # Generated artifacts are the suite's own output. Treating them as
    # meaningful changes would mark every run dirty by virtue of having run.
    if dirty:
        meaningful = [
            line for line in dirty.splitlines()
            if not line[3:].startswith(("eval/results.md", "eval/ledger.json",
                                        "eval/transcripts/"))
        ]
        if meaningful:
            return f"dirty:{sha}"
    return sha


def load() -> dict:
    if not LEDGER.exists():
        return {}
    try:
        data = json.loads(LEDGER.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def record(case_id: str, *, passed: bool, model: str, seconds: float,
           detail: str = "", rev: str | None = None) -> None:
    """Store the outcome of one case. Skipped cases must NOT be recorded —
    "we could not test it" is the absence of evidence, not evidence."""
    data = load()
    data[case_id] = {
        "passed": bool(passed),
        "model": model,
        "revision": rev or revision(),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(seconds, 1),
        "detail": detail[:300],
    }
    LEDGER.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status(case_id: str, *, rev: str | None = None) -> tuple[str, dict | None]:
    """(CURRENT | STALE | MISSING, entry)."""
    entry = load().get(case_id)
    if not entry:
        return MISSING, None
    rev = rev or revision()
    if entry.get("revision") == rev and not rev.startswith(("dirty:", "unknown")):
        return CURRENT, entry
    return STALE, entry


# Tool cases call no model at all and are recorded as "none". They count toward
# coverage on any run, because there is no model whose behaviour they depend on.
NO_MODEL = "none"


def coverage(case_ids: list[str], *, rev: str | None = None,
             model: str | None = None) -> dict:
    """How much of the suite is verified at this revision, ON THE GRADED MODEL.

    `green` requires three things: the case passed, at this revision, **on the
    model the artifact actually ships with**. The third is not pedantry. Without
    it, one `--model anthropic/claude-sonnet-5` run would turn the whole suite
    green and the summary would read as proof that the free-tier deliverable
    works — which is the precise claim it would not have tested. Those passes are
    real evidence about the *code*, so they are reported, but under their own
    heading and never counted as artifact coverage.

    A stale pass and a stale failure are both reported as stale: both need
    re-running before they mean anything about this revision.
    """
    rev = rev or revision()
    if model is None:
        from config import graded_model

        model = graded_model()
    out = {"revision": rev, "model": model, "green": [], "other_model": [],
           "failing": [], "stale": [], "missing": []}
    for case_id in case_ids:
        state, entry = status(case_id, rev=rev)
        if state == MISSING:
            out["missing"].append(case_id)
        elif state == STALE:
            out["stale"].append((case_id, (entry or {}).get("revision", "?"),
                                 (entry or {}).get("passed")))
        elif entry and entry["passed"]:
            ran_on = entry.get("model", "?")
            if ran_on in (model, NO_MODEL):
                out["green"].append(case_id)
            else:
                out["other_model"].append((case_id, ran_on))
        else:
            out["failing"].append(case_id)
    return out


def render(case_ids: list[str], *, rev: str | None = None,
           model: str | None = None) -> list[str]:
    """Markdown lines summarising coverage, for `results.md`."""
    cov = coverage(case_ids, rev=rev, model=model)
    total = len(case_ids)
    lines = [
        "## Coverage ledger",
        "",
        "Evidence accrues per case (`eval/ledger.json`), so a provider rate limit "
        "costs one case rather than the suite. A pass counts only at the commit it "
        "was earned at, and only on the model the artifact ships with.",
        "",
        f"- **Revision:** `{cov['revision']}`",
        f"- **Graded model:** `{cov['model']}`",
        f"- **Green at this revision, on the graded model:** {len(cov['green'])}/{total}",
    ]
    if cov["other_model"]:
        detail = ", ".join(f"{cid} (`{m}`)" for cid, m in cov["other_model"])
        lines += [
            f"- **Passed on a DIFFERENT model — evidence about the code, not the "
            f"artifact:** {detail}",
        ]
    if cov["failing"]:
        lines.append(f"- **Failing at this revision:** {', '.join(cov['failing'])}")
    if cov["stale"]:
        detail = ", ".join(f"{cid} ({'pass' if ok else 'fail'} @ `{r}`)"
                           for cid, r, ok in cov["stale"])
        lines.append(f"- **Stale — must be re-run:** {detail}")
    if cov["missing"]:
        lines.append(f"- **Never run:** {', '.join(cov['missing'])}")
    if cov["revision"].startswith("dirty:"):
        lines += ["", "> The working tree had uncommitted changes, so this run is "
                  "attributed to no commit and everything counts as stale."]
    lines.append("")
    return lines


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from eval.cases import AGENT_CASES, TOOL_CASES

    ids = [c["id"] for c in TOOL_CASES + AGENT_CASES]
    print(f"revision: {revision()}\n")
    for line in render(ids):
        print(line)
