"""Leak scanner: refuse to publish a tree that contains real-world data.

The demo build is the only thing that ever becomes a public repository. Its rule
is absolute — 100% synthetic property data, no real address, no real HOA, no real
licensed contractor, no key. This script is the gate that enforces it.

It is written *before* there is anything to leak, on purpose. Run it against the
current (pre-split) tree and it SHOULD report hits: that is the proof the
tripwires actually fire. Once the split lands, run it against `dist/public-demo`
and it must report none.

    python scripts/audit_public.py .                 # expect hits today
    python scripts/audit_public.py ../shf-public     # must be clean
    python scripts/audit_public.py . --json          # machine-readable

Exit codes:  0 = clean,  1 = leaks found,  2 = bad usage.

Design note: this is an ALLOWLIST-adjacent control, not the primary one. The
primary control is that `data/real/` is never copied into the public tree at all.
This script catches the case where a real string got pasted somewhere else — a
docstring, a README example, an eval fixture, a committed transcript.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# --- What counts as a leak -----------------------------------------------------
# Each rule is (name, compiled regex, why it matters). Keep the "why" filled in:
# when this script fails a build at 2am, the message is the whole user interface.
#
# Patterns are deliberately literal rather than clever. A regex that tries to
# detect "any street address" would false-positive on every synthetic address in
# the demo corpus, and a scanner that cries wolf gets disabled.
RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "real-street-address",
        re.compile(r"\b5143\s+White\s+Rose\s+(St|Street)\b", re.I),
        "the project owner's real home address",
    ),
    (
        "real-jurisdiction",
        re.compile(r"\bBonney\s+Lake\b", re.I),
        "the real WA city the full build is scoped to",
    ),
    (
        "real-hoa",
        re.compile(r"\bCascade\s+Vista\b", re.I),
        "HOA name tied to the real property",
    ),
    (
        "real-builder",
        re.compile(
            r"\b(Soundbuilt|MainVue|Richmond\s+American|D\.?R\.?\s*Horton"
            r"|Garrette\s+Custom|Tri\s*Pointe)\b",
            re.I,
        ),
        "a real homebuilder whose plans are full-build-only",
    ),
    (
        "real-home-id",
        re.compile(r"\bhome-wa-\d{3}\b"),
        "a full-build home_id; demo ids must match demo-NNN",
    ),
    (
        "real-doc-source",
        re.compile(r"\b(ecode360\.com|leg\.wa\.gov|app\.leg\.wa\.gov"
                   r"|codepublishing\.com|library\.municode\.com)\b", re.I),
        "a real municipal/statutory source — full build only",
    ),
    (
        "real-parcel-portal",
        re.compile(r"\b(atip\.piercecountywa\.gov|piercecountywa\.gov)\b", re.I),
        "real county parcel portal — full build only",
    ),
    (
        "lni-license-number",
        re.compile(r"\b[A-Z]{5,6}[A-Z*]{2}\d{3}[A-Z0-9]{2}\b"),
        "looks like a WA L&I contractor license number",
    ),
    (
        "real-data-path",
        # A path INTO the full tree, not the bare directory name.
        #
        # This used to match `data/full` anywhere, and the first real release
        # attempt was refused over `config.py` and `.gitignore` — files whose job
        # is to NAME both profiles. That is the mechanism, not a leak. A scanner
        # that cries wolf on its own architecture is a scanner someone eventually
        # switches off, which is worse than not having one.
        re.compile(r"\bdata[/\\](full|real)[/\\]\w|\bstate[/\\]full[/\\]\w"),
        "a path INTO the full-build data tree",
    ),
    (
        "api-key-shape",
        re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{30,}"
                   r"|sk-or-v1-[A-Za-z0-9]{16,})\b"),
        "a live API key",
    ),
]

# Files that are allowed to mention the tripwires, because their whole job is to
# define or document them. Anything not on this list gets scanned.
EXEMPT_NAMES = {"audit_public.py", "build_public.py", "public_allowlist.txt"}

# Directories never worth walking. Skipped for speed, not for safety — none of
# these are ever copied into a public tree.
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules",
    ".vite", "dist", ".cache", ".pytest_cache", ".idea", ".vscode",
    "memory/models",
}

# Binary and generated files we cannot meaningfully grep.
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".pdf", ".zip",
    ".db", ".sqlite", ".sqlite3", ".bin", ".onnx", ".woff", ".woff2",
    ".pyc", ".so", ".dll", ".lock",
}

MAX_BYTES = 4 * 1024 * 1024  # skip anything larger; corpora are small


def _should_skip(path: Path, root: Path) -> bool:
    parts = set(path.relative_to(root).parts)
    if parts & SKIP_DIRS:
        return True
    # SKIP_DIRS entries containing a separator (e.g. "memory/models")
    rel = path.relative_to(root).as_posix()
    if any("/" in d and rel.startswith(d + "/") for d in SKIP_DIRS):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    if path.name in EXEMPT_NAMES:
        return True
    return False


def scan(root: Path) -> list[dict]:
    """Walk `root` and return one finding per matching line."""
    findings: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _should_skip(path, root):
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            # errors="replace" so one odd byte cannot hide the rest of a file.
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pattern, why in RULES:
                match = pattern.search(line)
                if match:
                    findings.append({
                        "rule": name,
                        "why": why,
                        "file": path.relative_to(root).as_posix(),
                        "line": lineno,
                        "match": match.group(0)[:80],
                        "context": line.strip()[:160],
                    })
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refuse to publish a tree containing real-world data.")
    parser.add_argument("root", help="directory to scan")
    parser.add_argument("--json", action="store_true",
                        help="emit findings as JSON instead of text")
    parser.add_argument("--expect-hits", action="store_true",
                        help="invert the exit code: succeed only if leaks ARE "
                             "found. Used to prove the tripwires still fire.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    findings = scan(root)

    if args.json:
        print(json.dumps({"root": str(root), "count": len(findings),
                          "findings": findings}, indent=2))
    else:
        print(f"Scanned {root}")
        if not findings:
            print("CLEAN — no real-world data found.")
        else:
            by_rule: dict[str, list[dict]] = {}
            for f in findings:
                by_rule.setdefault(f["rule"], []).append(f)
            print(f"\n{len(findings)} finding(s) across {len(by_rule)} rule(s):\n")
            for rule, items in sorted(by_rule.items()):
                print(f"  [{rule}] {items[0]['why']} — {len(items)} hit(s)")
                for item in items[:5]:
                    print(f"      {item['file']}:{item['line']}  {item['match']!r}")
                if len(items) > 5:
                    print(f"      ... and {len(items) - 5} more")
                print()

    if args.expect_hits:
        if findings:
            print("OK — tripwires fired as expected.")
            return 0
        print("FAIL — expected to find real data here and found none. "
              "The rules may be broken.", file=sys.stderr)
        return 1

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
