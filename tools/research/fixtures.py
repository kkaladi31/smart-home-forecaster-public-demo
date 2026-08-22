"""A frozen snapshot of web-search results, for runs that must be reproducible.

**This does not replace live research — it is off by default.** Live evidence is
the Researcher's whole purpose. Fixtures exist for the two cases where the live
web is a liability rather than a feature: a graded run, which must give the same
answer to a grader months from now, and a recording, where a scrape that
rate-limits mid-take ruins the take.

`ddgs` scrapes DuckDuckGo's HTML endpoints — there is no free official API — and
`providers._UNAVAILABLE` exists precisely because that gets blocked in practice.
Depending on it at the moment someone presses record is depending on a scraper
staying unblocked.

**Where it intercepts, and why there.** At `providers.search`, the single network
boundary. Fixture results then flow through the real dedupe, the real injection
screen, the real cross-encoder ranking and the real pack builder — so a fixture
run exercises everything except the network. Freezing the finished evidence pack
instead would have been easier and would have tested nothing.

**On the synthetic-data rule.** These fixtures contain real public URLs and real
page snippets. That follows the precedent already set for the demo's video
fixtures: a public web page is public content, not the home / property / HOA /
contractor / utility data the 100%-synthetic rule enumerates. The demo already
answers questions from live manufacturer and .gov pages; this only freezes which
ones. Nothing here describes a real home.

Refresh with `python scripts/capture_research_fixtures.py`.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import config
from tools.research.providers import Result

PROVIDER = "fixtures"
_CACHE: dict[str, dict] | None = None
_CACHE_PATH: Path | None = None
_LOCK = threading.Lock()

# A query must overlap a fixture by this fraction of its own words to match.
# Exact-match keying alone is too brittle: `plan_queries` appends home context and
# an intent qualifier, so the string that reaches a provider is not the string a
# human typed, and it shifts whenever the home profile or the qualifier table
# changes. Overlap keeps a snapshot useful across those edits instead of silently
# falling through to "no evidence" — which looks exactly like a working search
# that found nothing.
MIN_OVERLAP = 0.6

_WORD = re.compile(r"[a-z0-9]+")
# Words that carry no retrieval signal. Dropped before overlap so a query does not
# match on "how", "the" and "my".
_STOP = frozenset("""a an and are as at be by can do does for from how i in is it
my of on or should that the this to what when where which who will with you your""".split())


def path() -> Path:
    return config.data_root() / "research_fixtures.json"


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def _load() -> dict[str, dict]:
    """Read the snapshot once per path. Reloads if the data root changes."""
    global _CACHE, _CACHE_PATH
    with _LOCK:
        current = path()
        if _CACHE is not None and _CACHE_PATH == current:
            return _CACHE
        try:
            data = json.loads(current.read_text(encoding="utf-8"))
            entries = data.get("queries", {}) if isinstance(data, dict) else {}
        except (OSError, ValueError):
            entries = {}
        _CACHE, _CACHE_PATH = entries, current
        return _CACHE


def reset() -> None:
    """Drop the in-process cache (tests, and after a re-capture)."""
    global _CACHE, _CACHE_PATH
    with _LOCK:
        _CACHE, _CACHE_PATH = None, None


def available() -> bool:
    return bool(_load())


def match(query: str) -> tuple[str | None, float]:
    """Best fixture key for a query, and how well it overlaps."""
    entries = _load()
    if not entries:
        return None, 0.0
    key = " ".join((query or "").lower().split())
    if key in entries:
        return key, 1.0

    wanted = _tokens(query)
    if not wanted:
        return None, 0.0
    best, best_score = None, 0.0
    for candidate in entries:
        overlap = len(wanted & _tokens(candidate)) / len(wanted)
        if overlap > best_score:
            best, best_score = candidate, overlap
    return (best, best_score) if best_score >= MIN_OVERLAP else (None, best_score)


def search(query: str, max_results: int = 6) -> list[Result]:
    """Serve one query from the snapshot.

    Raises `LookupError` when nothing matches, rather than returning an empty
    list. An empty list is indistinguishable from a working search that found
    nothing, and would let a stale snapshot quietly strip every citation out of a
    graded run while every case still reported "ok".
    """
    key, score = match(query)
    if not key:
        raise LookupError(
            f"no research fixture matches {query[:70]!r} (best overlap {score:.2f}). "
            "Re-capture with scripts/capture_research_fixtures.py")

    hits = _load()[key].get("results", [])[:max_results]
    return [
        Result(
            title=h.get("title", ""),
            url=h.get("url", ""),
            snippet=h.get("snippet", ""),
            content=h.get("content", ""),
            provider=PROVIDER,
            rank=i,
            extra={"fixture_key": key, "overlap": round(score, 2),
                   "captured": _load()[key].get("captured", "")},
        )
        for i, h in enumerate(hits)
    ]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    entries = _load()
    print(f"fixture file : {path()}")
    print(f"queries      : {len(entries)}\n")
    for key, entry in list(entries.items())[:8]:
        print(f"  {len(entry.get('results', [])):>2} results  {key[:72]}")

    if entries:
        probe = "How do I hang a 20 lb mirror on drywall installation instructions manufacturer"
        key, score = match(probe)
        print(f"\nprobe overlap {score:.2f} -> {key[:70] if key else '(no match)'}")
