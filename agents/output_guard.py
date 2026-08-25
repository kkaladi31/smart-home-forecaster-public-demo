"""Detect model output that is not an answer, and stop paying attention to it.

Small language models sometimes collapse into repeating one character. Observed
twice on this project, in two different places, and each time it was mistaken for
something else:

  * In the evaluation suite, a case returned ~33,000 tokens of ``!!!!!!!!`` and
    was scored as *"answer contained none of ['turf']"* — which reads as the
    multi-turn memory feature being broken.
  * In the live product, the same collapse streamed to the browser **without
    limit**. The user saw a wall of exclamation marks that did not stop, on a turn
    where they had just typed an SSN. Nothing in the streaming path bounded it.

The eval learned to retry it; the product had no equivalent, because the check
lived inside the test harness. Hence this module: **one definition, used by
both**, so the harness and the thing it tests cannot disagree about what counts
as an answer.

Thresholds sit deliberately far from anything real prose reaches. A legitimate
reply is overwhelmingly letters and never 60% one character; the observed
collapse measured zero letters and ~100% one character. A narrow margin here
would risk discarding a real answer, which is the one mistake this must not make.
"""
from __future__ import annotations

import re
from collections import Counter

# Below this length, nothing is judged. Short replies are legitimately
# punctuation-heavy ("No.", "Yes — see 4.2.") and there is not enough text to
# distinguish terse from broken.
MIN_LENGTH = 200

# A real answer is overwhelmingly letters. Markdown, code and tables push the
# ratio down, never remotely this far.
MIN_LETTER_RATIO = 0.10

# One character dominating this much of a long answer is a stuck decoder.
MAX_CHAR_SHARE = 0.60

# The low-letter test needs a second condition, because a legitimate answer CAN
# be letter-poor: a cost breakdown that is mostly digits, pipes and dashes would
# otherwise be cut off mid-stream. A stuck decoder emits one or two distinct
# characters; a table emits many. Requiring both makes a false positive — killing
# a real answer in front of the user — very unlikely, while the observed collapse
# (zero letters, one distinct character) still trips it comfortably.
MAX_DISTINCT_WHEN_LETTER_POOR = 15

# Streaming is checked on a shorter fuse than a finished answer. By 400
# characters a collapse is unambiguous, and every character after that is one the
# user watches arrive.
STREAM_MIN_LENGTH = 400


def degenerate_reason(text: str) -> str | None:
    """Why this text is not prose, or None if it is fine."""
    body = (text or "").strip()
    if len(body) < MIN_LENGTH:
        return None

    counts = Counter(body)
    char, count = counts.most_common(1)[0]
    if count / len(body) > MAX_CHAR_SHARE:
        return (f"the character {char!r} is {count / len(body):.0%} of a "
                f"{len(body)}-character answer — repetition collapse")

    letters = sum(c.isalpha() for c in body)
    distinct = len(counts)
    if letters / len(body) < MIN_LETTER_RATIO and distinct < MAX_DISTINCT_WHEN_LETTER_POOR:
        return (f"{len(body)} characters, {letters} letters, only {distinct} "
                "distinct characters — degenerate generation, not prose")
    return None


def unusable_reason(answer: str, trace: list | None = None) -> str | None:
    """Why a FINISHED answer cannot be checked or shown, or None.

    An empty answer counts only when tools actually ran. With no tool calls the
    agent did nothing, which is a genuine defect worth surfacing as one rather
    than excusing as a provider hiccup.
    """
    body = (answer or "").strip()
    if not body:
        return (f"model returned an empty answer after {len(trace)} trace step(s)"
                if trace else None)
    return degenerate_reason(body)


def stream_is_degenerate(accumulated: str) -> str | None:
    """Cut-off check for a stream in flight.

    Same shape test on a shorter fuse. Called per token, so it stays O(n) on the
    accumulated text rather than doing anything clever — at the lengths involved
    that is microseconds, and a cheaper incremental counter would be one more
    thing to get wrong.
    """
    if len(accumulated) < STREAM_MIN_LENGTH:
        return None
    return degenerate_reason(accumulated)


USER_MESSAGE = (
    "The model started repeating itself and the answer was stopped. This is a "
    "known limitation of the free model under load rather than a problem with "
    "your question — please try again."
)


# Citation artefacts some models emit around tool results — 【ask_advisor†sources】,
# 【4:0†file】, and sometimes an entire URL wrapped in the same brackets. They are the
# model narrating its own retrieval, not a reference anyone can follow, and they
# render as noise in the middle of a sentence.
#
# Stripped deterministically rather than prompted away: this is a token-level habit
# of the model, and a prompt line asking it to stop is exactly the kind of
# instruction a free-tier model drops under a long context. Real citations in this
# system are markdown links, so nothing legitimate uses these brackets.
#
# Unbounded inside the brackets on purpose. The first version capped the contents
# at 120 characters and promptly missed a wrapped URL carrying a 90-character
# tracking parameter — the exact case it was written for.
_CITATION_ARTEFACT = re.compile(r"【[^】]*】")

# A URL that is not already the target of a markdown link.
_BARE_URL = re.compile(r"(?<![(\[])\bhttps?://[^\s<>()\[\]]+")

# One item of a bulleted source list.
_SOURCE_ITEM = re.compile(r"^(?P<bullet>\s*[-*]\s+)(?P<body>.*\S)\s*$")


def _domain_of(url: str) -> str:
    host = re.sub(r"^https?://", "", url).split("/")[0]
    return host[4:] if host.startswith("www.") else host


def _linkify_source_list(text: str) -> str:
    """Rewrite `- label - https://x` as `- [label](https://x)`.

    The prompt asks for markdown links and usually gets them; when it does not,
    the answer ends in a column of unclickable URLs. Doing it in code is safe in a
    way that asking the model again is not — a bare URL becoming a link to itself
    cannot be wrong, and the alternative is a references section nobody can
    follow. Lines that already contain a markdown link are left untouched, and so
    is any bullet without a URL, which is how document citations survive.
    """
    out = []
    for line in text.splitlines():
        match = _SOURCE_ITEM.match(line)
        if not match or "](http" in line:
            out.append(line)
            continue
        body = match.group("body")
        found = _BARE_URL.search(body)
        if not found:
            out.append(line)
            continue
        url = found.group(0).rstrip(".,;:")
        label = body[:found.start()].strip().rstrip("-–—:").strip()
        tail = body[found.start() + len(found.group(0)):].strip()
        label = label or _domain_of(url)
        out.append(f"{match.group('bullet')}[{label}]({url})" + (f" {tail}" if tail else ""))
    return "\n".join(out)


# A "source" that is really the model describing itself. Observed live: asked how
# it decides to bypass, the agent listed its own routing rules and closed with
# "Sources: System prompt" — then, challenged, denied having done it.
#
# The disclosure itself was harmless (these instructions are published, and hold
# no secrets) but the CITATION was not: sources in this system mean retrieved
# documents and tool results, and a line that cites the prompt dresses
# self-description up as evidence. The prompt now tells the model to describe its
# design in prose and cite nothing; this removes the line if it does it anyway.
_SELF_CITATION = re.compile(
    r"^\s*[-*]\s+(the\s+)?(system\s+prompt|my\s+(system\s+)?(prompt|instructions|training|"
    r"guidelines)|internal\s+(instructions|guidelines)|operational\s+guidelines)"
    r"\s*\.?\s*$", re.I)


def _drop_self_citations(text: str) -> str:
    """Remove source bullets that cite the model's own instructions."""
    return "\n".join(l for l in text.splitlines() if not _SELF_CITATION.match(l))


def clean_answer(text: str) -> str:
    """Strip citation artefacts and make a source list clickable."""
    if not text:
        return text
    cleaned = _CITATION_ARTEFACT.sub("", text)
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = _drop_self_citations(cleaned)
    return _linkify_source_list(cleaned)


_URL_IN_TEXT = re.compile(r"https?://[^\s\"'<>)\]]+")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _canonical(url: str) -> str:
    """A URL's identity for comparison: no scheme, no `www.`, no trailing slash.

    Deliberately loose on those three and strict on everything else. A model that
    copies a URL correctly but writes `http` for `https` should still match; a
    model that invents a plausible-looking path should not.
    """
    u = url.strip().rstrip(".,;:)\"'")
    u = re.sub(r"^https?://", "", u)
    if u.startswith("www."):
        u = u[4:]
    return u.rstrip("/").lower()


def collect_source_urls(trace: list | None) -> set[str]:
    """Every URL that a tool actually returned during this turn.

    The trace holds each tool result verbatim, so this is the authoritative set:
    if a URL is not in here, no tool produced it and the model composed it.
    """
    urls: set[str] = set()
    for step in trace or []:
        if step.get("kind") != "result":
            continue
        for found in _URL_IN_TEXT.findall(str(step.get("content", ""))):
            urls.add(_canonical(found))
    return urls


def enforce_links(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Demote any link the tools did not actually return to plain text.

    Citations in this system are supposed to be evidence, not decoration, and a
    link is the one part of an answer a reader will act on without checking. A
    model that half-remembers a URL produces something that looks authoritative
    and lands on a 404 — or worse, on a real page that says something else.

    So the rule is not "warn about" but "remove": an unverifiable link loses its
    href and keeps its words. The reader still sees what was claimed, and cannot
    be sent somewhere the system never looked.

    Returns the cleaned text and the list of URLs that were demoted, so the
    caller can record it rather than fixing the symptom silently.
    """
    if not text:
        return text, []
    demoted: list[str] = []

    def check(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        if not allowed or _canonical(url) in allowed:
            return match.group(0)
        demoted.append(url)
        return label

    cleaned = _MD_LINK.sub(check, text)
    return cleaned, demoted


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    real = ("**Yes — but you need written approval from the HOA's Architectural "
            "Review Committee.** Submit a design plan showing the proposed stone "
            "layout, drainage, and any retaining walls. Wait for approval before "
            "beginning any work, and keep a copy for your records. " * 2)
    # A realistic cost breakdown: letter-poor, but many distinct characters. This
    # is the case the second condition exists to protect — cutting it off in front
    # of the user would be far worse than letting a rare collapse run on.
    cost_table = ("| Measure | Cost | Saves |\n|---|---|---|\n" +
                  "".join(f"| {i} | ${i * 37}.50 | ${i * 12}.25/yr |\n"
                          for i in range(1, 40)))
    cases = [
        ("real prose", real, False),
        ("the observed collapse", "!" * 4000, True),
        ("long dashes", "-" * 500, True),
        ("short and terse", "No. See CC&R 4.2.", False),
        ("markdown-heavy", "## Steps\n\n1. " + "Insulate the pipe fully. " * 40, False),
        ("numeric cost table", cost_table, False),
        ("degenerate with 2 chars", "!?" * 2000, True),
        # All-whitespace strips to empty, so it is not DEGENERATE — it is EMPTY,
        # and `unusable_reason` owns that case. Asserted separately below.
        ("whitespace padding", " \n" * 900, False),
    ]
    ok = True
    for label, text, want in cases:
        got = degenerate_reason(text) is not None
        ok &= got == want
        print(f"  {'ok  ' if got == want else 'FAIL'} {label:<24} degenerate={got}")

    print()
    empties = [
        ("whitespace, tools ran", " \n" * 900, [{"kind": "call"}], True),
        ("empty, tools ran", "", [{"kind": "call"}], True),
        ("empty, NO tools -> real defect", "", [], False),
    ]
    for label, text, trace, want in empties:
        got = unusable_reason(text, trace) is not None
        ok &= got == want
        print(f"  {'ok  ' if got == want else 'FAIL'} {label:<32} unusable={got}")

    print()
    stream = [("short collapse not cut yet", "!" * 100, False),
              ("collapse past the fuse", "!" * 500, True)]
    for label, text, want in stream:
        got = stream_is_degenerate(text) is not None
        ok &= got == want
        print(f"  {'ok  ' if got == want else 'FAIL'} {label:<32} cut={got}")

    print("\nPASS" if ok else "\nFAIL — a threshold needs revisiting")
    raise SystemExit(0 if ok else 1)
