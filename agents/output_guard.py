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
