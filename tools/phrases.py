"""Token-boundary phrase matching, written once because getting it wrong is a bug
this codebase has already shipped.

`tools/contractors.py` mapped loose language to trades with a dict of substrings
and `if key in query`. The entry `"ac": "hvac"` therefore matched "repl**ac**e my
lawn" and "surf**ac**e", and because the Advisor passes the user's *entire
question* as the trade argument, a lawn question could return an HVAC company.

The fix is not "be careful when adding terms" — it is to make the careless
version impossible. Phrases are authored as plain words and compiled here into
`\\b`-anchored patterns, so a term cannot match inside a longer word no matter
who adds it or when. Both callers that map user language onto a vocabulary — the
Router's intent table and the Pro Finder's trade table — go through this, and
each asserts the property over its whole table in the eval suite rather than on a
chosen example.

Whitespace inside a phrase is matched flexibly, so "how   do  I" hits "how do i".
"""
from __future__ import annotations

import re

# A compiled phrase keeps the original wording beside its pattern. That is what
# lets a match be reported as the words a human wrote — `matched: ["freezing"]` —
# instead of as a regex, which is the difference between a diagnosable decision
# and an opaque one.
Compiled = list[tuple[str, re.Pattern]]


def compile_phrases(terms: tuple[str, ...] | list[str]) -> Compiled:
    """Compile plain phrases into token-boundary patterns, keeping each phrase."""
    compiled: Compiled = []
    for term in terms:
        body = r"\s+".join(re.escape(word) for word in term.split())
        compiled.append((term, re.compile(rf"\b{body}\b", re.I)))
    return compiled


def find_terms(text: str, compiled: Compiled) -> list[str]:
    """Every phrase present in `text`, in the order the phrases were declared.

    Declaration order rather than order of appearance: callers use these lists to
    break ties, and a tie-break that depends on where a word landed in a sentence
    is not reproducible across two ways of asking the same question.
    """
    return [term for term, pattern in compiled if pattern.search(text)]


def matches(text: str, compiled: Compiled) -> bool:
    return any(pattern.search(text) for _, pattern in compiled)


def leaks_across_boundaries(terms: tuple[str, ...] | list[str]) -> list[str]:
    """Phrases that still match when glued inside a longer word — always empty.

    Exposed as a helper rather than written out in each test, so a new phrase
    table gets the guarantee by calling one function. Gluing word characters to
    both ends must kill every match; anything that survives can fire from inside
    an unrelated word, which is the original bug.
    """
    compiled = compile_phrases(terms)
    return [term for term, pattern in compiled if pattern.search(f"zq{term}zq")]


if __name__ == "__main__":
    # The regression that motivated the module, stated as an example.
    trade_terms = ("ac", "air conditioning", "lawn", "roof")
    compiled = compile_phrases(trade_terms)
    for sentence in ("replace my lawn", "the surface is cracked",
                     "my ac is broken", "check the roof"):
        print(f"  {sentence:<28} -> {find_terms(sentence, compiled) or '(nothing)'}")
    print(f"\n  boundary leaks: {leaks_across_boundaries(trade_terms) or 'none'}")
