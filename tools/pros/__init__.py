"""Pro Finder — licensed professionals, where the licence is a gate and not a score.

The rule this package exists to enforce: **a contractor whose registration is not
current is never recommended.** They are withheld, and the reason is stated.
Ranking an expired licensee lower still puts them in front of the user, and "we
showed them, just further down" is not a defence when the user hires one.

That is a change in kind from what it replaced. `tools/contractors.py` sorted a
directory by star rating and had no concept of a licence at all, so the claim
"we never recommend an unlicensed contractor" was simply not true of the code.
The demo fixture makes the difference concrete: the highest-rated plumber for the
Minneapolis home is the one the gate refuses.

    tools/pros/core.py      the gate, the ranking, and the shared shapes
    tools/pros/trades.py    a homeowner's words -> L&I's two ways of naming a trade
    tools/pros/lni_wa.py    live state registry client, full profile only —
                            not present in a demo build, and not imported by one
    tools/pros/fixtures.py  the synthetic directory, demo profile

Import from the submodules — `from tools.pros.core import find_pros` — rather than
re-exporting here. Pulling `core` into the package `__init__` makes
`python -m tools.pros.trades` warn about a module already in `sys.modules`, and
the self-tests are meant to be the easy way to check this code.
"""
