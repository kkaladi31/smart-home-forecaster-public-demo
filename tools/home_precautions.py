"""What an official advisory means for the HOUSE, in deterministic code.

The National Weather Service tells people how to stay safe: drink water, seek
shelter, avoid travel. It does not tell them to disconnect the garden hose before
a freeze, or to latch the garage door before a wind event, because protecting the
building is not its job.

That gap is this product's entire subject. A forecaster that relays "Heat
Advisory — stay hydrated" and stops has repeated the radio; the reason to have a
*home* forecaster is the sentence after that one.

WHY THIS IS A TABLE AND NOT A PROMPT. It sits behind an unprompted pop-up. The
same reasoning that keeps freeze and heat levels out of the model's hands applies
with more force here, because nobody asked for this text and nobody is waiting to
judge it: it must be instant, free, identical every time, and incapable of
inventing a precaution. Advisory categories are a small closed set that changes
about never, which is exactly the shape a table is for.

Matched on the NWS `event` string, which is a controlled vocabulary
("Excessive Heat Warning", "Hard Freeze Watch", "Red Flag Warning"). Keywords are
matched rather than exact strings so that Watch/Warning/Advisory variants of the
same hazard share one entry — the difference between them is urgency, and the
thing to do to your house is the same.
"""
from __future__ import annotations

import re

# Ordered: the FIRST category whose pattern matches wins, so specific hazards are
# listed before the broad ones they would otherwise be swallowed by. "Winter
# Storm" must beat "Storm"; "Freezing Fog" must not read as a freeze event that
# threatens plumbing.
_CATEGORIES: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "freeze",
        r"freeze|frost|wind\s*chill|extreme\s*cold|cold\s*weather",
        (
            "Disconnect garden hoses and cover outdoor spigots — a trapped hose is "
            "the most common cause of a burst outdoor tap.",
            "Let a pencil-thin trickle run from taps on exterior walls overnight; "
            "moving water is far harder to freeze than still water.",
            "Open cabinet doors under kitchen and bathroom sinks so household heat "
            "reaches the pipes behind them.",
            "Keep the thermostat at 55 °F or above even if the house is empty.",
            "Find your main water shut-off now and make sure it turns — you do not "
            "want to be looking for it while water is running.",
        ),
    ),
    (
        "snow_ice",
        r"winter\s*storm|ice\s*storm|blizzard|heavy\s*snow|snow\s*squall|sleet",
        (
            "Clear roof drains and gutters if you can do it safely from the ground; "
            "blocked drainage is what turns melting snow into an ice dam.",
            "Move the car out from under heavy limbs and away from the roof line.",
            "Check that heating vents and the furnace intake outside are not buried.",
            "Have the water shut-off and a torch to hand in case power goes.",
            "Do not go up on the roof to clear snow — that is work at height, and "
            "it belongs to someone equipped for it.",
        ),
    ),
    (
        "heat",
        r"heat|hot\s*weather",
        (
            "Close blinds and curtains on the sun-facing side during the day — "
            "stopping the sun at the glass beats cooling the room afterwards.",
            "Check and replace the AC filter; a clogged filter makes the system work "
            "harder for less cooling, exactly when you need it most.",
            "Run ceiling fans counter-clockwise, and only in rooms being used — a fan "
            "cools people, not rooms.",
            "Avoid the oven and dryer during the hottest hours; both dump heat into "
            "the house that the AC then has to remove.",
            "Make sure the outdoor condenser unit is clear of leaves and debris.",
        ),
    ),
    (
        "wind",
        r"high\s*wind|wind\s*advisory|gale|hurricane|tropical\s*storm",
        (
            "Bring in or tie down patio furniture, umbrellas, bins and trampolines — "
            "loose objects become the thing that breaks a window.",
            "Close and latch the garage door; a garage door failing is a common way "
            "wind gets inside and lifts a roof.",
            "Park away from trees and power lines.",
            "Charge phones and torches in case the power goes.",
        ),
    ),
    (
        "flood",
        r"flood|storm\s*surge|heavy\s*rain",
        (
            "Move valuables, documents and electronics off basement and ground-level "
            "floors.",
            "Test the sump pump if you have one, and check its discharge is clear.",
            "Clear leaves from storm drains and downspouts near the house so water "
            "runs away from the foundation.",
            "Never drive or walk through moving water, and if water reaches outlets "
            "or the electrical panel, do not wade in — call for help.",
        ),
    ),
    (
        "fire",
        r"red\s*flag|fire\s*weather",
        (
            "Clear dry leaves, needles and brush within five feet of the walls, deck "
            "and under the porch.",
            "Move firewood, propane tanks and anything else that burns away from the "
            "structure.",
            "Close windows and any vents you can, and shut exterior doors — embers "
            "travel far ahead of a fire and get in through openings.",
            "Keep the car facing out with the keys to hand.",
        ),
    ),
    (
        "storm",
        r"thunderstorm|tornado|lightning|hail",
        (
            "Bring in or secure anything loose outside.",
            "Unplug sensitive electronics — a surge protector is not a substitute "
            "during a direct strike.",
            "Know which interior room on the lowest floor, away from windows, you "
            "would shelter in.",
            "Move the car under cover if hail is expected.",
        ),
    ),
    (
        "air",
        r"air\s*quality|smoke|dust",
        (
            "Close windows and outside doors.",
            "Set the HVAC to recirculate so it is not drawing outside air in.",
            "Run a clean, higher-rated filter if you have one; avoid vacuuming and "
            "burning candles, which add particles indoors.",
        ),
    ),
]

_COMPILED = [(name, re.compile(pattern, re.IGNORECASE), actions)
             for name, pattern, actions in _CATEGORIES]


def precautions_for(event: str) -> dict:
    """Home-protection steps for one advisory. Empty when nothing is known.

    Returns `{"category": str | None, "actions": [str, ...]}`. An unrecognised
    event yields no actions rather than generic filler: advice that fits any
    hazard tells the reader nothing and costs the specific advice its credibility.
    """
    text = (event or "").strip()
    if not text:
        return {"category": None, "actions": []}
    for name, pattern, actions in _COMPILED:
        if pattern.search(text):
            return {"category": name, "actions": list(actions)}
    return {"category": None, "actions": []}


if __name__ == "__main__":
    samples = [
        "Excessive Heat Warning", "Heat Advisory", "Hard Freeze Watch",
        "Wind Chill Advisory", "Winter Storm Warning", "High Wind Warning",
        "Flash Flood Warning", "Red Flag Warning", "Severe Thunderstorm Warning",
        "Air Quality Alert", "Special Weather Statement",
    ]
    for s in samples:
        got = precautions_for(s)
        print(f"{s:<32} {str(got['category']):<10} {len(got['actions'])} actions")
