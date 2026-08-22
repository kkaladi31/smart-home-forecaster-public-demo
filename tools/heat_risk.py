"""Assess heat risk from temperature + humidity, with protective guidance.

The twin of freeze_risk.py: a pure-Python, deterministic assessor (no network,
no LLM) so the heat rating is auditable. It computes the NWS "heat index" (how
hot it feels accounting for humidity) and maps it to the NWS risk categories,
each with concrete protective actions. This lets the agent warn about dangerous
heat proactively, complementing official NWS Excessive-Heat alerts.
"""
from __future__ import annotations

# (upper_bound_heat_index_f, level, headline). Evaluated low-to-high; first match wins.
# Bounds follow the NWS heat-index caution categories.
_RISK_BANDS = [
    (90.0, "low", "Caution: fatigue possible with prolonged exposure/activity."),
    (103.0, "moderate", "Extreme caution: heat cramps/exhaustion possible."),
    (125.0, "high", "Danger: heat exhaustion likely; heatstroke possible."),
    (999.0, "severe", "Extreme danger: heatstroke highly likely."),
]

_ACTIONS_BY_LEVEL = {
    "low": [
        "Stay hydrated and take breaks in shade during outdoor activity.",
        "Avoid the hottest part of the day (roughly 11am–5pm) for strenuous work.",
    ],
    "moderate": [
        "Drink water regularly; don't wait until you're thirsty.",
        "Limit outdoor exertion to early morning or evening.",
        "Check on elderly neighbors, young children, and anyone without air conditioning.",
        "Never leave people or pets in a parked car.",
    ],
    "high": [
        "Stay indoors with air conditioning; use a cooling center if you have none.",
        "Postpone strenuous outdoor activity.",
        "Hydrate continuously and watch for heat-illness signs (dizziness, nausea, cramping).",
        "Actively check on vulnerable people and pets; never leave anyone in a car.",
        "Provide shade and water for pets and outdoor animals.",
    ],
    "severe": [
        "Stay in air conditioning; treat this as dangerous — go to a cooling center if needed.",
        "Cancel outdoor activity and strenuous work.",
        "Know heatstroke signs (confusion, hot/dry skin, fainting) — call 911 if suspected.",
        "Frequently check on the elderly, children, and anyone medically vulnerable.",
        "Keep pets indoors and hydrated.",
    ],
    "none": ["No heat protection needed based on the current forecast."],
}


def heat_index_f(temp_f: float, humidity_pct: float | None) -> float:
    """NWS heat index (feels-like temperature). Falls back to air temp if humidity
    is unknown or the temperature is too low for the formula to apply (<80°F)."""
    if humidity_pct is None or temp_f < 80:
        return temp_f
    t, rh = temp_f, humidity_pct
    hi = (
        -42.379 + 2.04901523 * t + 10.14333127 * rh
        - 0.22475541 * t * rh - 0.00683783 * t * t - 0.05481717 * rh * rh
        + 0.00122874 * t * t * rh + 0.00085282 * t * rh * rh - 0.00000199 * t * t * rh * rh
    )
    # Low-humidity and high-humidity adjustments per the NWS formula.
    if rh < 13 and 80 <= t <= 112:
        hi -= ((13 - rh) / 4) * ((17 - abs(t - 95)) / 17) ** 0.5
    elif rh > 85 and 80 <= t <= 87:
        hi += ((rh - 85) / 10) * ((87 - t) / 5)
    return hi


def assess_heat_risk(max_temp_f: float, humidity_pct: float | None = None) -> dict:
    """Rate heat risk from the forecast high temperature and humidity.

    Args:
        max_temp_f: the highest forecast air temperature over the horizon.
        humidity_pct: relative humidity (%) at that time, if available.

    Returns:
        {"ok": True, level, headline, heat_index_f, actions:[...], notes}.
        `level` is one of: none | low | moderate | high | severe.
    """
    hi = heat_index_f(max_temp_f, humidity_pct)
    level, headline = "none", "No dangerous heat expected in the forecast horizon."
    if hi >= 80:
        for upper, lvl, head in _RISK_BANDS:
            if hi <= upper:
                level, headline = lvl, head
                break

    notes = ""
    if humidity_pct is None:
        notes = "Humidity was unavailable, so this uses air temperature only; humid conditions feel hotter."
    return {
        "ok": True,
        "level": level,
        "headline": headline,
        "air_temp_f": max_temp_f,
        "humidity_pct": humidity_pct,
        "heat_index_f": round(hi, 1),
        "actions": _ACTIONS_BY_LEVEL[level],
        "notes": notes,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(assess_heat_risk(104, humidity_pct=35), indent=2))
