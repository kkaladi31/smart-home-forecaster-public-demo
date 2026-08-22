"""Assess pipe/property freeze risk from a forecast, and list protective actions.

This is a *pure-Python* tool (no network, no LLM). Doing the numeric thresholding
in code — rather than asking the language model to "eyeball" temperatures — is a
deliberate design choice: it keeps the risk rating deterministic and auditable,
while the LLM focuses on explaining and prioritizing the guidance.

Thresholds are based on widely published cold-weather plumbing guidance:
- Pipes are broadly at risk when sustained outdoor temps fall to ~20F or below.
- 28-32F brings frost/freeze concerns for exposed spigots and shallow lines.
Values are conservative defaults; tune them for your climate.
"""
from __future__ import annotations

# (upper_bound_f, level, headline). Evaluated low-to-high; first match wins.
_RISK_BANDS = [
    (20.0, "severe", "Hard freeze: high risk of frozen/burst pipes."),
    (28.0, "high", "Freeze: exposed pipes and spigots are at real risk."),
    (32.0, "moderate", "Freezing point reached: frost and localized freezing possible."),
    (36.0, "low", "Near-freezing: worth watching, minimal action needed."),
]

_ACTIONS_BY_LEVEL = {
    "severe": [
        "Shut off the valve feeding outdoor spigots and drain the spigots.",
        "Insulate exposed pipes (foam sleeves) in unheated spaces (garage, crawlspace, attic).",
        "Let a pencil-thin stream of water drip from faucets on exterior walls overnight.",
        "Open cabinet doors under sinks on exterior walls so warm air reaches the pipes.",
        "Cover outdoor spigots with insulated faucet covers.",
        "Keep the home heated to at least ~55F, even if away.",
    ],
    "high": [
        "Shut off and drain outdoor spigots; cover them with insulated faucet covers.",
        "Insulate exposed/at-risk pipes in unheated areas.",
        "Consider a slow faucet drip overnight on the coldest exterior-wall fixtures.",
        "Open under-sink cabinet doors on exterior walls.",
    ],
    "moderate": [
        "Cover outdoor spigots and disconnect/drain garden hoses.",
        "Check that exposed pipes in the garage or crawlspace are insulated.",
    ],
    "low": [
        "Disconnect and drain garden hoses.",
        "Monitor the forecast in case temperatures drop further.",
    ],
    "none": [
        "No freeze protection needed based on the current forecast.",
    ],
}


def assess_freeze_risk(
    min_temp_f: float,
    min_wind_mph: float | None = None,
    elevation_ft: float | None = None,
) -> dict:
    """Rate freeze risk and return recommended protective actions.

    Args:
        min_temp_f: the lowest forecast air temperature over the horizon.
        min_wind_mph: optional wind speed at the cold point (raises effective risk).
        elevation_ft: optional elevation (context only; higher tends colder).

    Returns:
        {"ok": True, level, headline, min_temp_f, wind_chill_f, actions:[...], notes}.
        `level` is one of: none | low | moderate | high | severe.
    """
    level, headline = "none", "No freeze expected in the forecast horizon."
    for upper, lvl, head in _RISK_BANDS:
        if min_temp_f <= upper:
            level, headline = lvl, head
            break

    # Wind chill can push effective conditions colder; if it crosses the next band
    # down, note it (we escalate the *guidance*, not the raw level, to stay honest).
    wind_chill_f = _wind_chill(min_temp_f, min_wind_mph) if min_wind_mph else None
    notes = []
    if wind_chill_f is not None and wind_chill_f < min_temp_f - 3:
        notes.append(
            f"Wind chill near {wind_chill_f:.0f}F (wind {min_wind_mph:.0f} mph) makes "
            "exposed pipes lose heat faster than the air temperature alone suggests."
        )
    if elevation_ft is not None and elevation_ft > 3000:
        notes.append(
            f"Elevation ~{elevation_ft:.0f} ft can run colder than nearby lowland stations."
        )

    return {
        "ok": True,
        "level": level,
        "headline": headline,
        "min_temp_f": min_temp_f,
        "wind_chill_f": round(wind_chill_f, 1) if wind_chill_f is not None else None,
        "elevation_ft": elevation_ft,
        "actions": _ACTIONS_BY_LEVEL[level],
        "notes": " ".join(notes),
    }


def _wind_chill(temp_f: float, wind_mph: float) -> float:
    """US NWS wind-chill formula (valid for temp <= 50F and wind > 3 mph)."""
    if temp_f > 50 or wind_mph <= 3:
        return temp_f
    v = wind_mph ** 0.16
    return 35.74 + 0.6215 * temp_f - 35.75 * v + 0.4275 * temp_f * v


if __name__ == "__main__":
    import json

    print(json.dumps(assess_freeze_risk(18, min_wind_mph=15, elevation_ft=430), indent=2))
