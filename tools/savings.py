"""Estimate annual utility savings from concrete efficiency measures.

Pure-Python and deterministic (no network, no LLM), like freeze_risk/heat_risk:
the dollar figures come from transparent arithmetic the report can defend, while
the LLM only explains and prioritizes them.

Savings percentages follow widely published US Department of Energy / ENERGY STAR
guidance. Every measure returns its `basis` so the agent can state the assumption,
and the whole result carries an `assumptions` block — these are ESTIMATES, and the
accuracy depends on the home's actual usage.
"""
from __future__ import annotations

# Typical US home end-use split (approx., EIA Residential Energy Consumption Survey).
_END_USE_SHARE = {
    "heating_cooling": 0.48,
    "water_heating": 0.15,
    "lighting": 0.05,
    "other": 0.32,
}

# Baseline: average US single-family home ~10,500 kWh/yr at ~2,000 sq ft.
_BASELINE_KWH_PER_YEAR = 10500
_BASELINE_SQFT = 2000

# Rough climate multipliers for heating/cooling load by IECC zone digit.
_CLIMATE_MULTIPLIER = {"1": 1.15, "2": 1.10, "3": 1.00, "4": 0.98, "5": 1.05, "6": 1.15, "7": 1.25}


def estimate_annual_kwh(square_feet: float | None = None, climate_zone: str | None = None) -> float:
    """Estimate a home's annual electricity use (kWh) from size and climate zone."""
    sqft = square_feet or _BASELINE_SQFT
    kwh = _BASELINE_KWH_PER_YEAR * (sqft / _BASELINE_SQFT)
    if climate_zone:
        digit = next((c for c in str(climate_zone) if c.isdigit()), None)
        if digit:
            kwh *= _CLIMATE_MULTIPLIER.get(digit, 1.0)
    return round(kwh)


def estimate_savings(
    electricity_cents_kwh: float,
    square_feet: float | None = None,
    climate_zone: str | None = None,
    water_heater_temp_f: float | None = None,
    filter_overdue: bool = False,
) -> dict:
    """Return itemized annual savings estimates for common efficiency measures.

    Args:
        electricity_cents_kwh: the applicable electricity rate.
        square_feet, climate_zone: from the home profile, to size the baseline.
        water_heater_temp_f: current setting; savings only if above 120F.
        filter_overdue: True if the HVAC filter is past its change interval.

    Returns:
        {"ok": True, annual_kwh, annual_cost_usd, measures:[...],
         total_annual_savings_usd, assumptions}
    """
    annual_kwh = estimate_annual_kwh(square_feet, climate_zone)
    rate = electricity_cents_kwh / 100.0
    annual_cost = annual_kwh * rate

    hvac_cost = annual_cost * _END_USE_SHARE["heating_cooling"]
    water_cost = annual_cost * _END_USE_SHARE["water_heating"]
    light_cost = annual_cost * _END_USE_SHARE["lighting"]

    measures: list[dict] = []

    def add(name: str, saving: float, basis: str, effort: str) -> None:
        if saving >= 1:
            measures.append({
                "measure": name,
                "annual_savings_usd": round(saving, 2),
                "basis": basis,
                "effort": effort,
            })

    add("Set the thermostat back ~7-10F for 8 hours a day (or use a programmable/smart schedule)",
        hvac_cost * 0.10,
        "DOE: ~10% annual heating/cooling savings from an 8-hour daily setback", "free")

    if filter_overdue:
        add("Replace the overdue HVAC air filter (and keep to the schedule)",
            hvac_cost * 0.05,
            "A clogged filter restricts airflow and raises HVAC energy use ~5%", "low ($15-25)")
    else:
        add("Keep replacing the HVAC filter on schedule",
            hvac_cost * 0.02,
            "Maintaining clean airflow avoids a few percent of HVAC waste", "low")

    if water_heater_temp_f and water_heater_temp_f > 120:
        add(f"Lower the water heater from {water_heater_temp_f:.0f}F to 120F",
            water_cost * 0.08,
            "DOE: ~4-8% water-heating savings per 10F reduction", "free")

    add("Seal air leaks and weather-strip doors/windows",
        hvac_cost * 0.10,
        "DOE: air sealing typically cuts heating/cooling ~10%", "low ($30-80 DIY)")

    add("Replace remaining incandescent/CFL bulbs with LEDs",
        light_cost * 0.70,
        "LEDs use ~70-80% less energy than incandescent lighting", "low, one-time")

    measures.sort(key=lambda m: m["annual_savings_usd"], reverse=True)
    total = round(sum(m["annual_savings_usd"] for m in measures), 2)

    return {
        "ok": True,
        "annual_kwh": annual_kwh,
        "annual_cost_usd": round(annual_cost, 2),
        "electricity_cents_kwh": electricity_cents_kwh,
        "measures": measures,
        "total_annual_savings_usd": total,
        "assumptions": (
            f"Estimated {annual_kwh:,} kWh/yr for a {square_feet or _BASELINE_SQFT:,.0f} sq ft home "
            f"in climate zone {climate_zone or 'n/a'}, at {electricity_cents_kwh}c/kWh. "
            "End-use split from EIA RECS averages. These are ESTIMATES — actual savings "
            "depend on your real usage, equipment, and habits."
        ),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(
        estimate_savings(16.44, square_feet=2200, climate_zone="IECC 3A",
                         water_heater_temp_f=120, filter_overdue=False),
        indent=2))
