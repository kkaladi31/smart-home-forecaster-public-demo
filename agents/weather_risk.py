"""Deterministic multi-hazard weather-risk pipeline (no LLM).

Wires the raw tools together the way the agent is expected to: geocode ->
elevation -> forecast -> active alerts, then assesses freeze risk AND heat risk,
and surfaces any official NWS advisory (heat, fire/Red Flag, storm, flood, wind,
winter, ...). Runs with NO API key, so it powers the `--demo` smoke test and is a
readable reference for good tool orchestration.
"""
from __future__ import annotations

from tools.alerts import get_weather_alerts
from tools.elevation import get_elevation
from tools.freeze_risk import assess_freeze_risk
from tools.geocode import geocode_address
from tools.heat_risk import assess_heat_risk
from tools.weather import get_weather_forecast

SAFETY_DISCLAIMER = (
    "This is general guidance, not professional advice. For gas, electrical, flooding, "
    "burst-pipe, or medical emergencies, contact 911, a licensed professional, or your utility."
)


def run_weather_check(address: str, horizon_hours: int = 48, weather_source: str = "auto") -> dict:
    """Run the full multi-hazard weather-risk check for an address."""
    trace: list[str] = []

    geo = geocode_address(address)
    trace.append(f"geocode_address({address!r}) -> {'ok' if geo.get('ok') else 'FAILED'}")
    if not geo.get("ok"):
        return {"ok": False, "step": "geocode", "error": geo.get("error"), "trace": trace}
    lat, lon = geo["latitude"], geo["longitude"]

    elev = get_elevation(lat, lon)
    elevation_ft = elev.get("elevation_ft") if elev.get("ok") else None
    trace.append(f"get_elevation -> {'ok' if elev.get('ok') else 'skipped'}")

    wx = get_weather_forecast(lat, lon, horizon_hours=horizon_hours, source=weather_source)
    trace.append(f"get_weather_forecast(source={weather_source}) -> "
                 f"{wx.get('source') if wx.get('ok') else 'FAILED'}")
    if not wx.get("ok"):
        return {"ok": False, "step": "weather", "error": wx.get("error"), "trace": trace}

    alerts = get_weather_alerts(lat, lon)
    trace.append(f"get_weather_alerts -> {alerts.get('count') if alerts.get('ok') else 'unavailable'}")

    min_wind = next((p.get("wind_mph") for p in wx["periods"] if p.get("start") == wx["min_temp_time"]), None)
    freeze = assess_freeze_risk(wx["min_temp_f"], min_wind_mph=min_wind, elevation_ft=elevation_ft)
    heat = assess_heat_risk(wx["max_temp_f"], humidity_pct=wx.get("humidity_at_max"))
    trace.append(f"assess_freeze_risk -> {freeze['level']}; assess_heat_risk -> {heat['level']}")

    return {
        "ok": True,
        "address": geo["matched_address"],
        "coordinates": {"latitude": lat, "longitude": lon},
        "elevation_ft": elevation_ft,
        "forecast": {
            "source": wx["source"],
            "min_temp_f": wx["min_temp_f"], "min_temp_time": wx["min_temp_time"],
            "max_temp_f": wx["max_temp_f"], "max_temp_time": wx["max_temp_time"],
            "horizon_hours": horizon_hours,
        },
        "freeze": freeze,
        "heat": heat,
        "alerts": alerts,
        "sources": [geo["source"], "Open-Meteo/USGS (elevation)", wx["source"],
                    "NWS active alerts" if alerts.get("ok") else "alerts unavailable"],
        "disclaimer": SAFETY_DISCLAIMER,
        "trace": trace,
    }


def format_weather_report(report: dict) -> str:
    """Render a run_weather_check() result as readable console text."""
    if not report.get("ok"):
        return f"Could not complete weather check at step '{report.get('step')}': {report.get('error')}"

    fc = report["forecast"]
    lines = [
        f"Weather-risk check for: {report['address']}",
        f"Forecast ({fc['source']}, next {fc['horizon_hours']}h): "
        f"low {fc['min_temp_f']}F, high {fc['max_temp_f']}F",
    ]

    alerts = report["alerts"]
    if alerts.get("ok") and alerts.get("count"):
        lines.append("")
        lines.append(f"ACTIVE ADVISORIES ({alerts['count']}):")
        for a in alerts["alerts"]:
            lines.append(f"  ! {a['event']} ({a['severity']}) — {a['area'][:60]}")

    for hazard, label in (("freeze", "FREEZE"), ("heat", "HEAT")):
        h = report[hazard]
        if h["level"] != "none":
            lines += ["", f"{label} RISK: {h['level'].upper()} — {h['headline']}"]
            lines += [f"  - {a}" for a in h["actions"]]
            if h.get("notes"):
                lines.append(f"  Note: {h['notes']}")

    if (not (alerts.get("ok") and alerts.get("count"))
            and report["freeze"]["level"] == "none" and report["heat"]["level"] == "none"):
        lines += ["", "No significant weather risks (freeze/heat/advisories) in the forecast horizon."]

    lines += ["", f"Sources: {', '.join(report['sources'])}", "", report["disclaimer"]]
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_weather_report(run_weather_check("Phoenix, AZ")))
