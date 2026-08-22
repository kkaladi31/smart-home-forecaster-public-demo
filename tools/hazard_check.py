"""One-shot multi-hazard assessment for a location.

Why this exists
---------------
The agent used to reach the same answer by calling five tools in sequence:

    geocode_address -> get_elevation -> get_weather_forecast
                    -> get_weather_alerts -> assess_freeze_risk / assess_heat_risk

Each of those is a separate round-trip to the *model*, not just to the network,
because the model has to see one result before it can ask for the next. Measured
on this project, the model is ~88% of a turn's wall clock, so five extra model
turns cost far more than the five HTTP calls do. Collapsing the chain into one
tool call is the single largest latency win available.

Nothing about the *decisions* changes: this runs exactly the same deterministic
assessors (`tools/freeze_risk.py`, `tools/heat_risk.py`) on exactly the same
data. The model still never judges a temperature itself — it just gets the
verdicts in one response instead of five.

Two further savings fall out of doing it here:

* **The three fetches run concurrently.** Once coordinates are known, elevation,
  forecast and alerts are independent, so they overlap instead of queueing.
* **The 48 hourly periods never reach the model.** It only ever used the min/max
  and their humidity; sending the full array cost ~2k tokens per turn *and* those
  tokens were re-sent on every later turn in the conversation. The dashboard
  still reads the full series straight from `tools/weather.py`.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import telemetry
from tools.alerts import get_weather_alerts
from tools.elevation import get_elevation
from tools.freeze_risk import assess_freeze_risk
from tools.geocode import geocode_address
from tools.heat_risk import assess_heat_risk
from tools.weather import get_weather_forecast

# Only the alerts the user needs to act on. NWS can return a dozen overlapping
# products for one point; the list is already sorted most-severe-first.
_MAX_ALERTS = 4


def _wind_at(periods: list[dict], when: str | None) -> float | None:
    """Wind speed at a specific forecast hour (used for wind chill)."""
    if not when:
        return None
    for p in periods:
        if p.get("start") == when:
            return p.get("wind_mph")
    return None


def run_hazard_check(location: str, horizon_hours: int = 48) -> dict:
    """Resolve `location` and return freeze + heat + advisory findings together.

    Returns a compact dict; on failure `{"ok": False, "error": ...}` so the agent
    can recover the same way it does for any other tool.
    """
    with telemetry.span("tool", "hazard.check", f"Multi-hazard check for {location}") as s:
        geo = geocode_address(location)
        if not geo.get("ok"):
            s["resolved"] = False
            return {
                "ok": False,
                "error": geo.get("error", f"Could not locate {location!r}."),
                "hint": "Try a simpler form such as 'City, ST'.",
            }

        lat, lon = geo["latitude"], geo["longitude"]
        s["resolved"] = geo.get("matched_address")

        # Independent given coordinates — overlap them. Threads (not asyncio)
        # because every one of these is a blocking `requests` call.
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_elev = pool.submit(get_elevation, lat, lon)
            f_wx = pool.submit(get_weather_forecast, lat, lon, horizon_hours=horizon_hours)
            f_alerts = pool.submit(get_weather_alerts, lat, lon)
            elev, wx, alerts = f_elev.result(), f_wx.result(), f_alerts.result()

        if not wx.get("ok"):
            # Elevation and alerts may have succeeded, but with no forecast there
            # is no hazard verdict to give — say so rather than half-answering.
            return {
                "ok": False,
                "error": wx.get("error", "forecast unavailable"),
                "location": geo.get("matched_address"),
            }

        elevation_ft = elev.get("elevation_ft") if elev.get("ok") else None
        periods = wx.get("periods", [])

        freeze = assess_freeze_risk(
            wx["min_temp_f"],
            min_wind_mph=_wind_at(periods, wx.get("min_temp_time")),
            elevation_ft=elevation_ft,
        )
        heat = assess_heat_risk(wx["max_temp_f"], humidity_pct=wx.get("humidity_at_max"))

        active = alerts.get("alerts", []) if alerts.get("ok") else []
        s.update({"freeze": freeze.get("level"), "heat": heat.get("level"),
                  "alerts": len(active)})

        return {
            "ok": True,
            "location": {
                "resolved": geo.get("matched_address", location),
                "latitude": lat,
                "longitude": lon,
                "elevation_ft": elevation_ft,
                # Surfaced so the answer can say the location is place-level, per
                # the same rule geocode_address documents.
                "approximate": bool(geo.get("approximate")),
            },
            "forecast": {
                "source": wx["source"],
                "horizon_hours": horizon_hours,
                "min_temp_f": wx["min_temp_f"],
                "min_temp_time": wx["min_temp_time"],
                "max_temp_f": wx["max_temp_f"],
                "max_temp_time": wx["max_temp_time"],
                "humidity_at_max": wx.get("humidity_at_max"),
            },
            "freeze": freeze,
            "heat": heat,
            "alerts": {
                "available": bool(alerts.get("ok")),
                "count": len(active),
                "active": [
                    {
                        "event": a["event"],
                        "severity": a["severity"],
                        "headline": a["headline"],
                        "instruction": a["instruction"][:300],
                    }
                    for a in active[:_MAX_ALERTS]
                ],
            },
        }


if __name__ == "__main__":
    import json
    import time

    started = time.perf_counter()
    result = run_hazard_check("Dallas, TX")
    print(f"took {time.perf_counter() - started:.2f}s")
    print(json.dumps(result, indent=2)[:2000])
