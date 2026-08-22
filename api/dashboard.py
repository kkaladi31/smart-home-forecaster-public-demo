"""Assemble the live weather dashboard payload for the web UI.

Reuses the same tools the agent calls, so the dashboard and the agent can never
disagree about the weather: both read NWS/Open-Meteo through `tools/weather.py`
and both get their risk levels from the deterministic assessors.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from tools.alerts import get_weather_alerts
from tools.elevation import get_elevation
from tools.freeze_risk import assess_freeze_risk
from tools.heat_risk import assess_heat_risk, heat_index_f
from tools.weather import get_weather_forecast


def build_dashboard(
    latitude: float,
    longitude: float,
    label: str | None = None,
    horizon_hours: int = 48,
) -> dict:
    """Return everything the dashboard needs for one location."""
    # Elevation, forecast and alerts are three independent HTTP calls against
    # three different hosts, so they run concurrently rather than one after the
    # other — this is the whole latency of a dashboard load.
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_elev = pool.submit(get_elevation, latitude, longitude)
        f_wx = pool.submit(get_weather_forecast, latitude, longitude,
                           horizon_hours=horizon_hours)
        f_alerts = pool.submit(get_weather_alerts, latitude, longitude)
        elev, wx, alerts = f_elev.result(), f_wx.result(), f_alerts.result()

    elevation_ft = elev.get("elevation_ft") if elev.get("ok") else None

    if not wx.get("ok"):
        return {"ok": False, "error": wx.get("error", "forecast unavailable")}

    periods = wx["periods"]

    wind_at_min = next(
        (p.get("wind_mph") for p in periods if p.get("start") == wx["min_temp_time"]), None
    )
    freeze = assess_freeze_risk(wx["min_temp_f"], min_wind_mph=wind_at_min,
                                elevation_ft=elevation_ft)
    heat = assess_heat_risk(wx["max_temp_f"], humidity_pct=wx.get("humidity_at_max"))

    # Hourly series for the chart, with a computed feels-like so the chart can show
    # the gap between air temperature and what it actually feels like.
    hourly = []
    for p in periods:
        temp = p.get("temp_f")
        if temp is None:
            continue
        humidity = p.get("humidity")
        feels = p.get("feels_like_f")
        if feels is None:
            feels = heat_index_f(temp, humidity) if humidity is not None else temp
        hourly.append({
            "time": p["start"],
            "temp_f": round(float(temp), 1),
            "feels_like_f": round(float(feels), 1),
            "humidity": humidity,
            "wind_mph": p.get("wind_mph"),
            "condition": p.get("short", ""),
        })

    now = hourly[0] if hourly else {}
    return {
        "ok": True,
        "location": {
            "label": label or f"{latitude:.4f}, {longitude:.4f}",
            "latitude": latitude,
            "longitude": longitude,
            "elevation_ft": elevation_ft,
        },
        "source": wx["source"],
        "horizon_hours": horizon_hours,
        "current": {
            "temp_f": now.get("temp_f"),
            "feels_like_f": now.get("feels_like_f"),
            "humidity": now.get("humidity"),
            "wind_mph": now.get("wind_mph"),
            "condition": now.get("condition", ""),
            "time": now.get("time"),
        },
        "range": {
            "min_temp_f": wx["min_temp_f"], "min_temp_time": wx["min_temp_time"],
            "max_temp_f": wx["max_temp_f"], "max_temp_time": wx["max_temp_time"],
        },
        "hourly": hourly,
        "alerts": alerts.get("alerts", []) if alerts.get("ok") else [],
        "alerts_available": bool(alerts.get("ok")),
        "freeze": freeze,
        "heat": heat,
    }
