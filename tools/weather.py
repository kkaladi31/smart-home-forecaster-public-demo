"""Fetch an hourly temperature forecast for a location.

Primary source : US National Weather Service  api.weather.gov (free, no key, US, govt)
Fallback source: Open-Meteo forecast API           (free, no key, global)

The NWS -> Open-Meteo fallback is the concrete demonstration of the ReAct
requirement "recover from missteps": if the authoritative source is unavailable,
the agent still completes the task using the backup, and reports which it used.

Set `source="open-meteo"` to *force* the fallback path (used by the evaluation
suite to prove recovery works).
"""
from __future__ import annotations

import re

import requests

from config import HTTP_TIMEOUT, USER_AGENT
from tools.cache import TTL_FORECAST, cached
from tools.http import SESSION

NWS_POINTS_URL ="https://api.weather.gov/points/{lat},{lon}"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _parse_wind_mph(text: str | None) -> float | None:
    """NWS reports wind like '10 mph' or '5 to 10 mph'; take the largest number."""
    if not text:
        return None
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    return max(numbers) if numbers else None


def _via_nws(lat: float, lon: float, horizon_hours: int) -> dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    # Step 1: resolve the grid point to get the hourly-forecast URL.
    point = SESSION.get(
        NWS_POINTS_URL.format(lat=lat, lon=lon), headers=headers, timeout=HTTP_TIMEOUT
    )
    point.raise_for_status()
    hourly_url = point.json()["properties"]["forecastHourly"]

    # Step 2: fetch the hourly forecast periods. Same host as step 1, so the
    # pooled session reuses that connection instead of re-handshaking.
    fc = SESSION.get(hourly_url, headers=headers, timeout=HTTP_TIMEOUT)
    fc.raise_for_status()
    raw_periods = fc.json()["properties"]["periods"][:horizon_hours]

    periods = []
    for p in raw_periods:
        temp_f = float(p["temperature"]) if p.get("temperatureUnit") == "F" else None
        periods.append(
            {
                "start": p["startTime"],
                "temp_f": temp_f,
                "humidity": (p.get("relativeHumidity") or {}).get("value"),
                "wind_mph": _parse_wind_mph(p.get("windSpeed")),
                "short": p.get("shortForecast", ""),
            }
        )
    return {"source": "NWS (api.weather.gov)", "periods": periods}


def _via_open_meteo(lat: float, lon: float, horizon_hours: int) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "forecast_days": 3,
        "timezone": "auto",
    }
    resp = SESSION.get(
        OPEN_METEO_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    periods = []
    for i in range(min(horizon_hours, len(h["time"]))):
        periods.append(
            {
                "start": h["time"][i],
                "temp_f": float(h["temperature_2m"][i]),
                "humidity": h["relative_humidity_2m"][i],
                "feels_like_f": float(h["apparent_temperature"][i]),
                "wind_mph": float(h["wind_speed_10m"][i]),
                "short": "",
            }
        )
    return {"source": "Open-Meteo", "periods": periods}


@cached(TTL_FORECAST)
def get_weather_forecast(
    latitude: float,
    longitude: float,
    horizon_hours: int = 48,
    source: str = "auto",
) -> dict:
    """Return an hourly forecast plus the minimum temperature over the horizon.

    Args:
        latitude, longitude: location coordinates.
        horizon_hours: how many hours ahead to consider (default 48).
        source: "auto" (NWS then Open-Meteo), "nws", or "open-meteo" (force fallback).

    Returns:
        On success: {"ok": True, source, min_temp_f/time + humidity_at_min,
        max_temp_f/time + humidity_at_max, periods:[...]}.
        On failure: {"ok": False, "error": ...}.
    """
    attempts = []
    if source in ("auto", "nws"):
        attempts.append(_via_nws)
    if source in ("auto", "open-meteo"):
        attempts.append(_via_open_meteo)

    last_error = "no source attempted"
    for fetch in attempts:
        try:
            data = fetch(latitude, longitude, horizon_hours)
            periods = data["periods"]
            temps = [(p["temp_f"], p["start"]) for p in periods if p["temp_f"] is not None]
            if not temps:
                last_error = f"{data['source']} returned no usable temperatures"
                continue
            min_temp_f, min_temp_time = min(temps, key=lambda t: t[0])
            max_temp_f, max_temp_time = max(temps, key=lambda t: t[0])

            def _humidity_at(when: str):
                for p in periods:
                    if p["start"] == when:
                        return p.get("humidity")
                return None

            return {
                "ok": True,
                "source": data["source"],
                "latitude": latitude,
                "longitude": longitude,
                "horizon_hours": horizon_hours,
                "min_temp_f": min_temp_f,
                "min_temp_time": min_temp_time,
                "humidity_at_min": _humidity_at(min_temp_time),
                "max_temp_f": max_temp_f,
                "max_temp_time": max_temp_time,
                "humidity_at_max": _humidity_at(max_temp_time),
                "periods": periods,
            }
        except requests.RequestException as exc:
            last_error = f"{fetch.__name__} failed: {exc}"
            continue
    return {"ok": False, "error": f"Could not fetch forecast ({last_error})."}


if __name__ == "__main__":
    import json

    result = get_weather_forecast(32.7767, -96.7970, horizon_hours=24)
    # Trim periods for readable manual output.
    if result.get("ok"):
        result["periods"] = result["periods"][:3] + ["...trimmed..."]
    print(json.dumps(result, indent=2))
