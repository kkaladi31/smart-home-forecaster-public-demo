"""Rich weather detail: multi-day forecast, air quality, and derived indices.

This backs the dashboard's 24h / 48h / 7-day views and the per-day detail panel.
It is deliberately provider-agnostic at the call site: `get_weather_detail()` is
the only thing the API layer knows about, so a Google Maps Platform provider can
be slotted in later without touching the endpoint or the UI.

Current provider: **Open-Meteo** (free, no API key, no billing account) plus its
Air Quality service. Everything here was verified live against those endpoints.

Known coverage gap: Open-Meteo's pollen data is CAMS *Europe* only — US
locations return nulls. `pollen` is therefore `None` outside Europe rather than
faked, and the UI says so. Google's Pollen API covers the US and is the intended
upgrade path.
"""
from __future__ import annotations

import math
from datetime import date, datetime

import requests

from config import HTTP_TIMEOUT, USER_AGENT
from tools.cache import TTL_AIR_QUALITY, TTL_FORECAST, cached
from tools.http import SESSION

FORECAST_URL ="https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# WMO weather codes -> (label, emoji). Used for the condition shown per hour/day.
_WMO = {
    0: ("Clear", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"), 45: ("Fog", "🌫️"), 48: ("Rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Heavy drizzle", "🌦️"),
    56: ("Freezing drizzle", "🌧️"), 57: ("Freezing drizzle", "🌧️"),
    61: ("Light rain", "🌦️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
    66: ("Freezing rain", "🌧️"), 67: ("Freezing rain", "🌧️"),
    71: ("Light snow", "🌨️"), 73: ("Snow", "🌨️"), 75: ("Heavy snow", "❄️"),
    77: ("Snow grains", "🌨️"), 80: ("Rain showers", "🌦️"), 81: ("Rain showers", "🌧️"),
    82: ("Violent showers", "⛈️"), 85: ("Snow showers", "🌨️"), 86: ("Snow showers", "❄️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm, hail", "⛈️"), 99: ("Thunderstorm, hail", "⛈️"),
}


def describe_code(code) -> dict:
    label, icon = _WMO.get(code, ("—", "🌡️"))
    return {"code": code, "label": label, "icon": icon}


# --- derived indices ---------------------------------------------------------

def moon_phase(when: date | None = None) -> dict:
    """Moon phase for a date, computed rather than fetched (no free API needed).

    Uses days since a known new moon (2000-01-06) over the 29.53-day synodic
    month. Accurate to well within a day, which is all a weather panel needs.
    """
    when = when or date.today()
    known_new_moon = date(2000, 1, 6)
    synodic = 29.530588853
    age = ((when - known_new_moon).days % synodic + synodic) % synodic
    illumination = round((1 - math.cos(2 * math.pi * age / synodic)) / 2 * 100)

    if age < 1.85: name, icon = "New moon", "🌑"
    elif age < 5.5: name, icon = "Waxing crescent", "🌒"
    elif age < 9.2: name, icon = "First quarter", "🌓"
    elif age < 12.9: name, icon = "Waxing gibbous", "🌔"
    elif age < 16.6: name, icon = "Full moon", "🌕"
    elif age < 20.3: name, icon = "Waning gibbous", "🌖"
    elif age < 24.0: name, icon = "Last quarter", "🌗"
    elif age < 27.7: name, icon = "Waning crescent", "🌘"
    else: name, icon = "New moon", "🌑"

    return {"name": name, "icon": icon, "illumination_pct": illumination,
            "age_days": round(age, 1)}


def running_conditions(
    temp_f: float | None,
    humidity: float | None = None,
    aqi: float | None = None,
    uv: float | None = None,
    wind_mph: float | None = None,
) -> dict:
    """Rate how good conditions are for a run, deterministically.

    Same philosophy as the freeze/heat assessors: the judgement is arithmetic in
    code, not the language model's opinion, so it is consistent and explainable.
    Starts at 100 and deducts for each adverse factor.
    """
    if temp_f is None:
        return {"score": None, "verdict": "unknown", "reasons": ["No temperature data."]}

    score = 100
    reasons: list[str] = []

    # Ideal running temperature is roughly 45-60F.
    if temp_f > 85: score -= 35; reasons.append("Hot — hydrate and slow your pace.")
    elif temp_f > 75: score -= 15; reasons.append("Warm.")
    elif temp_f < 20: score -= 30; reasons.append("Very cold — cover extremities.")
    elif temp_f < 35: score -= 12; reasons.append("Cold — layer up.")

    if humidity is not None and humidity > 80 and temp_f > 70:
        score -= 15
        reasons.append("Humid — sweat evaporates poorly.")

    if aqi is not None:
        if aqi > 150: score -= 40; reasons.append("Unhealthy air — consider indoors.")
        elif aqi > 100: score -= 20; reasons.append("Air is unhealthy for sensitive groups.")
        elif aqi > 50: score -= 8; reasons.append("Moderate air quality.")

    if uv is not None and uv >= 8:
        score -= 10
        reasons.append("Very high UV — sunscreen, or go early/late.")

    if wind_mph is not None and wind_mph > 20:
        score -= 10
        reasons.append("Windy.")

    score = max(0, min(100, score))
    verdict = ("Great" if score >= 80 else "Good" if score >= 65
               else "Fair" if score >= 45 else "Poor")
    if not reasons:
        reasons.append("Comfortable conditions.")
    return {"score": score, "verdict": verdict, "reasons": reasons}


# --- providers ---------------------------------------------------------------

@cached(TTL_FORECAST)
def _fetch_forecast(lat: float, lon: float, days: int = 8) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "current": ("temperature_2m,relative_humidity_2m,apparent_temperature,is_day,"
                    "weather_code,surface_pressure,wind_speed_10m,wind_direction_10m"),
        "hourly": ("temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
                   "precipitation_probability,weather_code,surface_pressure,visibility,"
                   "wind_speed_10m,uv_index"),
        "daily": ("weather_code,temperature_2m_max,temperature_2m_min,"
                  "apparent_temperature_max,apparent_temperature_min,sunrise,sunset,"
                  "uv_index_max,precipitation_probability_max,precipitation_sum,"
                  "wind_speed_10m_max"),
        "past_days": 1, "forecast_days": days,
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "precipitation_unit": "inch", "timezone": "auto",
    }
    r = SESSION.get(FORECAST_URL, params=params, headers={"User-Agent": USER_AGENT},
                     timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


@cached(TTL_AIR_QUALITY)
def _fetch_air_quality(lat: float, lon: float) -> dict:
    """Current AQI/pollutants, plus pollen where the provider has coverage."""
    # Pollen fields are species-specific ("birch_pollen", not "tree_pollen") — an
    # invalid name 400s the entire request, taking the AQI down with it.
    species = ("alder", "birch", "grass", "mugwort", "olive", "ragweed")
    try:
        r = SESSION.get(
            AIR_QUALITY_URL,
            params={
                "latitude": lat, "longitude": lon,
                "current": ("us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide,carbon_monoxide,"
                            + ",".join(f"{s}_pollen" for s in species)),
                "timezone": "auto",
            },
            headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        cur = r.json().get("current", {}) or {}
    except requests.RequestException as exc:
        return {"available": False, "error": str(exc)[:200]}

    pollen_values = {s: cur.get(f"{s}_pollen") for s in species}
    pollen_values = {k: v for k, v in pollen_values.items() if v is not None}
    has_pollen = bool(pollen_values)

    aqi = cur.get("us_aqi")
    return {
        "available": True,
        "aqi": aqi,
        "category": aqi_category(aqi),
        "pm2_5": cur.get("pm2_5"), "pm10": cur.get("pm10"),
        "ozone": cur.get("ozone"), "no2": cur.get("nitrogen_dioxide"),
        "co": cur.get("carbon_monoxide"),
        # Pollen is Europe-only on this provider; report absence rather than zeros.
        "pollen": pollen_values if has_pollen else None,
        "pollen_note": None if has_pollen else
                       "Pollen data isn't available for this location from the current provider.",
    }


def aqi_category(aqi: float | None) -> dict:
    """US EPA AQI band. `status` maps onto the UI's status palette."""
    if aqi is None:
        return {"label": "Unknown", "status": "none"}
    if aqi <= 50: return {"label": "Good", "status": "good"}
    if aqi <= 100: return {"label": "Moderate", "status": "moderate"}
    if aqi <= 150: return {"label": "Unhealthy for sensitive groups", "status": "high"}
    if aqi <= 200: return {"label": "Unhealthy", "status": "high"}
    if aqi <= 300: return {"label": "Very unhealthy", "status": "severe"}
    return {"label": "Hazardous", "status": "severe"}


def get_weather_detail(lat: float, lon: float, days: int = 8) -> dict:
    """Full weather payload: current, hourly (incl. yesterday), and daily summaries."""
    try:
        raw = _fetch_forecast(lat, lon, days=days)
    except requests.RequestException as exc:
        return {"ok": False, "error": f"Forecast unavailable: {exc}"}

    # Provider strategy: Open-Meteo supplies the hourly/daily series (free,
    # unlimited, no key), and Google — when a key is configured — overlays the
    # richer current conditions, EPA air quality, and pollen. That is 3 Google
    # calls per dashboard load, which sits comfortably inside the free tier,
    # while keeping the app fully functional with no key at all.
    from tools import weather_google

    use_google = weather_google.available()
    air = weather_google.get_air_quality(lat, lon) if use_google else _fetch_air_quality(lat, lon)
    if not air.get("available"):
        air = _fetch_air_quality(lat, lon)  # fall back if Google's call failed

    pollen = weather_google.get_pollen(lat, lon) if use_google else None
    today_iso = date.today().isoformat()

    # --- hourly (includes yesterday because past_days=1) ---
    h = raw.get("hourly", {})
    hourly = []
    for i, t in enumerate(h.get("time", [])):
        def at(key):
            seq = h.get(key) or []
            return seq[i] if i < len(seq) else None
        temp = at("temperature_2m")
        if temp is None:
            continue
        hourly.append({
            "time": t,
            "temp_f": round(temp, 1),
            "feels_like_f": _round_or_none(at("apparent_temperature")),
            "humidity": at("relative_humidity_2m"),
            "dew_point_f": _round_or_none(at("dew_point_2m")),
            "precip_chance": at("precipitation_probability"),
            "pressure_hpa": _round_or_none(at("surface_pressure")),
            "visibility_mi": _meters_to_miles(at("visibility")),
            "wind_mph": _round_or_none(at("wind_speed_10m")),
            "uv": at("uv_index"),
            "condition": describe_code(at("weather_code")),
            "is_past": t[:10] < today_iso,
        })

    # --- daily summaries ---
    d = raw.get("daily", {})
    daily = []
    for i, day in enumerate(d.get("time", [])):
        def at(key):
            seq = d.get(key) or []
            return seq[i] if i < len(seq) else None
        day_hours = [x for x in hourly if x["time"][:10] == day]
        high = at("temperature_2m_max")
        low = at("temperature_2m_min")
        daily.append({
            "date": day,
            "is_yesterday": day < today_iso,
            "is_today": day == today_iso,
            "condition": describe_code(at("weather_code")),
            "high_f": _round_or_none(high),
            "low_f": _round_or_none(low),
            "feels_high_f": _round_or_none(at("apparent_temperature_max")),
            "feels_low_f": _round_or_none(at("apparent_temperature_min")),
            "sunrise": at("sunrise"), "sunset": at("sunset"),
            "uv_max": at("uv_index_max"),
            "precip_chance": at("precipitation_probability_max"),
            "precip_in": at("precipitation_sum"),
            "wind_max_mph": _round_or_none(at("wind_speed_10m_max")),
            "moon": moon_phase(_parse_date(day)),
            "hours": day_hours,
        })

    # --- current conditions + today's derived indices ---
    c = raw.get("current", {}) or {}
    today = next((x for x in daily if x["is_today"]), None)
    yesterday = next((x for x in daily if x["is_yesterday"]), None)
    tomorrow_index = next((i for i, x in enumerate(daily) if x["is_today"]), None)
    tomorrow = daily[tomorrow_index + 1] if tomorrow_index is not None and tomorrow_index + 1 < len(daily) else None

    now_hour = next((x for x in hourly if not x["is_past"]), None)
    current = {
        "temp_f": _round_or_none(c.get("temperature_2m")),
        "feels_like_f": _round_or_none(c.get("apparent_temperature")),
        "humidity": c.get("relative_humidity_2m"),
        "wind_mph": _round_or_none(c.get("wind_speed_10m")),
        "wind_dir": _compass(c.get("wind_direction_10m")),
        "pressure_hpa": _round_or_none(c.get("surface_pressure")),
        "condition": describe_code(c.get("weather_code")),
        "is_day": bool(c.get("is_day", 1)),
        "dew_point_f": now_hour["dew_point_f"] if now_hour else None,
        "visibility_mi": now_hour["visibility_mi"] if now_hour else None,
        "uv": now_hour["uv"] if now_hour else None,
        "time": c.get("time"),
    }

    # Google's current observation is richer (gusts, real visibility, icon art).
    # Merge it over the Open-Meteo baseline, keeping any field Google omits.
    if use_google:
        g_current = weather_google.get_current(lat, lon)
        for key, value in g_current.items():
            if value not in (None, {}, ""):
                current[key] = value

    running = running_conditions(
        current["temp_f"], current["humidity"],
        air.get("aqi"), current.get("uv"), current.get("wind_mph"),
    )

    comparison = None
    if yesterday and today and yesterday["high_f"] is not None and today["high_f"] is not None:
        delta = round(today["high_f"] - yesterday["high_f"])
        comparison = {
            "yesterday_high_f": yesterday["high_f"],
            "yesterday_low_f": yesterday["low_f"],
            "delta_high_f": delta,
            "summary": (f"{abs(delta)}° {'warmer' if delta > 0 else 'cooler'} than yesterday"
                        if delta else "About the same as yesterday"),
        }

    return {
        "ok": True,
        "provider": "Google + Open-Meteo" if use_google else "Open-Meteo",
        "timezone": raw.get("timezone"),
        "current": current,
        "air_quality": air,
        "pollen": pollen,
        "running": running,
        "comparison": comparison,
        "tomorrow": _slim_day(tomorrow),
        "daily": daily,
        "hourly": hourly,
    }


# --- small helpers -----------------------------------------------------------

def _round_or_none(v, digits: int = 1):
    return None if v is None else round(float(v), digits)


def _meters_to_miles(v):
    return None if v is None else round(float(v) / 1609.34, 1)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _compass(deg):
    if deg is None:
        return None
    points = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return points[int((float(deg) + 11.25) % 360 / 22.5)]


def _slim_day(day):
    if not day:
        return None
    return {k: day[k] for k in ("date", "condition", "high_f", "low_f", "precip_chance")}


if __name__ == "__main__":
    import json

    data = get_weather_detail(32.7767, -96.7970)
    print("ok:", data["ok"], "| provider:", data.get("provider"))
    print("current:", json.dumps(data["current"], indent=2)[:400])
    print("air:", json.dumps(data["air_quality"], indent=2)[:300])
    print("running:", data["running"])
    print("comparison:", data["comparison"])
    print("days:", len(data["daily"]), "| hourly:", len(data["hourly"]))
    print("day[1]:", json.dumps({k: v for k, v in data["daily"][1].items() if k != "hours"}, indent=2)[:500])
