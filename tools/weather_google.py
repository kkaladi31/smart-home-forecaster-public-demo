"""Google Maps Platform weather provider.

Used when GOOGLE_MAPS_API_KEY is set; otherwise the app falls back to the
no-key stack in `weather_detail.py` (Open-Meteo + Leaflet/OSM + RainViewer), so
the repository stays runnable by anyone who clones it without a billing account.

Why bother, given the free stack already works: **pollen**. Open-Meteo's pollen
product is CAMS *Europe* only and returns nulls for US locations; Google's Pollen
API covers the US (verified live). Google also supplies condition icons and a
Universal AQI with named pollutants.

All of these are called **server-side** so the API key never reaches the browser
and never lands in the public repo. Only the Maps JS key (a separate,
referrer-restricted key) is ever exposed client-side.
"""
from __future__ import annotations

import requests

import config
from config import GOOGLE_MAPS_API_KEY, HTTP_TIMEOUT, USER_AGENT
from tools.cache import TTL_AIR_QUALITY, TTL_FORECAST, cached
from tools.http import SESSION

WEATHER_CURRENT ="https://weather.googleapis.com/v1/currentConditions:lookup"
WEATHER_DAILY = "https://weather.googleapis.com/v1/forecast/days:lookup"
WEATHER_HOURLY = "https://weather.googleapis.com/v1/forecast/hours:lookup"
AIR_QUALITY = "https://airquality.googleapis.com/v1/currentConditions:lookup"
POLLEN = "https://pollen.googleapis.com/v1/forecast:lookup"

_HEADERS = {"User-Agent": USER_AGENT}


def available() -> bool:
    # Asks config each call (not a cached constant) so switching to demo mode
    # takes effect immediately and the free Open-Meteo path is used instead.
    return config.google_enabled()


def _c_to_f(celsius) -> float | None:
    return None if celsius is None else round(celsius * 9 / 5 + 32, 1)


def _val(node, key="value"):
    """Google wraps measures in objects, but the value key varies by measure type:
    temperatures use `degrees`, distances use `distance`, speeds use `value`.
    Reading the wrong key silently yields None, so each call site names its key.
    """
    if isinstance(node, dict):
        return node.get(key)
    return node


@cached(TTL_FORECAST)
def get_current(lat: float, lon: float) -> dict:
    """Current conditions. Returns {} on any failure so callers can fall back."""
    try:
        r = SESSION.get(WEATHER_CURRENT, params={
            "key": GOOGLE_MAPS_API_KEY,
            "location.latitude": lat, "location.longitude": lon,
            "unitsSystem": "IMPERIAL",
        }, headers=_HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        d = r.json()
    except (requests.RequestException, ValueError):
        return {}

    cond = d.get("weatherCondition", {}) or {}
    wind = d.get("wind", {}) or {}
    return {
        "temp_f": _val(d.get("temperature"), "degrees"),
        "feels_like_f": _val(d.get("feelsLikeTemperature"), "degrees"),
        "dew_point_f": _val(d.get("dewPoint"), "degrees"),
        "humidity": d.get("relativeHumidity"),
        "pressure_hpa": (d.get("airPressure") or {}).get("meanSeaLevelMillibars"),
        "visibility_mi": _val(d.get("visibility"), "distance"),
        "uv": d.get("uvIndex"),
        "wind_mph": _val(wind.get("speed")),
        "wind_gust_mph": _val(wind.get("gust")),
        "wind_dir": _cardinal((wind.get("direction") or {}).get("cardinal")),
        "is_day": d.get("isDaytime"),
        "condition": {
            "label": (cond.get("description") or {}).get("text"),
            "icon_url": f"{cond['iconBaseUri']}.png" if cond.get("iconBaseUri") else None,
            "type": cond.get("type"),
        },
        "provider": "Google Weather",
    }


@cached(TTL_AIR_QUALITY)
def get_air_quality(lat: float, lon: float) -> dict:
    """Universal AQI plus named pollutants."""
    try:
        r = SESSION.post(AIR_QUALITY, params={"key": GOOGLE_MAPS_API_KEY}, json={
            "location": {"latitude": lat, "longitude": lon},
            "extraComputations": ["LOCAL_AQI", "POLLUTANT_CONCENTRATION",
                                  "DOMINANT_POLLUTANT_CONCENTRATION", "HEALTH_RECOMMENDATIONS"],
            # Ask for the US EPA index specifically. Google's default "Universal
            # AQI" runs 0-100 where HIGHER IS BETTER — the opposite of the EPA
            # scale the fallback provider reports. Mixing the two would show the
            # same number meaning opposite things depending on the provider.
            "customLocalAqis": [{"regionCode": "us", "aqi": "usa_epa"}],
        }, headers=_HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        d = r.json()
    except (requests.RequestException, ValueError):
        return {"available": False}

    indexes = d.get("indexes") or []
    primary = next((i for i in indexes if i.get("code") == "usa_epa"),
                   indexes[0] if indexes else {})
    pollutants = {
        p.get("code"): {
            "name": p.get("displayName"),
            "value": (p.get("concentration") or {}).get("value"),
            "units": (p.get("concentration") or {}).get("units"),
        }
        for p in (d.get("pollutants") or [])
    }
    recs = d.get("healthRecommendations") or {}
    return {
        "available": True,
        "aqi": primary.get("aqi"),
        "category": {"label": primary.get("category", "").replace(" air quality", "").strip().title()
                     or "Unknown",
                     "status": _aqi_status(primary.get("aqi"))},
        "dominant_pollutant": primary.get("dominantPollutant"),
        "pollutants": pollutants,
        "advice": recs.get("generalPopulation"),
        "provider": "Google Air Quality",
    }


_COMPASS_WORDS = {
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE", "SOUTHWEST": "SW",
}


def _cardinal(value: str | None) -> str | None:
    """Google spells directions out ("SOUTH_SOUTHEAST") — abbreviate to "SSE".

    Note each word must map through the table: naively taking first letters turns
    SOUTH_SOUTHEAST into "SS" rather than "SSE".
    """
    if not value:
        return None
    return "".join(_COMPASS_WORDS.get(word, word[:1]) for word in value.split("_"))


def _aqi_status(aqi) -> str:
    """Map a US EPA AQI onto the UI status palette (higher = worse)."""
    if aqi is None:
        return "none"
    if aqi <= 50: return "good"
    if aqi <= 100: return "moderate"
    if aqi <= 150: return "high"
    return "severe"


@cached(TTL_AIR_QUALITY)
def get_pollen(lat: float, lon: float) -> dict:
    """Pollen forecast — the reason this provider exists (US coverage)."""
    try:
        r = SESSION.get(POLLEN, params={
            "key": GOOGLE_MAPS_API_KEY,
            "location.latitude": lat, "location.longitude": lon,
            "days": 1,
        }, headers=_HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        d = r.json()
    except (requests.RequestException, ValueError):
        return {"available": False}

    daily = (d.get("dailyInfo") or [{}])[0]
    types = []
    for t in daily.get("pollenTypeInfo") or []:
        index = t.get("indexInfo") or {}
        types.append({
            "code": t.get("code"),
            "name": t.get("displayName"),
            "value": index.get("value"),
            "category": index.get("category"),
            "in_season": t.get("inSeason"),
        })
    reported = [t for t in types if t["value"] is not None]
    return {
        "available": bool(reported),
        "region": d.get("regionCode"),
        "types": reported or types,
        "provider": "Google Pollen",
        "note": None if reported else "No pollen readings for this location today.",
    }


if __name__ == "__main__":
    import json

    if not available():
        print("GOOGLE_MAPS_API_KEY not set — provider inactive (app uses the free stack).")
    else:
        lat, lon = 32.7767, -96.7970
        print("current:", json.dumps(get_current(lat, lon), indent=2)[:600])
        print("\nair:", json.dumps(get_air_quality(lat, lon), indent=2)[:600])
        print("\npollen:", json.dumps(get_pollen(lat, lon), indent=2)[:700])
