"""Look up ground elevation (meters) for a lat/lon.

Primary source : Open-Meteo Elevation API (free, no key, global)
Fallback source: USGS EPQS (US Geological Survey, free, no key, US only)

Why elevation matters for a *freeze* forecaster: cold air pools in low spots and
higher elevations run colder, so elevation is a useful signal when reasoning
about frost/freeze risk for a specific property.
"""
from __future__ import annotations

import requests

from config import HTTP_TIMEOUT, USER_AGENT
from tools.cache import TTL_ELEVATION, cached
from tools.http import SESSION

OPEN_METEO_URL = "https://api.open-meteo.com/v1/elevation"
USGS_EPQS_URL = "https://epqs.nationalmap.gov/v1/json"


def _via_open_meteo(lat: float, lon: float) -> float | None:
    resp = SESSION.get(
        OPEN_METEO_URL,
        params={"latitude": lat, "longitude": lon},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    elevations = resp.json().get("elevation")
    if elevations:
        return float(elevations[0])
    return None


def _via_usgs(lat: float, lon: float) -> float | None:
    resp = SESSION.get(
        USGS_EPQS_URL,
        params={"x": lon, "y": lat, "units": "Meters", "wkid": 4326, "includeDate": "false"},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    value = resp.json().get("value")
    return float(value) if value not in (None, "") else None


@cached(TTL_ELEVATION)
def get_elevation(latitude: float, longitude: float) -> dict:
    """Return ground elevation in meters and feet for the given coordinates.

    Returns:
        On success: {"ok": True, elevation_m, elevation_ft, source}.
        On failure: {"ok": False, "error": ...}.
    """
    for resolver, name in ((_via_open_meteo, "Open-Meteo"), (_via_usgs, "USGS EPQS")):
        try:
            meters = resolver(latitude, longitude)
            if meters is not None:
                return {
                    "ok": True,
                    "elevation_m": round(meters, 1),
                    "elevation_ft": round(meters * 3.28084, 1),
                    "source": name,
                }
        except requests.RequestException:
            continue
    return {"ok": False, "error": "Could not determine elevation for those coordinates."}


if __name__ == "__main__":
    import json

    # Dallas, TX approx.
    print(json.dumps(get_elevation(32.7767, -96.7970), indent=2))
