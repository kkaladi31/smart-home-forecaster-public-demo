"""Turn a street address into latitude/longitude coordinates.

Primary source : US Census Geocoder     (free, no key, authoritative for US street addresses)
Fallback source: Open-Meteo Geocoding    (free, no key, global; matches city/place names)

Both are public services, satisfying the capstone's "public data only" rule.
The fallback is what demonstrates the agent's ability to *recover from missteps*
(ReAct): if the Census service returns nothing (e.g. a city-only query, or a
brand-new/synthetic address it doesn't know), we transparently try the next
source and report which one answered.

Note: the OpenStreetMap/Nominatim public endpoint was evaluated first but it
returns HTTP 403 for generic automated clients under its usage policy, so
Open-Meteo's geocoder is used instead as the more reliable free fallback.
"""
from __future__ import annotations

import re

import requests

from config import HTTP_TIMEOUT, USER_AGENT
from tools.cache import TTL_GEOCODE, cached
from tools.http import SESSION

CENSUS_URL ="https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"


def _via_census(address: str) -> dict | None:
    """Try the US Census one-line geocoder (best for full US street addresses)."""
    params = {"address": address, "benchmark": "Public_AR_Current", "format": "json"}
    resp = SESSION.get(
        CENSUS_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    matches = resp.json().get("result", {}).get("addressMatches", [])
    if not matches:
        return None
    top = matches[0]
    coords = top["coordinates"]  # note: x = longitude, y = latitude
    return {
        "latitude": float(coords["y"]),
        "longitude": float(coords["x"]),
        "matched_address": top.get("matchedAddress", address),
        "source": "US Census Geocoder",
    }


def _place_candidates(address: str) -> list[str]:
    """Extract likely place-name queries from a free-text address, best guess first.

    Open-Meteo's geocoder matches *place names* ("Dallas"), not full street
    addresses or "City, ST" strings, so we derive candidate city names and try
    them in priority order.
    """
    parts = [p.strip() for p in address.split(",") if p.strip()]
    candidates: list[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if value and not value.isdigit() and value not in candidates:
            candidates.append(value)

    n = len(parts)
    if n >= 3:
        add(parts[-2])            # "street, CITY, state zip" -> city is second-to-last
    if n == 2:
        add(parts[0])             # "CITY, ST" -> city is first
    # Clean each part of a leading street number, ZIP code, and trailing state code.
    for p in parts:
        c = re.sub(r"^\d+\s+", "", p)
        c = re.sub(r"\b\d{5}(-\d{4})?\b", "", c)
        c = re.sub(r"\b[A-Z]{2}\b\s*$", "", c).strip()
        add(c)
    add(address)                  # whole string as a last resort
    return candidates


def _via_open_meteo_geo(address: str) -> dict | None:
    """Try Open-Meteo's geocoder using extracted place-name candidates."""
    for name in _place_candidates(address):
        resp = SESSION.get(
            OPEN_METEO_GEO_URL,
            params={"name": name, "count": 1, "language": "en", "format": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if results:
            top = results[0]
            label = ", ".join(
                str(x) for x in (top.get("name"), top.get("admin1"), top.get("country_code")) if x
            )
            return {
                "latitude": float(top["latitude"]),
                "longitude": float(top["longitude"]),
                "matched_address": label,
                "source": "Open-Meteo Geocoding",
                "approximate": True,  # place-level, not street-level
            }
    return None


@cached(TTL_GEOCODE)
def geocode_address(address: str) -> dict:
    """Resolve a free-text address to coordinates.

    Tries Google first when a key is configured (it handles street addresses,
    landmarks, and businesses), then falls back to the no-key providers.

    Args:
        address: e.g. "1600 Amphitheatre Parkway, Mountain View, CA" or "Dallas, TX".

    Returns:
        On success: {"ok": True, latitude, longitude, matched_address, source, ...}.
        On failure: {"ok": False, "error": <human-readable reason>}.
        (We return errors as data rather than raising so the agent can *reason*
        about them and try another approach.)
    """
    from tools import geocode_google

    if geocode_google.available():
        hit = geocode_google.geocode(address)
        if hit:
            return {"ok": True, **hit}

    for resolver in (_via_census, _via_open_meteo_geo):
        try:
            result = resolver(address)
            if result:
                return {"ok": True, **result}
        except requests.RequestException:
            # Network/HTTP problem with this provider — fall through to the next.
            continue
    return {
        "ok": False,
        "error": f"Could not geocode address: {address!r}. "
        "Try a full street address, or 'City, ST'.",
    }


def suggest_addresses(
    query: str,
    count: int = 5,
    lat: float | None = None,
    lon: float | None = None,
) -> list[dict]:
    """Return location candidates for a typeahead / autocomplete field.

    With a Google key this is full Places autocomplete — street addresses,
    businesses, landmarks — biased toward `lat`/`lon` so nearby matches rank
    first. Without a key it falls back to Open-Meteo, which only matches place
    names (cities), so street addresses will not appear.

    Short queries are ignored so we don't call out on every keystroke.
    """
    query = (query or "").strip()
    if len(query) < 3:
        return []

    from tools import geocode_google

    if geocode_google.available():
        hits = geocode_google.suggest(query, lat=lat, lon=lon, limit=count)
        if hits:
            return hits
    try:
        resp = SESSION.get(
            OPEN_METEO_GEO_URL,
            params={"name": query, "count": count, "language": "en", "format": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except requests.RequestException:
        return []

    out = []
    for r in results:
        label = ", ".join(
            str(x) for x in (r.get("name"), r.get("admin1"), r.get("country_code")) if x
        )
        out.append({
            "label": label,
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "source": "Open-Meteo Geocoding",
        })
    return out


@cached(TTL_GEOCODE)
def reverse_geocode(latitude: float, longitude: float) -> dict:
    """Turn coordinates back into a human-readable place.

    Used by map clicks and by "use my location". Google returns a full street
    address; the Census fallback returns only a city/county, and failing that we
    show formatted coordinates so the UI always has a label.
    """
    from tools import geocode_google

    if geocode_google.available():
        hit = geocode_google.reverse(latitude, longitude)
        if hit:
            return hit

    fallback = {"ok": True, "label": f"{latitude:.4f}, {longitude:.4f}",
                "source": "coordinates", "approximate": True}
    try:
        resp = SESSION.get(
            "https://geocoding.geo.census.gov/geocoder/geographies/coordinates",
            params={"x": longitude, "y": latitude, "benchmark": "Public_AR_Current",
                    "vintage": "Current_Current", "format": "json",
                    "layers": "Incorporated Places,Counties,States"},
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        geos = resp.json().get("result", {}).get("geographies", {})
        place = (geos.get("Incorporated Places") or [{}])[0].get("NAME")
        county = (geos.get("Counties") or [{}])[0].get("NAME")
        state = (geos.get("States") or [{}])[0].get("STUSAB")
        parts = [p for p in (place or county, state) if p]
        if parts:
            return {"ok": True, "label": ", ".join(parts), "source": "US Census",
                    "county": county, "state": state, "approximate": False}
    except (requests.RequestException, KeyError, IndexError, ValueError):
        pass
    return fallback


if __name__ == "__main__":
    import json

    for q in ["1600 Pennsylvania Avenue NW, Washington, DC 20500",
              "5000 Maple Street, Dallas, TX 75201",  # synthetic -> fallback
              "Dallas, TX"]:
        print(f"--- {q}")
        print(json.dumps(geocode_address(q), indent=2))
    print("--- suggest 'Minneap'")
    print(json.dumps(suggest_addresses("Minneap"), indent=2))
    print("--- reverse 32.78,-96.80")
    print(json.dumps(reverse_geocode(32.7830, -96.8067), indent=2))
