"""Google-backed location search: autocomplete, forward and reverse geocoding.

Used when GOOGLE_MAPS_API_KEY is set. The no-key fallbacks in `geocode.py`
(US Census + Open-Meteo) still work, but they only resolve *place names* for
autocomplete — typing a street address there returns nothing useful. This module
is what makes the search box behave the way people expect from Google Maps.

Called server-side so the key stays out of the browser.
"""
from __future__ import annotations

import requests

import config
from config import GOOGLE_MAPS_API_KEY, HTTP_TIMEOUT, USER_AGENT
from tools.http import SESSION

AUTOCOMPLETE = "https://places.googleapis.com/v1/places:autocomplete"
GEOCODE = "https://maps.googleapis.com/maps/api/geocode/json"

_HEADERS = {"User-Agent": USER_AGENT}


def available() -> bool:
    # Asks config each call (not a cached constant) so switching to demo mode
    # takes effect immediately and the free Nominatim path is used instead.
    return config.google_enabled()


def suggest(query: str, lat: float | None = None, lon: float | None = None,
            limit: int = 6) -> list[dict]:
    """Autocomplete over addresses, cities, and places.

    `lat`/`lon` bias results toward what the user is currently looking at —
    without it, "5000 Maple" surfaces a match three states away instead of the
    one down the road, which is the main thing that makes an autocomplete feel
    broken.
    """
    query = (query or "").strip()
    if len(query) < 3 or not available():
        return []

    body: dict = {"input": query}
    if lat is not None and lon is not None:
        body["locationBias"] = {
            "circle": {"center": {"latitude": lat, "longitude": lon}, "radius": 50000.0}
        }

    try:
        r = SESSION.post(
            AUTOCOMPLETE,
            headers={**_HEADERS, "Content-Type": "application/json",
                     "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY},
            json=body, timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        suggestions = r.json().get("suggestions") or []
    except (requests.RequestException, ValueError):
        return []

    out = []
    for s in suggestions[:limit]:
        p = s.get("placePrediction") or {}
        text = (p.get("text") or {}).get("text")
        if not text:
            continue
        structured = p.get("structuredFormat") or {}
        out.append({
            "label": text,
            "primary": (structured.get("mainText") or {}).get("text") or text,
            "secondary": (structured.get("secondaryText") or {}).get("text") or "",
            "place_id": p.get("placeId"),
            "source": "Google Places",
        })
    return out


def geocode(address: str) -> dict | None:
    """Resolve free text (street address, city, landmark) to coordinates."""
    if not available() or not address:
        return None
    try:
        r = SESSION.get(GEOCODE, params={"key": GOOGLE_MAPS_API_KEY, "address": address},
                         headers=_HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None

    if data.get("status") != "OK" or not data.get("results"):
        return None
    top = data["results"][0]
    loc = top["geometry"]["location"]
    # "approximate" location types mean the point is a region centroid, not a rooftop.
    location_type = top["geometry"].get("location_type", "")
    return {
        "latitude": float(loc["lat"]),
        "longitude": float(loc["lng"]),
        "matched_address": top.get("formatted_address", address),
        "source": "Google Geocoding",
        "approximate": location_type in ("APPROXIMATE", "GEOMETRIC_CENTER"),
    }


def reverse(lat: float, lon: float) -> dict | None:
    """Coordinates -> a human address, for map clicks and browser geolocation."""
    if not available():
        return None
    try:
        r = SESSION.get(GEOCODE, params={"key": GOOGLE_MAPS_API_KEY, "latlng": f"{lat},{lon}"},
                         headers=_HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None

    if data.get("status") != "OK" or not data.get("results"):
        return None
    results = data["results"]
    # Prefer a street address; fall back to the most specific result available.
    best = next((r_ for r_ in results if "street_address" in (r_.get("types") or [])), results[0])
    return {
        "ok": True,
        "label": best.get("formatted_address"),
        "source": "Google Geocoding",
        "approximate": False,
    }


if __name__ == "__main__":
    import json

    if not available():
        print("GOOGLE_MAPS_API_KEY not set — Google location search inactive.")
    else:
        print("unbiased 'Maple St':")
        for s in suggest("Maple St"):
            print("  -", s["label"])
        print("\nbiased to Dallas:")
        for s in suggest("Maple St", lat=32.7767, lon=-96.7970):
            print("  -", s["label"])
        print("\ngeocode:", json.dumps(geocode("5000 Maple Ave, Dallas, TX"), indent=2))
        print("reverse:", json.dumps(reverse(32.7767, -96.7970), indent=2))
