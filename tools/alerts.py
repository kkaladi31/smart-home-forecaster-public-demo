"""Fetch active official weather advisories/watches/warnings for a location.

Source: US National Weather Service active-alerts API (free, no key, US, govt).
    https://api.weather.gov/alerts/active?point=lat,lon

This is the authoritative feed for *any* weather advisory — Excessive Heat
Warning, Red Flag Warning (fire), High Wind, Flood Watch, Winter Storm, Severe
Thunderstorm, and so on — so one tool covers all hazard types beyond our own
freeze/heat calculations. (Alerts are US-only; other locations return none.)
"""
from __future__ import annotations

import requests

from config import HTTP_TIMEOUT, USER_AGENT
from tools.cache import TTL_ALERTS, cached
from tools.home_precautions import precautions_for
from tools.http import SESSION

NWS_ALERTS_URL ="https://api.weather.gov/alerts/active"

# NWS severity ranking, most serious first, for sorting.
_SEVERITY_RANK = {"Extreme": 0, "Severe": 1, "Moderate": 2, "Minor": 3, "Unknown": 4}


@cached(TTL_ALERTS)
def get_weather_alerts(latitude: float, longitude: float) -> dict:
    """Return active NWS alerts for a point, most severe first.

    Returns:
        On success: {"ok": True, "count": N, "alerts": [ {event, severity, urgency,
        headline, instruction, expires, area} ]}. count 0 means no active alerts.
        On failure: {"ok": False, "error": ...}.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    try:
        resp = SESSION.get(
            NWS_ALERTS_URL,
            params={"point": f"{latitude},{longitude}"},
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"Could not fetch alerts: {exc}"}

    features = resp.json().get("features", [])
    alerts = []
    for f in features:
        p = f.get("properties", {})
        instruction = (p.get("instruction") or "").strip()
        alerts.append(
            {
                "event": p.get("event", ""),
                "severity": p.get("severity", "Unknown"),
                "urgency": p.get("urgency", ""),
                "headline": p.get("headline", ""),
                "instruction": instruction[:400],
                "expires": p.get("expires", ""),
                "area": p.get("areaDesc", ""),
                # What the advisory means for the BUILDING. The NWS instruction
                # tells people how to stay safe and stops there; protecting the
                # house is this product's job, so it is attached here rather than
                # in the UI — the agent reads the same field.
                "home_actions": precautions_for(p.get("event", ""))["actions"],
            }
        )
    alerts.sort(key=lambda a: _SEVERITY_RANK.get(a["severity"], 4))
    return {"ok": True, "count": len(alerts), "alerts": alerts}


if __name__ == "__main__":
    import json

    # Phoenix, AZ — often under heat alerts in summer.
    print(json.dumps(get_weather_alerts(33.4484, -112.0740), indent=2)[:1500])
