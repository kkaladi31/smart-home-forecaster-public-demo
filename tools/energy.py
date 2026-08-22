"""Look up residential energy prices for a US state.

Primary source: US Energy Information Administration (EIA) API v2 — live, official
state-level residential prices for electricity (cents/kWh) and natural gas ($/Mcf).
Requires a free EIA_API_KEY.

If EIA is unavailable (no key, rate limit, outage), we fall back to documented
national averages rather than failing — the cost analysis still works, and the
result clearly reports `live: False` plus the as-of date so the agent can say so.

Users can always override with their own rate from their bill, which is the most
accurate option (especially in deregulated markets like Texas, where the address
determines the delivery utility but NOT the retail rate).
"""
from __future__ import annotations

import re

import requests

from config import EIA_API_KEY, HTTP_TIMEOUT, USER_AGENT
from tools.http import SESSION

EIA_ELEC_URL = "https://api.eia.gov/v2/electricity/retail-sales/data/"
EIA_GAS_URL = "https://api.eia.gov/v2/natural-gas/pri/sum/data/"

# Documented fallback: US average residential prices. Used only when EIA is
# unavailable; always reported with live=False so the answer stays honest.
FALLBACK = {
    "electricity_cents_kwh": 16.5,
    "gas_dollars_mcf": 15.5,
    "as_of": "2025 US average (EIA published averages)",
}

_STATE_RE = re.compile(r"\b([A-Z]{2})\b(?:\s+\d{5})?\s*$")


def state_from_address(address: str) -> str | None:
    """Best-effort two-letter state code from a US address string."""
    for part in reversed([p.strip() for p in address.split(",")]):
        m = _STATE_RE.search(part.upper())
        if m:
            return m.group(1)
    return None


def _eia_electricity(state: str) -> tuple[float, str] | None:
    resp = SESSION.get(
        EIA_ELEC_URL,
        params={
            "api_key": EIA_API_KEY, "frequency": "monthly", "data[0]": "price",
            "facets[stateid][]": state, "facets[sectorid][]": "RES",
            "sort[0][column]": "period", "sort[0][direction]": "desc", "length": "1",
        },
        headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json().get("response", {}).get("data", [])
    if not rows:
        return None
    return float(rows[0]["price"]), rows[0]["period"]


def _eia_gas(state: str) -> tuple[float, str] | None:
    resp = SESSION.get(
        EIA_GAS_URL,
        params={
            "api_key": EIA_API_KEY, "frequency": "monthly", "data[0]": "value",
            "facets[series][]": f"N3010{state}3",
            "sort[0][column]": "period", "sort[0][direction]": "desc", "length": "1",
        },
        headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json().get("response", {}).get("data", [])
    if not rows or rows[0].get("value") is None:
        return None
    return float(rows[0]["value"]), rows[0]["period"]


def get_energy_prices(state: str = "WA") -> dict:
    """Return residential energy prices for a state.

    Returns:
        {"ok": True, live, state, electricity_cents_kwh, gas_dollars_mcf,
         period, source, note}. `live` False means the documented fallback was used.
    """
    state = (state or "WA").upper()[:2]
    if EIA_API_KEY:
        try:
            elec = _eia_electricity(state)
            gas = _eia_gas(state)
            if elec:
                return {
                    "ok": True, "live": True, "state": state,
                    "electricity_cents_kwh": elec[0],
                    "gas_dollars_mcf": gas[0] if gas else FALLBACK["gas_dollars_mcf"],
                    "period": elec[1],
                    "source": "US Energy Information Administration (EIA) API v2",
                    "note": "" if gas else "Gas price unavailable for this state; used US average.",
                }
        except requests.RequestException:
            pass  # fall through to the documented fallback

    return {
        "ok": True, "live": False, "state": state,
        "electricity_cents_kwh": FALLBACK["electricity_cents_kwh"],
        "gas_dollars_mcf": FALLBACK["gas_dollars_mcf"],
        "period": FALLBACK["as_of"],
        "source": "Documented US average (EIA live data unavailable)",
        "note": "Live EIA prices were unavailable, so this uses a published US average. "
                "For an accurate figure, provide the rate from your utility bill.",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(get_energy_prices("MN"), indent=2))
    print("state_from_address:", state_from_address("1412 Larkspur Lane, Minneapolis, MN 55409"))
    print("state_from_address:", state_from_address("5000 Maple Street, Dallas, TX 75201"))
