"""The Cost specialist: a sub-agent for utility-cost and savings analysis.

One of the project's specialized agents (alongside Weather, Policy, and Advisor).
It follows the same principle as the other numeric features: the *arithmetic* is
done by deterministic tools (live EIA prices + the savings estimator), and the LLM
only explains, prioritizes, and presents. That keeps the dollar figures auditable.

Gathers: home profile -> state -> live energy prices (EIA) -> utility providers
(synthetic directory) -> itemized savings estimate, then writes the recommendation.
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

import config
from agents.llm import build_llm
from tools.energy import get_energy_prices, state_from_address
from tools.homes import current_home_id, load_home, resolve_home_id
from tools.savings import estimate_savings

_DATA = config.data_root()
_UTILITIES_PATH = _DATA / "utilities.csv"


def load_utilities(home_id: str | None = None) -> list[dict]:
    """The synthetic utility-provider directory (rates + contacts) for one home."""
    wanted = resolve_home_id(home_id) if home_id else current_home_id()
    with open(_UTILITIES_PATH, encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["home_id"] == wanted]


def _filter_overdue(home: dict) -> bool:
    """True if the HVAC filter is past its replacement interval."""
    hvac = home.get("systems", {}).get("hvac", {})
    last, interval = hvac.get("filter_last_changed"), hvac.get("filter_interval_days")
    if not last or not interval:
        return False
    try:
        last_date = datetime.strptime(last, "%Y-%m-%d").date()
    except ValueError:
        return False
    return (date.today() - last_date).days > int(interval)


def run_cost_analysis(
    question: str = "How can I lower my utility bills?",
    user_rate_cents_kwh: float | None = None,
    model: str | None = None,
    home_id: str | None = None,
) -> dict:
    """Analyze utility costs and return itemized savings with a written summary.

    Args:
        question: the user's question, for framing the write-up.
        user_rate_cents_kwh: the user's ACTUAL rate from their bill, if they gave
            one — more accurate than a state average (important in deregulated
            markets, where the address does not determine the retail rate).
        home_id: which saved home to analyze; defaults to the active home.
    """
    home_id = resolve_home_id(home_id) if home_id else current_home_id()
    home = load_home(home_id)
    # The profile's own jurisdiction is authoritative; parsing the address string
    # is the fallback for a profile that predates that field.
    state = (home.get("jurisdiction") or {}).get("state") \
        or state_from_address(home.get("address", "")) \
        or "WA"
    prices = get_energy_prices(state)
    utilities = load_utilities(home_id)

    rate = user_rate_cents_kwh or prices["electricity_cents_kwh"]
    rate_source = (
        "the rate you provided" if user_rate_cents_kwh
        else f"{prices['source']} ({prices['period']})"
    )

    savings = estimate_savings(
        electricity_cents_kwh=rate,
        square_feet=home.get("square_feet"),
        climate_zone=home.get("climate_zone"),
        water_heater_temp_f=home.get("systems", {}).get("water_heater", {}).get("thermostat_f"),
        filter_overdue=_filter_overdue(home),
    )

    provider_lines = [
        f"{u['service']}: {u['provider_name']} ({u['type']}) — {u['rate']} {u['rate_unit']}, "
        f"${u['fixed_monthly_usd']}/mo fixed, phone {u['phone']}"
        for u in utilities
    ]

    llm = build_llm(model=model)
    sys = SystemMessage(content=(
        "You are the home utility-cost specialist. Using ONLY the computed figures given to "
        "you, write a concise, practical answer: (1) the estimated current annual electricity "
        "cost, (2) a prioritized list of savings measures with their dollar amounts and effort, "
        "(3) the total potential annual savings. State the rate you used and its source, and "
        "note these are estimates based on typical usage. Do NOT invent new numbers or measures "
        "— use exactly the ones provided. If the prices are not live, say so. Keep it tight."
    ))
    hum = HumanMessage(content=(
        f"QUESTION: {question}\n\n"
        f"HOME: {home.get('label')} — {home.get('address')}.\n"
        f"{home.get('square_feet')} sq ft, climate zone {home.get('climate_zone')}, "
        f"{home.get('dwelling_type')}, built {home.get('year_built')}.\n"
        f"RATE USED: {rate} cents/kWh (source: {rate_source}); live={prices['live']}\n"
        f"GAS PRICE: ${prices['gas_dollars_mcf']}/Mcf\n\n"
        f"COMPUTED SAVINGS:\n{json.dumps(savings, indent=2)}\n\n"
        f"UTILITY PROVIDERS ON FILE (synthetic demo data):\n" + "\n".join(provider_lines)
    ))
    write_up = llm.invoke([sys, hum]).content

    return {
        "answer": write_up,
        "home_id": home_id,
        "home_label": home.get("label"),
        "prices": prices,
        "rate_used_cents_kwh": rate,
        "rate_source": rate_source,
        "savings": savings,
        "utilities": utilities,
    }


if __name__ == "__main__":
    result = run_cost_analysis()
    print(f"[home {result['home_label']} ({result['home_id']})]")
    print(f"[rate {result['rate_used_cents_kwh']}c/kWh from {result['rate_source']}]")
    print(f"[total potential savings ${result['savings']['total_annual_savings_usd']}/yr]\n")
    print(result["answer"])
