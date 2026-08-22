"""Command-line entry point for the Smart-Home Forecaster.

Usage:
  python main.py --demo                 # deterministic freeze check, NO API key needed
  python main.py --demo --address "Chicago, IL"
  python main.py --demo --force-backup  # force the NWS->Open-Meteo fallback path
  python main.py "Are my pipes at risk of freezing this week?"   # LLM agent (needs key)

The --demo path runs the raw tool pipeline so you can verify everything works
before setting up an OpenRouter key. The plain-question path runs the ReAct agent.
"""
from __future__ import annotations

import argparse
import sys

from agents.weather_risk import format_weather_report, run_weather_check


def main() -> int:
    parser = argparse.ArgumentParser(description="Smart-Home Forecaster agent")
    parser.add_argument("question", nargs="*", help="A question for the LLM agent.")
    parser.add_argument("--demo", action="store_true",
                        help="Run the deterministic freeze check (no API key needed).")
    parser.add_argument("--address", default=None,
                        help="Address to use in --demo mode (defaults to the primary home).")
    parser.add_argument("--home", default=None,
                        help="Which saved home to use (e.g. demo-001). Defaults to the primary.")
    parser.add_argument("--force-backup", action="store_true",
                        help="In --demo mode, force the Open-Meteo fallback to prove recovery.")
    parser.add_argument("--hours", type=int, default=48, help="Forecast horizon in hours.")
    args = parser.parse_args()

    if args.demo:
        address = args.address
        if not address:
            # Default to the selected home (the primary one unless --home says otherwise).
            from tools.homes import load_home

            address = load_home(args.home)["address"]
        source = "open-meteo" if args.force_backup else "auto"
        report = run_weather_check(address, horizon_hours=args.hours, weather_source=source)
        print(format_weather_report(report))
        return 0 if report.get("ok") else 1

    if not args.question:
        parser.print_help()
        return 0

    # LLM agent path (needs OPENROUTER_API_KEY).
    from agents.orchestrator import run_agent

    try:
        answer = run_agent(" ".join(args.question), home_id=args.home)
    except RuntimeError as exc:
        print(exc)
        return 1
    print("\n=== answer ===")
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
