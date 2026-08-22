"""Is the pinned free model still real? Run this before you demo anything.

WHY THIS EXISTS
---------------
On 2026-08-21 OpenRouter withdrew `openai/gpt-oss-20b:free` from the free tier.
The slug did not slow down or degrade — it started returning

    404  This model is unavailable for free. The paid version is available now

so every LLM-dependent evaluation case failed simultaneously and the demo could
not answer a single question. Nothing in the repo had changed.

A free slug is not a stable dependency. It is someone else's promotional pricing,
and it can be withdrawn between one day and the next. The demo pins ONE model on
purpose — a recorded run has to be reproducible, and silently failing over to a
different model mid-demo would make the trace unexplainable — so the pin needs a
cheap way to be checked rather than automatic failover.

    python scripts/check_free_model.py            # is the pin alive?
    python scripts/check_free_model.py --all      # also test every fallback
    python scripts/check_free_model.py --list     # every free tool-calling slug

Exit code is 0 when the pinned model works, 1 when it does not — so it can gate
a demo-day checklist.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from config import FREE_LLM_FALLBACKS, FREE_LLM_MODEL, OPENROUTER_API_KEY  # noqa: E402

MODELS_URL = "https://openrouter.ai/api/v1/models"
# Small enough to cost nothing and still prove the slug answers.
PROBE = "Reply with exactly one word: ready"


def free_tool_models() -> list[tuple[str, int]]:
    """Every free slug that advertises tool calling, widest context first."""
    data = requests.get(MODELS_URL, timeout=30).json()["data"]
    out = [
        (m["id"], m.get("context_length") or 0)
        for m in data
        if m["id"].endswith(":free") and "tools" in (m.get("supported_parameters") or [])
    ]
    return sorted(out, key=lambda pair: -pair[1])


def probe(model: str) -> tuple[bool, str, float]:
    """Actually call the model. Listing it is not proof that it answers."""
    started = time.perf_counter()
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": model, "max_tokens": 12,
                  "messages": [{"role": "user", "content": PROBE}]},
            timeout=90,
        )
        elapsed = time.perf_counter() - started
        if r.status_code == 200:
            body = r.json()
            text = (body["choices"][0]["message"].get("content") or "").strip()
            return True, text[:40] or "(empty reply)", elapsed
        detail = ""
        try:
            detail = (r.json().get("error") or {}).get("message", "")
        except ValueError:
            detail = r.text[:120]
        return False, f"HTTP {r.status_code}: {detail[:110]}", elapsed
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:120], time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true",
                        help="also probe every configured fallback")
    parser.add_argument("--list", action="store_true",
                        help="list every free tool-calling slug on OpenRouter")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("No OPENROUTER_API_KEY. Put it in .env first.")
        return 2

    if args.list:
        try:
            models = free_tool_models()
        except Exception as exc:
            print(f"Could not reach OpenRouter: {exc}")
            return 2
        print(f"{len(models)} free slugs advertising tool support "
              f"(context, widest first):\n")
        for slug, ctx in models:
            mark = "  <- pinned" if slug == FREE_LLM_MODEL else ""
            print(f"  {ctx:>9,}  {slug}{mark}")
        print("\nAdvertising tool support is not the same as working. Probe before "
              "trusting one:\n  python scripts/check_free_model.py --all")
        return 0

    print(f"pinned free model: {FREE_LLM_MODEL}\n")
    ok, detail, elapsed = probe(FREE_LLM_MODEL)
    print(f"  {'OK  ' if ok else 'DEAD'}  {elapsed:5.1f}s  {detail}")

    if not ok:
        print("\nThe pinned model is not answering. The demo cannot run on it.\n"
              "Configured fallbacks, probed now:")
        for slug in FREE_LLM_FALLBACKS:
            f_ok, f_detail, f_elapsed = probe(slug)
            print(f"  {'OK  ' if f_ok else 'no  '}  {f_elapsed:5.1f}s  "
                  f"{slug}\n          {f_detail}")
        print("\nRepoint FREE_LLM_MODEL in config.py at one that answered, or set\n"
              "FREE_LLM_MODEL=<slug> in .env to override without editing code.\n"
              "Then re-run: python eval/run_eval.py")
        return 1

    if args.all:
        print("\nfallbacks:")
        for slug in FREE_LLM_FALLBACKS:
            f_ok, f_detail, f_elapsed = probe(slug)
            print(f"  {'OK  ' if f_ok else 'no  '}  {f_elapsed:5.1f}s  {slug}")
            if not f_ok:
                print(f"          {f_detail}")

    print("\nPinned model is alive. Safe to demo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
