"""Grounded web search via DuckDuckGo (no API key required).

Used by the Advisor to fetch external DIY/product guidance and return real,
citable source URLs. Fails gracefully: if search is unavailable (rate limit,
network), it returns ok=False and the Advisor proceeds on its other grounding
rather than crashing.

To upgrade to higher-quality results later, swap this for Tavily (free tier,
needs TAVILY_API_KEY) — keep the same return shape.
"""
from __future__ import annotations

try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS  # type: ignore


def web_search(query: str, max_results: int = 4) -> dict:
    """Search the web and return a few results with titles, URLs, and snippets.

    Returns:
        {"ok": True, "query": ..., "results": [{title, url, snippet}]}, or
        {"ok": False, "error": ..., "results": []} on failure.
    """
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        results = [
            {"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")}
            for h in hits
        ]
        return {"ok": True, "query": query, "results": results}
    except Exception as exc:  # keep the Advisor alive if search is unavailable
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "results": []}


if __name__ == "__main__":
    import json

    print(json.dumps(web_search("how to safely hang a 20 lb mirror on drywall"), indent=2)[:1200])
