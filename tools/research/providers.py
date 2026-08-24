"""Web-search providers behind one result shape.

Replaces a single hard-coded DuckDuckGo call (`tools/websearch.py`, since
deleted) which had no caching, no deduplication, no telemetry, no rate-limit
handling, and did not use the project's pooled retrying session.

Two providers:

  duckduckgo  free, keyless. The DEMO default, and the fallback everywhere.
  tavily      keyed, 1000 credits/month free. FULL builds only. Returns cleaned
              page CONTENT rather than a snippet, which is the entire reason it
              was chosen — the old pipeline showed the model 160 characters per
              result and asked it to reason from that.

Selection is by build profile, not by preference: `config.provider_allowed`
refuses every keyed provider in a demo build even when the key is present, so the
published artifact cannot quietly depend on one.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import config
import telemetry
from tools.cache import TTL_RESEARCH, cached
from tools.http import SESSION

TAVILY_URL = "https://api.tavily.com/search"

# Tracking parameters carry no meaning for a document's identity, so two URLs that
# differ only by campaign tags are the same source and must not both occupy a slot
# in a six-passage evidence pack.
_JUNK_PARAMS = re.compile(
    r"^(utm_|fbclid$|gclid$|mc_[ce]id$|igshid$|ref$|ref_src$|s?ref$|source$|"
    r"campaign$|_ga$|yclid$|msclkid$)", re.I)

# Providers that have started refusing us. Sticky for the process, modelled on
# `memory/rerank.py:_UNAVAILABLE`: once DuckDuckGo rate-limits, hammering it makes
# the block longer and costs the user a slow failure on every subsequent question.
_UNAVAILABLE: set[str] = set()
_LOCK = threading.Lock()


@dataclass
class Result:
    """One search hit, normalised across providers."""
    title: str
    url: str
    snippet: str = ""
    content: str = ""          # full cleaned page text, when the provider supplies it
    provider: str = ""
    rank: int = 0              # position in that provider's own ordering
    score: float | None = None  # provider-reported relevance, if any
    extra: dict = field(default_factory=dict)

    @property
    def domain(self) -> str:
        try:
            host = urlsplit(self.url).netloc.lower()
            return host[4:] if host.startswith("www.") else host
        except ValueError:
            return ""

    @property
    def best_text(self) -> str:
        """Whatever this provider actually gave us to reason over."""
        return self.content or self.snippet

    def as_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet,
                "domain": self.domain, "provider": self.provider, "rank": self.rank}


def canonical_url(url: str) -> str:
    """A URL's identity for deduplication.

    Drops the scheme, `www.`, tracking parameters, fragments, AMP suffixes and a
    trailing slash, then sorts what remains. `https://www.x.com/a/?utm_source=q#top`
    and `http://x.com/a` collapse to the same key.
    """
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/(amp|amp\.html)$", "", parts.path.rstrip("/")) or "/"
    query = urlencode(sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if not _JUNK_PARAMS.match(k)))
    return urlunsplit(("", host, path, query, ""))


def dedupe(results: list[Result], per_domain: int = 2) -> list[Result]:
    """Collapse duplicate URLs and cap how much of the pack one domain may own.

    The per-domain cap is a quality control, not a fairness one: a single site
    returning four pages of its own content crowds out corroboration, and an
    evidence pack whose sources all agree because they are the same source is
    worse than a smaller one.
    """
    seen: set[str] = set()
    per_host: dict[str, int] = {}
    kept: list[Result] = []
    for r in results:
        key = canonical_url(r.url)
        if not key or key in seen:
            continue
        host = r.domain
        if per_host.get(host, 0) >= per_domain:
            continue
        seen.add(key)
        per_host[host] = per_host.get(host, 0) + 1
        kept.append(r)
    return kept


# --- providers -----------------------------------------------------------------

def _duckduckgo(query: str, max_results: int) -> list[Result]:
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS  # type: ignore[no-redef]

    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))
    return [
        Result(title=h.get("title", ""), url=h.get("href") or h.get("url", ""),
               snippet=h.get("body", ""), provider="duckduckgo", rank=i)
        for i, h in enumerate(hits)
    ]


def _tavily(query: str, max_results: int) -> list[Result]:
    key = config.TAVILY_API_KEY
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not configured")
    response = SESSION.post(
        TAVILY_URL,
        json={
            "api_key": key,
            "query": query,
            "max_results": max_results,
            # The point of paying for Tavily: cleaned article text, so the model
            # reasons over the page rather than over a search-result blurb.
            "include_raw_content": True,
            "search_depth": "basic",     # "advanced" costs 2 credits per call
        },
        timeout=config.HTTP_TIMEOUT * 2,  # it fetches pages before answering
    )
    response.raise_for_status()
    payload = response.json()
    return [
        Result(title=item.get("title", ""), url=item.get("url", ""),
               snippet=item.get("content", "") or "",
               content=(item.get("raw_content") or "")[:20000],
               provider="tavily", rank=i, score=item.get("score"))
        for i, item in enumerate(payload.get("results", []))
    ]


def _fixtures(query: str, max_results: int) -> list[Result]:
    from tools.research import fixtures

    return fixtures.search(query, max_results)


_PROVIDERS = {"duckduckgo": _duckduckgo, "tavily": _tavily, "fixtures": _fixtures}


def available_providers() -> list[str]:
    """Providers this build may use, best first.

    Tavily leads when allowed because it returns page content; DuckDuckGo is
    always present as the fallback and is the only option a demo build has.
    """
    # Fixture mode is exclusive, NOT a preference with live fallback. The point
    # of a reproducible run is that it cannot quietly reach the network — a
    # fallback would mean a stale snapshot silently became a live search, and the
    # run would look reproducible while not being it. A missing fixture must be a
    # loud failure, so `fixtures.search` raises rather than returning nothing.
    if config.research_fixtures_enabled():
        return ["fixtures"]

    order = []
    if config.provider_allowed("tavily") and config.TAVILY_API_KEY:
        order.append("tavily")
    order.append("duckduckgo")
    return [p for p in order if p not in _UNAVAILABLE] or ["duckduckgo"]


def reset_unavailable() -> None:
    """Clear the sticky rate-limit flags.

    For the fixture capture, which pauses and retries deliberately. In normal
    operation the flag SHOULD stay set for the process — hammering a provider
    that is refusing us makes the block longer and costs the user a slow failure
    on every subsequent question.
    """
    with _LOCK:
        _UNAVAILABLE.clear()


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(m in text for m in ("ratelimit", "rate limit", "429", "too many requests"))


@cached(TTL_RESEARCH)
def search(query: str, max_results: int = 6, provider: str | None = None) -> dict:
    """Search the web through the best available provider, with fallback.

    Returns `{ok, query, provider, results: [Result-as-dict...], error}` — and the
    live `Result` objects under `"_results"` for in-process callers, since the
    cache stores whatever this returns.

    Never raises. A failed search should cost the caller its citations, not its
    answer, which is why the Advisor has always treated web grounding as
    best-effort.
    """
    candidates = [provider] if provider else available_providers()
    last_error = ""

    for name in candidates:
        fn = _PROVIDERS.get(name)
        if fn is None:
            continue
        try:
            with telemetry.span("research", "research.search",
                                f"{name}: {query[:60]}") as span:
                results = dedupe(fn(query, max_results))
                span.update({"provider": name, "results": len(results)})
            return {"ok": True, "query": query, "provider": name,
                    "results": [r.as_dict() for r in results], "_results": results,
                    "error": ""}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if _is_rate_limit(exc):
                with _LOCK:
                    _UNAVAILABLE.add(name)
                telemetry.record("research", "research.provider_blocked",
                                 f"{name} is rate-limiting; not retrying this process",
                                 level="warn", data={"provider": name})
            else:
                telemetry.record("research", "research.provider_failed",
                                 f"{name} failed: {last_error}", level="warn",
                                 data={"provider": name})

    return {"ok": False, "query": query, "provider": "", "results": [],
            "_results": [], "error": last_error or "no provider available"}


if __name__ == "__main__":
    print("profile:", config.build_profile(), "| providers:", available_providers())

    print("\n--- canonical_url collapses the variants that matter ---")
    for u in ["https://www.example.com/guide/?utm_source=x&b=2#top",
              "http://example.com/guide?b=2",
              "https://example.com/guide/amp",
              "https://example.com/other"]:
        print(f"  {u:<52} -> {canonical_url(u)}")

    print("\n--- dedupe caps one domain at 2 ---")
    rs = [Result(title=f"t{i}", url=u, provider="x", rank=i) for i, u in enumerate([
        "https://a.com/1", "https://www.a.com/1?utm_source=z", "https://a.com/2",
        "https://a.com/3", "https://b.com/1"])]
    print("  kept:", [r.url for r in dedupe(rs)])

    print("\n--- live search ---")
    out = search("how to hang a heavy mirror on drywall safely", max_results=5)
    print(f"  ok={out['ok']} provider={out['provider']} error={out['error'][:60]}")
    for r in out["results"]:
        body = next((x.best_text for x in out["_results"] if x.url == r["url"]), "")
        print(f"    [{r['domain']:<24}] {r['title'][:52]:<52} {len(body):>6} chars")
