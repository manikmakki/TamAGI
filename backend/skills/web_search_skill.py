"""
Web Search Skill — Multi-provider web search for TamAGI.

Providers (in priority order):
  1. DuckDuckGo (free, no API key, via duckduckgo-search library)
  2. Brave Search (requires API key, high quality)
  3. SearXNG (self-hosted, free, meta-search)

The active provider is selected via config.yaml. Falls back gracefully
if the preferred provider isn't available.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.skills.base import Skill, SkillResult

logger = logging.getLogger("tamagi.skills.web_search")


# ── Provider Implementations ─────────────────────────────────

async def _search_duckduckgo(query: str, max_results: int = 5, **kwargs) -> list[dict]:
    """Search using the duckduckgo-search Python library (free, no key)."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise RuntimeError(
                "ddgs not installed. Run: pip install ddgs"
            )

    import asyncio

    def _do_search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    # Run sync library in thread pool
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, _do_search)

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", r.get("link", "")),
            "snippet": r.get("body", r.get("snippet", "")),
            "source": "duckduckgo",
        }
        for r in results
    ]


async def _search_brave(
    query: str, max_results: int = 5, api_key: str = "", **kwargs
) -> list[dict]:
    """Search using Brave Search API (requires API key)."""
    if not api_key:
        raise RuntimeError("Brave Search requires an API key (web_search.brave_api_key)")

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(max_results, 20)},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for r in data.get("web", {}).get("results", [])[:max_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
            "source": "brave",
        })
    return results


async def _search_searxng(
    query: str, max_results: int = 5, base_url: str = "", **kwargs
) -> list[dict]:
    """Search using a self-hosted SearXNG instance."""
    if not base_url:
        raise RuntimeError("SearXNG requires a base_url (web_search.searxng_url)")

    base_url = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "general",
                "engines": "google,duckduckgo,bing",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for r in data.get("results", [])[:max_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "source": f"searxng ({r.get('engine', 'unknown')})",
        })
    return results


# ── Provider Registry ────────────────────────────────────────

PROVIDERS = {
    "duckduckgo": _search_duckduckgo,
    "brave": _search_brave,
    "searxng": _search_searxng,
}


# ── Skill Class ──────────────────────────────────────────────

class WebSearchSkill(Skill):
    """Search the web for information, news, documentation, or anything else."""

    name = "web_search"
    description = (
        "Search the web for current information, news, documentation, "
        "how-to guides, or any topic. Returns titles, URLs, and snippets. "
        "Use this when you need up-to-date information or facts you don't know."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "The search query. Be specific for better results.",
            "required": True,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (1-10). Default: 5.",
            "default": 5,
        },
    }

    def __init__(
        self,
        provider: str = "duckduckgo",
        brave_api_key: str = "",
        searxng_url: str = "",
    ):
        self.provider = provider.lower()
        self.brave_api_key = brave_api_key
        self.searxng_url = searxng_url

        # Validate provider
        if self.provider not in PROVIDERS:
            logger.warning(
                f"Unknown search provider '{self.provider}', falling back to duckduckgo"
            )
            self.provider = "duckduckgo"

    async def execute(self, **kwargs: Any) -> SkillResult:
        query = kwargs.get("query", "").strip()
        if not query:
            return SkillResult(success=False, error="No search query provided")

        max_results = min(max(int(kwargs.get("max_results", 5)), 1), 10)

        # Try preferred provider, fall back through chain
        providers_to_try = [self.provider]
        for p in ["duckduckgo", "brave", "searxng"]:
            if p not in providers_to_try:
                providers_to_try.append(p)

        last_error = ""
        for provider_name in providers_to_try:
            search_fn = PROVIDERS[provider_name]
            try:
                results = await search_fn(
                    query=query,
                    max_results=max_results,
                    api_key=self.brave_api_key,
                    base_url=self.searxng_url,
                )

                if not results:
                    last_error = f"{provider_name}: no results"
                    continue

                # Format output as readable text
                lines = [f"Web search results for: {query}\n"]
                for i, r in enumerate(results, 1):
                    lines.append(f"{i}. {r['title']}")
                    lines.append(f"   URL: {r['url']}")
                    lines.append(f"   {r['snippet']}")
                    lines.append("")

                logger.info(
                    f"web_search: '{query}' → {len(results)} results via {provider_name}"
                )
                return SkillResult(
                    success=True,
                    output="\n".join(lines),
                    data={"results": results, "provider": provider_name, "query": query},
                )

            except Exception as e:
                last_error = f"{provider_name}: {e}"
                logger.warning(f"Search provider {provider_name} failed: {e}")
                continue

        return SkillResult(
            success=False,
            error=f"All search providers failed. Last error: {last_error}",
        )
