"""Search client abstraction for ResearcherAgent."""

import logging

import httpx

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"


class SearchClientError(Exception):
    """Raised when the search provider fails."""


class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily when `TAVILY_API_KEY` is configured. Otherwise falls back to a deterministic
    offline mock so the workflow stays runnable without external network access or keys.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        if self._settings.tavily_api_key:
            try:
                return self._tavily_search(query, max_results)
            except SearchClientError:
                logger.warning("Tavily search failed, falling back to mock search", exc_info=True)

        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            response = httpx.post(
                _TAVILY_ENDPOINT,
                json={
                    "api_key": self._settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=self._settings.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchClientError(f"Tavily request failed: {exc}") from exc

        payload = response.json()
        results = payload.get("results", [])[:max_results]
        return [
            SourceDocument(
                title=item.get("title") or query,
                url=item.get("url"),
                snippet=item.get("content", "")[:800],
                metadata={"score": item.get("score"), "provider": "tavily"},
            )
            for item in results
        ]

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Deterministic offline fallback used when no API key is available."""

        topics = [
            "overview and definitions",
            "key techniques and recent advances",
            "production trade-offs and limitations",
            "benchmark results and comparisons",
            "open problems and future directions",
        ]
        results: list[SourceDocument] = []
        for idx, topic in enumerate(topics[:max_results], start=1):
            results.append(
                SourceDocument(
                    title=f"{query.strip().rstrip('?.')} — {topic}",
                    url=None,
                    snippet=(
                        f"Mock source #{idx} covering {topic} related to '{query}'. "
                        "Replace with a real search provider (Tavily/Bing/SerpAPI) for "
                        "production use; TAVILY_API_KEY was not configured."
                    ),
                    metadata={"provider": "mock", "rank": idx},
                )
            )
        return results
