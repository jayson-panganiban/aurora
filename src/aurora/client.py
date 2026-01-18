"""Perplexity API client wrapper"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from perplexity import Perplexity  # type: ignore[import-untyped]

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1


@dataclass
class SearchResult:
    """Search result from Perplexity API."""

    title: str
    snippet: str
    url: str | None = None
    published_date: str | None = None


@dataclass
class ChatResult:
    id: str
    model: str
    created: int
    content: str
    finish_reason: str
    usage: dict[str, int]
    search_results: list[SearchResult] | None = None


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    pass


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a rate limit error."""
    error_str = str(error).lower()
    return "429" in str(error) or "rate limit" in error_str


def _get_search_result_value(
    obj: dict | object, key: str, default: str | None = None
) -> str | None:
    """Extract value from dict or object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


async def _retry_with_backoff(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute function with exponential backoff retry on rate limit.

    Args:
        func: Async function to call
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result

    Raises:
        RateLimitError: If rate limited after max retries
        Other exceptions: Propagated as-is
    """
    for attempt in range(MAX_RETRIES):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                await asyncio.sleep(delay)
                continue
            if _is_rate_limit_error(e):
                raise RateLimitError(str(e)) from e
            raise


class AuroraClient:
    """Async wrapper around Perplexity SDK with retry logic."""

    def __init__(self) -> None:
        self._client = Perplexity()

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 0,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        search_mode: str = "web",
        disable_search: bool = False,
        search_recency_filter: str | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        """Chat completion using Perplexity API.

        Args:
            model: Model name (sonar, sonar-pro, sonar-reasoning-pro, sonar-deep-research)
            messages: List of chat messages with role and content
            max_tokens: Maximum tokens in response
            temperature: Randomness (0-2)
            top_p: Nucleus sampling (0-1)
            top_k: Keep top k tokens (0=disabled)
            frequency_penalty: Reduce repetition (-2 to 2)
            presence_penalty: Encourage new topics (-2 to 2)
            search_mode: Search type (web, academic, sec)
            disable_search: Use training data only
            search_recency_filter: Filter by recency (day, week, month, year)
            reasoning_effort: Effort level for deep research (low, medium, high)

        Returns:
            ChatResult with content, usage, and optional search results

        Raises:
            RateLimitError: If rate limited (will retry automatically)
            ValueError: If response format is invalid
        """

        async def _make_request() -> Any:  # type: ignore[no-untyped-def]
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "search_mode": search_mode,
                "disable_search": disable_search,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if search_recency_filter:
                kwargs["search_recency_filter"] = search_recency_filter
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

            return self._client.chat.completions.create(**kwargs)  # type: ignore[no-untyped-call]

        response = await _retry_with_backoff(_make_request)

        if not response.choices:
            raise ValueError("Response has no choices")

        search_results = None
        if hasattr(response, "search_results") and response.search_results:
            search_results = [
                SearchResult(
                    title=_get_search_result_value(r, "title", "") or "",
                    snippet=(_get_search_result_value(r, "snippet", "") or "")[:500],
                    url=_get_search_result_value(r, "url"),
                    published_date=_get_search_result_value(r, "published_date"),
                )
                for r in response.search_results
            ]

        return ChatResult(
            id=response.id,
            model=response.model,
            created=response.created,
            content=response.choices[0].message.content,
            finish_reason=response.choices[0].finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            search_results=search_results,
        )

    async def search(
        self,
        query: str,
        max_results: int = 5,
        max_tokens_per_page: int = 2048,
        search_recency_filter: str | None = None,
        search_domain_filter: list[str] | None = None,
        search_language_filter: list[str] | None = None,
        country: str | None = None,
    ) -> list[SearchResult]:
        """Search using Perplexity API with filters.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (1-50)
            max_tokens_per_page: Max tokens per search result
            search_recency_filter: Filter by recency (day, week, month, year)
            search_domain_filter: Domain allow/denylist (max 20)
            search_language_filter: Language filter (max 10 ISO 639-1)
            country: Geographic filter (ISO 3166-1 alpha-2)

        Returns:
            List of search results

        Raises:
            RateLimitError: If rate limited (will retry automatically)
        """

        async def _make_request() -> Any:  # type: ignore[no-untyped-def]
            kwargs = {
                "query": query,
                "max_results": max_results,
                "max_tokens_per_page": max_tokens_per_page,
            }
            if search_recency_filter:
                kwargs["search_recency_filter"] = search_recency_filter
            if search_domain_filter:
                kwargs["search_domain_filter"] = search_domain_filter
            if search_language_filter:
                kwargs["search_language_filter"] = search_language_filter
            if country:
                kwargs["country"] = country

            return self._client.search.create(**kwargs)  # type: ignore[no-untyped-call]

        response = await _retry_with_backoff(_make_request)

        return [
            SearchResult(
                title=_get_search_result_value(r, "title", "") or "",
                snippet=(_get_search_result_value(r, "snippet", "") or "")[:500],
                url=_get_search_result_value(r, "url"),
                published_date=_get_search_result_value(r, "published_date"),
            )
            for r in response.results
        ]
