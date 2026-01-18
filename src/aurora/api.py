"""Aurora API"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from aurora.client import AuroraClient, RateLimitError, SearchResult
from aurora.schemas import (
    ChatChoice,
    ChatMessage,
    ChatQuery,
    ChatResponse,
    ChatUsage,
    SearchQuery,
    SearchResponse,
    SearchResultItem,
)


def _search_result_to_item(result: SearchResult) -> SearchResultItem:
    """Convert SearchResult to SearchResultItem."""
    return SearchResultItem(
        title=result.title,
        snippet=result.snippet,
        url=result.url,
        published_date=result.published_date,
    )


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    client = AuroraClient()

    app = FastAPI(
        title="Aurora API",
        description="Perplexity AI API wrapper",
        version="0.1.0",
    )

    # CORS middleware for cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health")
    async def _health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    # Chat completion endpoint
    @app.post("/api/chat")
    async def _chat(query_params: ChatQuery) -> ChatResponse:
        """Chat completion using Perplexity API.

        Args:
            query_params: Chat query parameters

        Returns:
            Chat completion response with model response and usage stats

        Raises:
            HTTPException: 429 if rate limited (after retries), or other API errors
        """
        try:
            result = await client.chat(
                model=query_params.model,
                messages=[{"role": m.role, "content": m.content} for m in query_params.messages],
                max_tokens=query_params.max_tokens,
                temperature=query_params.temperature,
                top_p=query_params.top_p,
                top_k=query_params.top_k,
                frequency_penalty=query_params.frequency_penalty,
                presence_penalty=query_params.presence_penalty,
                search_mode=query_params.search_mode,
                disable_search=query_params.disable_search,
                search_recency_filter=query_params.search_recency_filter,
                reasoning_effort=query_params.reasoning_effort,
            )

            search_items = None
            if result.search_results:
                search_items = [_search_result_to_item(r) for r in result.search_results]

            return ChatResponse(
                id=result.id,
                model=result.model,
                created=result.created,
                choices=[
                    ChatChoice(
                        message=ChatMessage(role="assistant", content=result.content),
                        finish_reason=result.finish_reason,
                    )
                ],
                usage=ChatUsage(
                    prompt_tokens=result.usage["prompt_tokens"],
                    completion_tokens=result.usage["completion_tokens"],
                    total_tokens=result.usage["total_tokens"],
                ),
                search_results=search_items,
            )
        except RateLimitError as e:
            raise HTTPException(
                status_code=429, detail="Rate limited. Too many requests. Try again later."
            ) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat API error: {str(e)}") from e

    # Search endpoint
    @app.post("/api/search")
    async def _search(query_params: SearchQuery) -> SearchResponse:
        """Search using Perplexity API with advanced filtering.

        Args:
            query_params: Search query parameters

        Returns:
            Search results

        Raises:
            HTTPException: 429 if rate limited (after retries), or other API errors
        """
        try:
            results = await client.search(
                query=query_params.query,
                max_results=query_params.max_results,
                max_tokens_per_page=query_params.max_tokens_per_page,
                search_recency_filter=query_params.search_recency_filter,
                search_domain_filter=query_params.search_domain_filter,
                search_language_filter=query_params.search_language_filter,
                country=query_params.country,
            )

            items = [_search_result_to_item(r) for r in results]
            return SearchResponse(results=items, query=query_params.query)
        except RateLimitError as e:
            raise HTTPException(
                status_code=429, detail="Rate limited. Too many requests. Try again later."
            ) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}") from e

    return app
