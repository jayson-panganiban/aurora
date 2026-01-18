"""Tests for Perplexity client."""

from unittest.mock import MagicMock, patch

import pytest

from aurora.client import AuroraClient, RateLimitError, _is_rate_limit_error


# Tests for rate limit detection helper
def test_is_rate_limit_error_with_429_code() -> None:
    """Test rate limit detection with 429 status code."""
    error = Exception("429 Too Many Requests")
    assert _is_rate_limit_error(error)


def test_is_rate_limit_error_with_rate_limit_text() -> None:
    """Test rate limit detection with rate limit text."""
    error = Exception("rate limit exceeded")
    assert _is_rate_limit_error(error)


def test_is_rate_limit_error_negative() -> None:
    """Test rate limit detection with non-rate-limit error."""
    error = Exception("Connection timeout")
    assert not _is_rate_limit_error(error)


# Tests for search result value extraction helper
def test_get_search_result_value_from_dict() -> None:
    """Test extracting value from dictionary."""
    from aurora.client import _get_search_result_value

    data = {"title": "Test Title", "snippet": "Test snippet"}
    assert _get_search_result_value(data, "title") == "Test Title"
    assert _get_search_result_value(data, "missing", "default") == "default"
    assert _get_search_result_value(data, "missing") is None


def test_get_search_result_value_from_object() -> None:
    """Test extracting value from object with attributes."""
    from aurora.client import _get_search_result_value

    class Result:
        title = "Test Title"
        snippet = "Test snippet"

    obj = Result()
    assert _get_search_result_value(obj, "title") == "Test Title"
    assert _get_search_result_value(obj, "missing", "default") == "default"


# Tests for search method
@pytest.mark.asyncio
async def test_search_with_filters() -> None:
    """Test search with advanced filters."""
    with patch("aurora.client.Perplexity") as mock_perplexity:
        mock_response = MagicMock()
        mock_response.results = [
            MagicMock(
                title="Python Tutorial",
                snippet="Learn Python programming",
                url="https://example.com",
                published_date="2024-01-01",
            ),
        ]
        mock_perplexity.return_value.search.create.return_value = mock_response

        client = AuroraClient()
        results = await client.search(
            query="python",
            max_results=5,
            search_domain_filter=["github.com"],
            search_recency_filter="week",
        )

        assert len(results) == 1
        assert results[0].title == "Python Tutorial"
        assert results[0].snippet == "Learn Python programming"
        assert results[0].url == "https://example.com"
        assert results[0].published_date == "2024-01-01"


@pytest.mark.asyncio
async def test_search_with_rate_limit_retry() -> None:
    """Test search with rate limit retry logic."""
    with patch("aurora.client.Perplexity") as mock_perplexity:
        mock_response = MagicMock()
        mock_response.results = [
            MagicMock(
                title="Result",
                snippet="Snippet",
                url="http://example.com",
                published_date="2024-01-01",
            ),
        ]
        # First call fails with 429, second succeeds
        mock_perplexity.return_value.search.create.side_effect = [
            Exception("429 Rate limited"),
            mock_response,
        ]

        client = AuroraClient()
        results = await client.search(query="test")

        assert len(results) == 1
        assert results[0].title == "Result"
        assert results[0].snippet == "Snippet"
        assert results[0].url == "http://example.com"
        assert results[0].published_date == "2024-01-01"
        assert mock_perplexity.return_value.search.create.call_count == 2


@pytest.mark.asyncio
async def test_search_rate_limit_max_retries_exceeded() -> None:
    """Test search fails after max retries exceeded."""
    with patch("aurora.client.Perplexity") as mock_perplexity:
        # All calls fail with rate limit
        mock_perplexity.return_value.search.create.side_effect = Exception("429 Rate limited")

        client = AuroraClient()
        with pytest.raises(RateLimitError):
            await client.search(query="test")

        # Should have tried MAX_RETRIES times (3)
        assert mock_perplexity.return_value.search.create.call_count == 3


# Tests for chat method
@pytest.mark.asyncio
async def test_chat_completion() -> None:
    """Test chat completion endpoint."""
    with patch("aurora.client.Perplexity") as mock_perplexity:
        mock_response = MagicMock()
        mock_response.id = "test-123"
        mock_response.model = "sonar"
        mock_response.created = 1234567890
        mock_response.choices = [
            MagicMock(
                message=MagicMock(role="assistant", content="Hello, world!"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        mock_response.search_results = []
        mock_perplexity.return_value.chat.completions.create.return_value = mock_response

        client = AuroraClient()
        result = await client.chat(
            model="sonar",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.id == "test-123"
        assert result.model == "sonar"
        assert result.created == 1234567890
        assert result.content == "Hello, world!"
        assert result.finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 20
        assert result.usage["total_tokens"] == 30
        # Empty list is treated as falsy, so search_results becomes None
        assert result.search_results is None


@pytest.mark.asyncio
async def test_chat_with_search_results_objects() -> None:
    """Test chat completion with search results as objects."""
    with patch("aurora.client.Perplexity") as mock_perplexity:
        mock_response = MagicMock()
        mock_response.id = "test-456"
        mock_response.model = "sonar"
        mock_response.created = 1234567890
        mock_response.choices = [
            MagicMock(
                message=MagicMock(role="assistant", content="Here are results"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=15,
            completion_tokens=25,
            total_tokens=40,
        )
        # Create objects with attributes instead of dict
        mock_response.search_results = [
            MagicMock(
                title="Python Docs",
                snippet="Official Python documentation",
                url="https://docs.python.org",
                published_date="2024-01-01",
            ),
        ]
        mock_perplexity.return_value.chat.completions.create.return_value = mock_response

        client = AuroraClient()
        result = await client.chat(
            model="sonar",
            messages=[{"role": "user", "content": "Tell me about Python"}],
        )

        assert result.id == "test-456"
        assert result.content == "Here are results"
        assert result.search_results is not None
        assert len(result.search_results) == 1
        assert result.search_results[0].title == "Python Docs"
        assert result.search_results[0].url == "https://docs.python.org"


@pytest.mark.asyncio
async def test_chat_with_all_parameters() -> None:
    """Test chat with all optional parameters."""
    with patch("aurora.client.Perplexity") as mock_perplexity:
        mock_response = MagicMock()
        mock_response.id = "test-full"
        mock_response.model = "sonar-pro"
        mock_response.created = 1234567890
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Response"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        )
        mock_response.search_results = None
        mock_perplexity.return_value.chat.completions.create.return_value = mock_response

        client = AuroraClient()
        result = await client.chat(
            model="sonar-pro",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1000,
            temperature=0.5,
            top_p=0.8,
            top_k=10,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            search_mode="academic",
            disable_search=True,
            search_recency_filter="month",
            reasoning_effort="high",
        )

        assert result.id == "test-full"
        # Verify all parameters were passed correctly
        call_kwargs = mock_perplexity.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "sonar-pro"
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.8
        assert call_kwargs["top_k"] == 10
        assert call_kwargs["frequency_penalty"] == 0.5
        assert call_kwargs["presence_penalty"] == 0.5
        assert call_kwargs["search_mode"] == "academic"
        assert call_kwargs["disable_search"] is True
        assert call_kwargs["search_recency_filter"] == "month"
        assert call_kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_chat_with_rate_limit_then_success() -> None:
    """Test chat with rate limit retry and eventual success."""
    with patch("aurora.client.Perplexity") as mock_perplexity:
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.model = "sonar"
        mock_response.created = 1234567890
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Success"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        )
        mock_response.search_results = None

        # First call fails with rate limit, second succeeds
        mock_perplexity.return_value.chat.completions.create.side_effect = [
            Exception("429 Rate limited"),
            mock_response,
        ]

        client = AuroraClient()
        result = await client.chat(
            model="sonar",
            messages=[{"role": "user", "content": "Test"}],
        )

        assert result.content == "Success"
        assert mock_perplexity.return_value.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_chat_response_validation_empty_choices() -> None:
    """Test chat with empty choices raises ValueError."""
    with patch("aurora.client.Perplexity") as mock_perplexity:
        mock_response = MagicMock()
        mock_response.choices = []  # Empty choices

        mock_perplexity.return_value.chat.completions.create.return_value = mock_response

        client = AuroraClient()
        with pytest.raises(ValueError, match="Response has no choices"):
            await client.chat(
                model="sonar",
                messages=[{"role": "user", "content": "Test"}],
            )


@pytest.mark.asyncio
async def test_chat_rate_limit_max_retries_exceeded() -> None:
    """Test chat fails after max retries exceeded."""
    with patch("aurora.client.Perplexity") as mock_perplexity:
        # All calls fail with rate limit
        mock_perplexity.return_value.chat.completions.create.side_effect = Exception(
            "429 Rate limited"
        )

        client = AuroraClient()
        with pytest.raises(RateLimitError):
            await client.chat(
                model="sonar",
                messages=[{"role": "user", "content": "Test"}],
            )

        # Should have tried MAX_RETRIES times (3)
        assert mock_perplexity.return_value.chat.completions.create.call_count == 3
