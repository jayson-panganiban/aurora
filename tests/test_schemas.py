"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

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


# SearchQuery tests
def test_search_query_with_filters() -> None:
    """Test search query with all filters."""
    query = SearchQuery(
        query="python",
        max_results=20,
        max_tokens_per_page=4096,
        search_recency_filter="week",
        search_domain_filter=["github.com", "stackoverflow.com"],
        search_language_filter=["en", "tl"],
        country="AU",
    )
    assert query.search_recency_filter == "week"
    assert query.search_domain_filter == ["github.com", "stackoverflow.com"]
    assert query.search_language_filter == ["en", "tl"]
    assert query.country == "AU"


def test_search_query_invalid_max_results() -> None:
    """Test search query with invalid max_results."""
    with pytest.raises(ValidationError):
        SearchQuery(query="test", max_results=0)

    with pytest.raises(ValidationError):
        SearchQuery(query="test", max_results=51)


def test_search_query_empty_query() -> None:
    """Test search query with empty query string."""
    with pytest.raises(ValidationError):
        SearchQuery(query="")


# SearchResult and SearchResponse tests
def test_search_result_item_valid() -> None:
    """Test valid search result item."""
    item = SearchResultItem(
        title="Test",
        snippet="Test snippet",
        url="https://example.com",
        published_date="2024-01-01",
    )
    assert item.title == "Test"
    assert item.snippet == "Test snippet"
    assert item.url == "https://example.com"
    assert item.published_date == "2024-01-01"


def test_search_result_item_optional_fields() -> None:
    """Test search result item with optional fields as None."""
    item = SearchResultItem(title="Test", snippet="Snippet")
    assert item.url is None
    assert item.published_date is None


def test_search_response_with_count() -> None:
    """Test search response computed count field."""
    response = SearchResponse(
        query="python",
        results=[
            SearchResultItem(title="1", snippet="s1"),
            SearchResultItem(title="2", snippet="s2"),
            SearchResultItem(title="3", snippet="s3"),
        ],
    )
    assert response.count == 3
    assert len(response.results) == 3
    assert response.query == "python"


def test_search_response_empty_results() -> None:
    """Test search response with no results."""
    response = SearchResponse(query="test", results=[])
    assert response.count == 0
    assert len(response.results) == 0


# ChatMessage tests
def test_chat_message_valid_roles() -> None:
    """Test chat message with valid roles."""
    user_msg = ChatMessage(role="user", content="Hello")
    assert user_msg.role == "user"

    assistant_msg = ChatMessage(role="assistant", content="Hi")
    assert assistant_msg.role == "assistant"

    system_msg = ChatMessage(role="system", content="System prompt")
    assert system_msg.role == "system"


def test_chat_message_invalid_role() -> None:
    """Test chat message with invalid role."""
    with pytest.raises(ValidationError):
        ChatMessage(role="invalid", content="Test")  # type: ignore[arg-type]


# ChatQuery tests
def test_chat_query_all_parameters() -> None:
    """Test chat query with all parameters."""
    query = ChatQuery(
        model="sonar-pro",
        messages=[ChatMessage(role="user", content="Test")],
        max_tokens=1000,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stream=False,
        search_mode="academic",
        disable_search=True,
        search_recency_filter="month",
        reasoning_effort="high",
    )
    assert query.model == "sonar-pro"
    assert query.max_tokens == 1000
    assert query.temperature == 0.7
    assert query.search_mode == "academic"
    assert query.reasoning_effort == "high"


def test_chat_query_invalid_model() -> None:
    """Test chat query with invalid model."""
    with pytest.raises(ValidationError):
        ChatQuery(
            model="invalid-model",  # type: ignore[arg-type]
            messages=[ChatMessage(role="user", content="Test")],
        )


def test_chat_query_invalid_temperature() -> None:
    """Test chat query with out-of-range temperature."""
    with pytest.raises(ValidationError):
        ChatQuery(
            model="sonar",
            messages=[ChatMessage(role="user", content="Test")],
            temperature=3.0,
        )

    with pytest.raises(ValidationError):
        ChatQuery(
            model="sonar",
            messages=[ChatMessage(role="user", content="Test")],
            temperature=-1.0,
        )


def test_chat_query_invalid_search_mode() -> None:
    """Test chat query with invalid search mode."""
    with pytest.raises(ValidationError):
        ChatQuery(
            model="sonar",
            messages=[ChatMessage(role="user", content="Test")],
            search_mode="invalid",  # type: ignore[arg-type]
        )


# ChatResponse tests
def test_chat_response_complete() -> None:
    """Test complete chat response."""
    response = ChatResponse(
        id="test-123",
        model="sonar",
        created=1234567890,
        choices=[
            ChatChoice(
                message=ChatMessage(role="assistant", content="Response"),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        search_results=None,
    )
    assert response.id == "test-123"
    assert response.model == "sonar"
    assert len(response.choices) == 1
    assert response.choices[0].message.content == "Response"
    assert response.usage.total_tokens == 30
    assert response.search_results is None


def test_chat_response_with_search_results() -> None:
    """Test chat response with search results."""
    response = ChatResponse(
        id="test-456",
        model="sonar",
        created=1234567890,
        choices=[
            ChatChoice(
                message=ChatMessage(role="assistant", content="Answer"),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40),
        search_results=[
            SearchResultItem(
                title="Result 1",
                snippet="Snippet 1",
                url="https://example.com",
            )
        ],
    )
    assert response.search_results is not None
    assert len(response.search_results) == 1
    assert response.search_results[0].title == "Result 1"


def test_chat_usage_validation() -> None:
    """Test chat usage token counts."""
    usage = ChatUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30
