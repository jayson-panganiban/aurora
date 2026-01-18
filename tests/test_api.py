"""Tests for the application."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from aurora.api import create_app
from aurora.client import ChatResult, SearchResult


def test_health() -> None:
    """Test health check endpoint."""
    with patch("aurora.api.AuroraClient"):
        app = create_app()
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_chat_endpoint() -> None:
    """Test chat endpoint with successful response."""
    mock_result = ChatResult(
        id="test-123",
        model="sonar",
        created=1234567890,
        content="Test response",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        search_results=None,
    )

    with patch("aurora.api.AuroraClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=mock_result)
        mock_client_class.return_value = mock_client

        app = create_app()
        client = TestClient(app)

        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-123"
        assert data["model"] == "sonar"
        assert data["created"] == 1234567890
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Test response"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 20
        assert data["usage"]["total_tokens"] == 30
        assert data["search_results"] is None


def test_search_endpoint() -> None:
    """Test search endpoint with successful response."""
    mock_result = [
        SearchResult(
            title="Result 1",
            snippet="Snippet 1",
            url="http://example.com",
            published_date="2024-01-01",
        ),
    ]

    with patch("aurora.api.AuroraClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=mock_result)
        mock_client_class.return_value = mock_client

        app = create_app()
        client = TestClient(app)

        payload = {"query": "python", "max_results": 3}
        response = client.post("/api/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "query" in data
        assert "count" in data
        assert data["query"] == "python"
        assert data["count"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "Result 1"
        assert data["results"][0]["snippet"] == "Snippet 1"
        assert data["results"][0]["url"] == "http://example.com"
        assert data["results"][0]["published_date"] == "2024-01-01"
