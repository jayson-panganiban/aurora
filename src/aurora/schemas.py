"""Pydantic request/response schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class SearchQuery(BaseModel):
    """Search request schema."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=50)
    max_tokens_per_page: int = Field(default=2048, ge=256, le=4096)
    search_recency_filter: Literal["day", "week", "month", "year"] | None = None
    search_domain_filter: list[str] | None = Field(default=None, max_length=20)
    search_language_filter: list[str] | None = Field(default=None, max_length=10)
    country: str | None = None


class SearchResultItem(BaseModel):
    """Individual search result."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str
    snippet: str
    url: str | None = None
    published_date: str | None = None


class SearchResponse(BaseModel):
    """Search response schema."""

    model_config = ConfigDict(str_strip_whitespace=True)

    results: list[SearchResultItem]
    query: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Total number of results."""
        return len(self.results)


class ChatMessage(BaseModel):
    """Chat message."""

    model_config = ConfigDict(str_strip_whitespace=True)

    role: Literal["user", "assistant", "system"]
    content: str


class ChatQuery(BaseModel):
    """Chat completion request schema."""

    model_config = ConfigDict(str_strip_whitespace=True)

    model: Literal["sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-deep-research"] = "sonar"
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    top_k: int = Field(default=0, ge=0)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    stream: bool = False
    search_mode: Literal["web", "academic", "sec"] = "web"
    disable_search: bool = False
    search_recency_filter: Literal["day", "week", "month", "year"] | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None


class ChatChoice(BaseModel):
    """Chat completion choice."""

    model_config = ConfigDict(str_strip_whitespace=True)

    message: ChatMessage
    finish_reason: str


class ChatUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """Chat completion response schema."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str
    model: str
    created: int
    choices: list[ChatChoice]
    usage: ChatUsage
    search_results: list[SearchResultItem] | None = None
