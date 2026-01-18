# Aurora

A chatbot application built with a FastAPI backend and a lightweight Gradio web interface that wraps the Perplexity AI API.

## Quick Start

```bash
# Setup
uv sync
export PERPLEXITY_API_KEY=your_api_key_here

# Run
uv run main.py
```

- **Gradio UI**: http://localhost:7860
- **FastAPI API**: http://localhost:8000

## Features

- **Chat Completions**: All 4 Sonar models (sonar, sonar-pro, sonar-reasoning-pro, sonar-deep-research)
- **Advanced Search**: Web/academic/sec modes, domain/language/recency filters
- **Rate Limit Handling**: Automatic exponential backoff with 3 retries
- **Async-First**: FastAPI + asyncio for non-blocking operations
- **Type Safe**: Strict mypy enforcement, Pydantic v2 validation
- **Minimal UI**: Lightweight Gradio interface

## API

```bash
# Chat Completions
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "sonar", "messages": [{"role": "user", "content": "Hello"}]}'

# Search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "python", "max_results": 5}'

# Health Check
curl http://localhost:8000/health
```

## Development

```bash
# Run
uv run main.py                            # Both UI + API
uv run main.py --mode api                 # FastAPI only
uv run main.py --mode ui                  # Gradio only

# Quality
uv run ruff check aurora/ tests/           # Lint
uv run ruff format aurora/ tests/          # Format
uv run mypy aurora/ tests/ --strict        # Type check
uv run pytest tests/ -v                    # Test
uv run pytest tests/ --cov=aurora          # Coverage
```

## Requirements

- Python 3.13+
- Perplexity API key (https://perplexity.ai/account/api)

## License

MIT
