"""Entry point for Aurora application."""

import argparse
import asyncio

import uvicorn

from aurora.ui import create_ui


async def run_api() -> None:
    """Run FastAPI server."""
    config = uvicorn.Config(
        "aurora.api:create_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        factory=True,
    )
    server = uvicorn.Server(config)
    await server.serve()


def run_ui() -> None:
    """Run Gradio interface."""
    demo = create_ui()
    demo.launch(share=False, show_error=True)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Aurora: Perplexity API wrapper")
    parser.add_argument(
        "--mode",
        choices=["api", "ui", "both"],
        default="both",
        help="Run mode: api (FastAPI), ui (Gradio), or both",
    )
    args = parser.parse_args()

    match args.mode:
        case "api":
            asyncio.run(run_api())
        case "ui":
            run_ui()
        case "both":
            print("Running Gradio UI (FastAPI at http://localhost:8000)")
            run_ui()


if __name__ == "__main__":
    main()
