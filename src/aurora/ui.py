"""Gradio web interface"""

import gradio as gr

from aurora.client import AuroraClient


async def chat_handler(
    message: str,
    model: str,
    temperature: float,
    search_mode: str,
) -> tuple[str, str, str]:
    """Handle chat requests."""
    if not message.strip():
        raise gr.Error("Enter a message")

    client = AuroraClient()
    try:
        result = await client.chat(
            model=model,
            messages=[{"role": "user", "content": message}],
            temperature=temperature,
            search_mode=search_mode,
        )

        response = result.content
        usage = (
            f"Tokens: {result.usage['total_tokens']} "
            f"(prompt: {result.usage['prompt_tokens']}, "
            f"completion: {result.usage['completion_tokens']})"
        )
        sources = ""
        if result.search_results:
            sources = "\n".join(f"- [{s.title}]({s.url})" for s in result.search_results if s.url)

        return response, usage, sources

    except Exception as e:
        raise gr.Error(f"Chat failed: {str(e)}") from e


async def search_handler(
    query: str,
    max_results: int,
    recency_filter: str,
) -> str:
    """Handle search requests."""
    if not query.strip():
        raise gr.Error("Enter a search query")

    client = AuroraClient()
    try:
        results = await client.search(
            query=query,
            max_results=max_results,
            search_recency_filter=recency_filter if recency_filter != "none" else None,
        )

        if not results:
            return "No results found"

        lines = []
        for i, result in enumerate(results, 1):
            lines.append(f"**{i}. {result.title}**")
            lines.append(result.snippet)
            if result.url:
                lines.append(f"[{result.url}]({result.url})")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        raise gr.Error(f"Search failed: {str(e)}") from e


def create_ui() -> gr.Blocks:
    """Create minimal Gradio interface with chat and search tabs."""
    demo = gr.Blocks(title="Aurora")

    with demo:
        gr.Markdown("# Aurora Chatbot\nPerplexity AI Wrapper")

        with gr.Tabs():
            # Chat tab
            with gr.TabItem("Chat"):
                message = gr.Textbox(label="Message", lines=3)

                with gr.Row():
                    model = gr.Dropdown(
                        label="Model",
                        choices=[
                            "sonar",
                            "sonar-pro",
                            "sonar-reasoning-pro",
                            "sonar-deep-research",
                        ],
                        value="sonar",
                    )
                    temperature = gr.Slider(
                        label="Temperature", minimum=0, maximum=2, value=0.2, step=0.1
                    )
                    search_mode = gr.Dropdown(
                        label="Search Mode",
                        choices=["web", "academic", "sec"],
                        value="web",
                    )

                response = gr.Textbox(label="Response", interactive=False, lines=6)
                usage = gr.Textbox(label="Usage", interactive=False)
                sources = gr.Textbox(label="Sources", interactive=False, lines=3)

                gr.Button("Send").click(
                    fn=chat_handler,
                    inputs=[message, model, temperature, search_mode],
                    outputs=[response, usage, sources],
                    queue=True,
                )

            # Search tab
            with gr.TabItem("Search"):
                query = gr.Textbox(label="Query", lines=2)

                with gr.Row():
                    max_results = gr.Slider(label="Results", minimum=1, maximum=20, value=5, step=1)
                    recency = gr.Dropdown(
                        label="Recency",
                        choices=["none", "day", "week", "month", "year"],
                        value="none",
                    )

                results_output = gr.Textbox(label="Results", interactive=False, lines=10)

                gr.Button("Search").click(
                    fn=search_handler,
                    inputs=[query, max_results, recency],
                    outputs=results_output,
                    queue=True,
                )

    demo.queue(max_size=100)
    return demo
