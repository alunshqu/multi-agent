import asyncio
import sys
from anthropic import AsyncAnthropic
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from agents.orchestrator import Orchestrator
from core.context import SharedContext
import config

console = Console()


async def main():
    if not config.ANTHROPIC_API_KEY:
        console.print("[red]错误：请在 .env 文件中设置 ANTHROPIC_API_KEY[/red]")
        sys.exit(1)

    client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    context = SharedContext()
    orchestrator = Orchestrator(client, context)

    console.print(Panel("[bold cyan]Multi-Agent Framework[/bold cyan]\n输入任务，输入 'exit' 退出", expand=False))

    while True:
        try:
            user_input = console.input("\n[bold green]任务>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]退出[/yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        with console.status("[cyan]处理中...[/cyan]"):
            result = await orchestrator.run(user_input)

        console.print(Panel(Markdown(result), title="[bold]结果[/bold]", border_style="blue"))


if __name__ == "__main__":
    asyncio.run(main())
