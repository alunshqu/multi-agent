import asyncio
import sys
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from agents.orchestrator import Orchestrator
from core.context import SharedContext
from memory.store import MemoryStore
import config

console = Console()


async def main():
    if not config.OPENAI_API_KEY:
        console.print("[red]错误：请设置 OPENAI_API_KEY 环境变量[/red]")
        sys.exit(1)

    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
    store = MemoryStore()
    context = SharedContext()
    orchestrator = Orchestrator(client, context, store)

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
