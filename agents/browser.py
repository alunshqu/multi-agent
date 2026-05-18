from openai import AsyncOpenAI
from agents.base import BaseAgent
from core.task import SubTask, TaskResult
from core.context import SharedContext
from tools.computer import execute_computer_action, get_screen_size
from tools.executor import execute_bash
import config


_COMPUTER_TOOLS = [
    {
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 0,
        "display_height_px": 0,
    },
    {
        "type": "bash_20241022",
        "name": "bash",
    },
]


class BrowserAgent(BaseAgent):
    def __init__(self, client: AsyncOpenAI, context: SharedContext):
        super().__init__(client, context)
        w, h = get_screen_size()
        _COMPUTER_TOOLS[0]["display_width_px"] = w
        _COMPUTER_TOOLS[0]["display_height_px"] = h

    @property
    def name(self) -> str:
        return "BrowserAgent"

    @property
    def system_prompt(self) -> str:
        w, h = get_screen_size()
        return f"""你是一个专业的浏览器和 GUI 操作 agent。
你可以控制鼠标、键盘，操作浏览器和桌面应用。
屏幕分辨率: {w}x{h}。
操作系统: macOS。
执行任务时先截图了解当前状态，再采取行动。
完成后返回简洁的结果摘要。"""

    @property
    def tools(self) -> list:
        return _COMPUTER_TOOLS

    async def execute(self, task: SubTask) -> TaskResult:
        messages = [{"role": "user", "content": task.description}]
        try:
            output = await self._agentic_loop(
                messages,
                self.tools,
                extra_kwargs={"betas": ["computer-use-2024-10-22"]},
            )
            self.context.add_event(self.name, task.description, output)
            return TaskResult(task.id, self.name, True, output)
        except Exception as e:
            return TaskResult(task.id, self.name, False, "", str(e))

    async def _handle_tool(self, name: str, input: dict) -> list:
        if name == "computer":
            action = input.get("action", "screenshot")
            result = execute_computer_action(action, **{k: v for k, v in input.items() if k != "action"})
            return [result]
        elif name == "bash":
            output = execute_bash(input.get("command", ""))
            return [{"type": "text", "text": output}]
        return [{"type": "text", "text": f"Unknown tool: {name}"}]
