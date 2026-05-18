from openai import AsyncOpenAI
from agents.base import BaseAgent
from core.task import SubTask, TaskResult
from core.context import SharedContext
from tools.executor import execute_python, execute_bash
import config

_TOOLS = [
    {
        "name": "run_python",
        "description": "执行 Python 数据分析代码（pandas, matplotlib, numpy 等）",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "run_bash",
        "description": "执行 bash 命令，用于文件操作或调用命令行工具",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


class DataAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "DataAgent"

    @property
    def system_prompt(self) -> str:
        return """你是一个专业的数据分析 agent。
擅长使用 pandas、numpy、matplotlib、seaborn 进行数据处理和可视化。
分析数据时先了解数据结构，再进行清洗、分析、可视化。
输出清晰的分析结论和关键指标。"""

    @property
    def tools(self) -> list:
        return _TOOLS

    async def execute(self, task: SubTask) -> TaskResult:
        messages = [{"role": "user", "content": task.description}]
        try:
            output = await self._agentic_loop(messages, self.tools)
            self.context.add_event(self.name, task.description, output)
            return TaskResult(task.id, self.name, True, output)
        except Exception as e:
            return TaskResult(task.id, self.name, False, "", str(e))

    async def _handle_tool(self, name: str, input: dict) -> list:
        if name == "run_python":
            out = execute_python(input["code"])
            return [{"type": "text", "text": out}]
        elif name == "run_bash":
            out = execute_bash(input["command"])
            return [{"type": "text", "text": out}]
        return [{"type": "text", "text": f"Unknown tool: {name}"}]
