import json
from openai import AsyncOpenAI
from agents.base import BaseAgent
from core.task import SubTask, TaskResult
from core.context import SharedContext
from tools.executor import execute_bash, execute_python
from tools.forge import ToolForge, _TOOLS_DIR
import config

_STATIC_TOOLS = [
    {
        "name": "run_bash",
        "description": "执行 bash 命令",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "run_python",
        "description": "执行 Python 代码",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "write_file",
        "description": "将内容写入文件",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
]


class CodeAgent(BaseAgent):
    def __init__(self, client: AsyncOpenAI, context: SharedContext, forge: ToolForge):
        super().__init__(client, context)
        self._forge = forge
        self._dynamic_tools: list[dict] = forge.load_all()

    @property
    def name(self) -> str:
        return "CodeAgent"

    @property
    def system_prompt(self) -> str:
        dynamic_names = [t["name"] for t in self._dynamic_tools]
        extra = ""
        if dynamic_names:
            extra = f"\n可用的扩展工具：{', '.join(dynamic_names)}"
        return f"""你是一个专业的代码生成和执行 agent。
你可以生成代码、执行脚本、操作文件系统。
优先使用 Python 完成任务，必要时使用 bash。
执行前先思考安全性，避免破坏性操作。
返回执行结果和关键输出。{extra}"""

    @property
    def tools(self) -> list:
        return _STATIC_TOOLS + self._dynamic_tools

    def add_tool(self, tool_def: dict):
        """运行时注册新工具（去重）。"""
        existing = {t["name"] for t in self._dynamic_tools}
        if tool_def["name"] not in existing:
            self._dynamic_tools.append(tool_def)

    async def execute(self, task: SubTask) -> TaskResult:
        messages = [{"role": "user", "content": task.description}]
        try:
            output = await self._agentic_loop(messages, self.tools)
            self.context.add_event(self.name, task.description, output)
            return TaskResult(task.id, self.name, True, output)
        except Exception as e:
            return TaskResult(task.id, self.name, False, "", str(e))

    async def _handle_tool(self, name: str, input: dict) -> list:
        if name == "run_bash":
            return [{"type": "text", "text": execute_bash(input["command"])}]

        if name == "run_python":
            return [{"type": "text", "text": execute_python(input["code"])}]

        if name == "write_file":
            try:
                with open(input["path"], "w", encoding="utf-8") as f:
                    f.write(input["content"])
                return [{"type": "text", "text": f"Written to {input['path']}"}]
            except Exception as e:
                return [{"type": "text", "text": f"Error: {e}"}]

        # 动态工具：从文件加载并在子进程中执行
        dynamic_names = {t["name"] for t in self._dynamic_tools}
        if name in dynamic_names:
            return [{"type": "text", "text": self._run_dynamic(name, input)}]

        return [{"type": "text", "text": f"Unknown tool: {name}"}]

    def _run_dynamic(self, name: str, params: dict) -> str:
        code = self._forge.get_code(name)
        if not code:
            return f"Error: tool file for '{name}' not found"
        # 在子进程中执行：加载工具代码，调用 run(params)，打印结果
        runner = (
            f"{code}\n\n"
            f"import json, sys\n"
            f"_params = json.loads(sys.argv[1])\n"
            f"print(run(_params))"
        )
        import tempfile, subprocess, sys, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(runner)
            tmp = f.name
        try:
            result = subprocess.run(
                [sys.executable, tmp, json.dumps(params)],
                capture_output=True, text=True, timeout=60,
                env={**os.environ},
            )
            out = result.stdout.strip()
            if result.stderr:
                out += f"\nSTDERR: {result.stderr.strip()}"
            return out or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: tool timed out"
        except Exception as e:
            return f"Error: {e}"
        finally:
            os.unlink(tmp)
