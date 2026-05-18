import aiohttp
import json
from anthropic import AsyncAnthropic
from agents.base import BaseAgent
from core.task import SubTask, TaskResult
from core.context import SharedContext
import config

_TOOLS = [
    {
        "name": "http_request",
        "description": "发送 HTTP 请求到 API",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
                "url": {"type": "string"},
                "headers": {"type": "object"},
                "body": {"type": "object"},
                "params": {"type": "object"},
            },
            "required": ["method", "url"],
        },
    },
]


class APIAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "APIAgent"

    @property
    def system_prompt(self) -> str:
        return f"""你是一个专业的 API 调用 agent。
你可以调用 REST API，处理 JSON 响应，整合多个 API 的数据。
当前会话上下文：
{self.context.recent_summary()}"""

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
        if name == "http_request":
            return [{"type": "text", "text": await self._http(input)}]
        return [{"type": "text", "text": f"Unknown tool: {name}"}]

    async def _http(self, input: dict) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                method = input["method"].upper()
                url = input["url"]
                headers = input.get("headers", {})
                params = input.get("params", {})
                body = input.get("body")

                async with session.request(
                    method, url, headers=headers, params=params,
                    json=body if body else None, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    text = await resp.text()
                    try:
                        data = json.loads(text)
                        return f"Status: {resp.status}\n{json.dumps(data, ensure_ascii=False, indent=2)}"
                    except Exception:
                        return f"Status: {resp.status}\n{text[:2000]}"
        except Exception as e:
            return f"Request failed: {e}"
