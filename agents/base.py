from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from core.task import SubTask, TaskResult
from core.context import SharedContext
import config


class BaseAgent(ABC):
    def __init__(self, client: AsyncOpenAI, context: SharedContext):
        self.client = client
        self.context = context

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @property
    def tools(self) -> list:
        return []

    @abstractmethod
    async def execute(self, task: SubTask) -> TaskResult:
        pass

    def _oai_tools(self, tools: list) -> list:
        """把工具定义转成 OpenAI function 格式。"""
        result = []
        for t in tools:
            # 跳过 computer_use / bash beta 工具（browser agent 专用，单独处理）
            if t.get("type") and t["type"] != "function":
                continue
            result.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return result

    async def _agentic_loop(self, messages: list, tools: list, **kwargs) -> str:
        """OpenAI tool_calls agentic loop，直到模型不再调用工具为止。"""
        oai_tools = self._oai_tools(tools)
        system = [{"role": "system", "content": self.system_prompt}]

        history = system + messages
        call_kwargs = dict(
            model=config.AGENT_MODEL,
            max_tokens=config.MAX_TOKENS,
            messages=history,
        )
        if oai_tools:
            call_kwargs["tools"] = oai_tools
            call_kwargs["tool_choice"] = "auto"

        while True:
            resp = await self.client.chat.completions.create(**call_kwargs)
            msg = resp.choices[0].message

            # 没有工具调用 → 返回文本
            if not msg.tool_calls:
                return msg.content or ""

            # 追加 assistant 消息
            history = history + [msg]

            # 执行所有工具调用
            tool_results = []
            for tc in msg.tool_calls:
                import json
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                result_content = await self._handle_tool(tc.function.name, args)
                # result_content 是 list[dict]，取第一个 text
                text = result_content[0].get("text", "") if result_content else ""
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": text,
                })

            history = history + tool_results
            call_kwargs["messages"] = history

    async def _handle_tool(self, name: str, input: dict) -> list:
        return [{"type": "text", "text": f"Tool {name} not implemented"}]
