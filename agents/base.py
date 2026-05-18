from abc import ABC, abstractmethod
from anthropic import AsyncAnthropic
from core.task import SubTask, TaskResult
from core.context import SharedContext
import config


class BaseAgent(ABC):
    def __init__(self, client: AsyncAnthropic, context: SharedContext):
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

    def _system_block(self) -> list:
        return [{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}]

    async def _agentic_loop(self, messages: list, tools: list, extra_kwargs: dict = None) -> str:
        """Run the tool-use loop until Claude stops calling tools."""
        kwargs = {
            "model": config.AGENT_MODEL,
            "max_tokens": config.MAX_TOKENS,
            "system": self._system_block(),
            "messages": messages,
            "tools": tools,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        while True:
            response = await self.client.messages.create(**kwargs)

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                texts = [b.text for b in response.content if b.type == "text"]
                return "\n".join(texts)

            messages = messages + [{"role": "assistant", "content": response.content}]
            tool_results = []
            for tu in tool_uses:
                result_content = await self._handle_tool(tu.name, tu.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_content if isinstance(result_content, list) else [result_content],
                })
            messages = messages + [{"role": "user", "content": tool_results}]
            kwargs["messages"] = messages

    async def _handle_tool(self, name: str, input: dict) -> list:
        return [{"type": "text", "text": f"Tool {name} not implemented"}]
