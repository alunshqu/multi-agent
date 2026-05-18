import asyncio
import json
from openai import AsyncOpenAI
from core.task import SubTask, TaskResult, AgentType
from core.context import SharedContext
from agents.browser import BrowserAgent
from agents.code import CodeAgent
from agents.data import DataAgent
from agents.api_agent import APIAgent
from memory.store import MemoryStore
from memory.retriever import MemoryRetriever
from memory.promoter import MemoryPromoter
from memory.crystallizer import SkillCrystallizer
from memory.layers import Episode
from tools.forge import ToolForge
import config

_DECOMPOSE_TOOL = {
    "name": "create_plan",
    "description": "将用户请求分解为有序的子任务列表",
    "input_schema": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "agent": {
                            "type": "string",
                            "enum": ["browser", "code", "data", "api"],
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["id", "description", "agent"],
                },
            },
            "systems": {
                "type": "array",
                "description": "任务涉及的系统或工具名称，如 excel、crm、email 等",
                "items": {"type": "string"},
            },
            "tool_requests": {
                "type": "array",
                "description": "需要动态创建的新工具（当现有 agent 能力不足时填写）",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "工具名称，英文 snake_case",
                        },
                        "description": {
                            "type": "string",
                            "description": "工具功能的一句话描述",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "参数名到类型的映射，如 {\"file_path\": \"string\", \"sheet\": \"string\"}",
                        },
                        "reason": {
                            "type": "string",
                            "description": "为什么现有工具无法完成，需要新建",
                        },
                    },
                    "required": ["name", "description", "parameters"],
                },
            },
        },
        "required": ["tasks"],
    },
}

_ORCHESTRATOR_SYSTEM = """你是一个任务编排器，负责将用户请求分解并分配给专属 agent。

可用 agents：
- browser: 浏览器操作、网页交互、GUI 控制、截图
- code: 代码生成、脚本执行、文件操作、自动化（支持动态扩展工具）
- data: 数据分析、pandas、可视化、统计
- api: REST API 调用、HTTP 请求、数据集成

规则：
1. 尽量并行执行（depends_on 为空）
2. 只在有真实依赖时才设置 depends_on
3. 简单任务用单个 agent，复杂任务合理拆分
4. 每个子任务描述要完整、自包含
5. systems 字段填写任务涉及的具体系统/工具名称
6. 如果某个子任务需要特定能力但现有 agent 工具不足，在 tool_requests 中声明需要创建的新工具
   - 新工具会在执行前自动生成并挂载到 code agent
   - 只在确实需要时才请求，不要过度创建"""


class Orchestrator:
    def __init__(self, client: AsyncOpenAI, context: SharedContext, store: MemoryStore):
        self.client = client
        self.context = context
        self.store = store
        self.retriever = MemoryRetriever(store)
        self.promoter = MemoryPromoter(store, client)
        self.crystallizer = SkillCrystallizer(store, client)
        self.last_episode_id: str | None = None
        self._forge = ToolForge(client)
        self._code_agent = CodeAgent(client, context, self._forge)
        self._agents = {
            AgentType.BROWSER: BrowserAgent(client, context),
            AgentType.CODE: self._code_agent,
            AgentType.DATA: DataAgent(client, context),
            AgentType.API: APIAgent(client, context),
        }
        self.tools_created: list[str] = []  # 本次任务新创建的工具名

    async def run(self, user_request: str) -> str:
        self.tools_created = []

        # 1. 检索记忆（初次，agents 未知，用空列表）
        memory_ctx = self.retriever.retrieve(user_request, agents=[], systems=[])
        memory_prompt = memory_ctx.to_prompt()

        # 2. 分解任务
        subtasks, systems, tool_requests = await self._decompose(user_request, memory_prompt)

        # 3. 用已知 agents + systems 精确检索（可能命中 Skill）
        agents_used = list({t.agent.value for t in subtasks})
        if agents_used or systems:
            memory_ctx = self.retriever.retrieve(user_request, agents=agents_used, systems=systems)
            memory_prompt = memory_ctx.to_prompt()

        # 4. 创建缺失工具（执行前完成，让 agent 能立即使用）
        if tool_requests:
            await self._forge_tools(tool_requests)

        # 5. 执行
        results = await self._execute(subtasks)

        # 6. 整合结果
        final_text = await self._synthesize(user_request, results)

        # 7. 写入 Episode，触发提炼和进化（异步）
        asyncio.create_task(
            self._record_episode(user_request, subtasks, systems, results, final_text)
        )

        return final_text

    async def _decompose(self, request: str, memory_prompt: str) -> tuple[list[SubTask], list[str], list[dict]]:
        context_note = self.context.recent_summary()
        content = f"请求：{request}\n\n会话历史：\n{context_note}"
        if memory_prompt:
            content = f"{memory_prompt}\n\n{content}"

        oai_tool = {
            "type": "function",
            "function": {
                "name": _DECOMPOSE_TOOL["name"],
                "description": _DECOMPOSE_TOOL["description"],
                "parameters": _DECOMPOSE_TOOL["input_schema"],
            },
        }
        response = await self.client.chat.completions.create(
            model=config.ORCHESTRATOR_MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": _ORCHESTRATOR_SYSTEM},
                {"role": "user", "content": content},
            ],
            tools=[oai_tool],
            tool_choice={"type": "function", "function": {"name": "create_plan"}},
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            import json as _json
            args = _json.loads(msg.tool_calls[0].function.arguments)
            raw_tasks = args.get("tasks", [])
            systems = args.get("systems", [])
            tool_requests = args.get("tool_requests", [])
            subtasks = [
                SubTask(
                    id=t["id"],
                    description=t["description"],
                    agent=AgentType(t["agent"]),
                    depends_on=t.get("depends_on", []),
                )
                for t in raw_tasks
            ]
            return subtasks, systems, tool_requests
        return [SubTask("t1", request, AgentType.CODE)], [], []

    async def _forge_tools(self, requests: list[dict]):
        """并行创建所有请求的工具，注册到 CodeAgent。"""
        async def _create_one(req: dict):
            try:
                tool_def = await self._forge.create(
                    name=req["name"],
                    description=req["description"],
                    parameters=req.get("parameters", {}),
                    reason=req.get("reason", ""),
                )
                self._code_agent.add_tool(tool_def)
                self.tools_created.append(req["name"])
            except Exception as e:
                # 工具创建失败不阻断任务，记录即可
                self.tools_created.append(f"{req['name']}(失败: {e})")

        await asyncio.gather(*[_create_one(r) for r in requests])

    async def _execute(self, subtasks: list[SubTask]) -> list[TaskResult]:
        completed: dict[str, TaskResult] = {}
        pending = list(subtasks)
        while pending:
            ready = [t for t in pending if all(d in completed for d in t.depends_on)]
            if not ready:
                break
            batch = await asyncio.gather(
                *[self._agents[t.agent].execute(t) for t in ready]
            )
            for task, result in zip(ready, batch):
                completed[task.id] = result
                pending.remove(task)
        return list(completed.values())

    async def _synthesize(self, request: str, results: list[TaskResult]) -> str:
        if not results:
            return "没有执行任何任务。"
        parts = []
        for r in results:
            mark = "✓" if r.success else "✗"
            parts.append(f"[{mark} {r.agent}]\n{r.output or r.error}")
        combined = "\n\n".join(parts)
        response = await self.client.chat.completions.create(
            model=config.ORCHESTRATOR_MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": "你是一个结果整合器。将多个 agent 的执行结果整合为清晰、简洁的最终回复。"},
                {"role": "user", "content": f"原始请求：{request}\n\n各 agent 结果：\n{combined}\n\n请整合为最终回复。"},
            ],
        )
        text = response.choices[0].message.content or ""
        if self.tools_created:
            names = "、".join(self.tools_created)
            text += f"\n\n*（本次自动创建了新工具：{names}，已保存供后续使用）*"
        return text

    async def _record_episode(
        self,
        request: str,
        subtasks: list[SubTask],
        systems: list[str],
        results: list[TaskResult],
        summary: str,
    ):
        agents_used = list({t.agent.value for t in subtasks})
        task_shape = (
            "single" if len(subtasks) == 1
            else "parallel" if all(not t.depends_on for t in subtasks)
            else "sequential"
        )
        all_success = all(r.success for r in results)
        outcome = "success" if all_success else ("partial" if any(r.success for r in results) else "failure")
        failure_reason = None
        if not all_success:
            failed = [r.error for r in results if not r.success and r.error]
            failure_reason = failed[0][:200] if failed else "unknown"

        episode = Episode(
            intent=request,
            agents_used=agents_used,
            task_shape=task_shape,
            systems=systems or agents_used,
            outcome=outcome,
            execution_summary=summary[:400],
            failure_reason=failure_reason,
        )
        self.store.save_episode(episode)
        self.last_episode_id = episode.id
        self.context.add_event("Orchestrator", request, summary[:200])

        # 触发失败 Pattern 提炼
        await self.promoter.process(episode)
        # 触发 Skill 提炼 / 进化
        await self.crystallizer.process_episode(episode)
