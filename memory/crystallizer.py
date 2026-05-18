import json
from openai import AsyncOpenAI
from memory.store import MemoryStore
from memory.layers import Episode, Skill, EvolutionEntry
from datetime import datetime
import config

# 触发提炼的最低相似成功次数
_CRYSTALLIZE_THRESHOLD = 2
# 触发进化的最低偏差次数
_EVOLVE_THRESHOLD = 2

_CRYSTALLIZE_PROMPT = """以下是 {n} 次成功完成类似任务的记录：

{summaries}

请提炼出一个可复用的工作流技能，以 JSON 格式输出，包含以下字段：
- name: 技能名称（2-6个字，中文）
- description: 一句话描述这个技能做什么
- trigger_patterns: 触发词列表（哪些关键词/意图会用到这个技能，3-6个）
- workflow: 分步骤的工作流程描述（含从这些案例中学到的注意事项，用\\n分隔步骤）
- parameters: 可变参数字典（每次执行可能不同的东西，如 {{"month": "报表月份"}}，没有则为 {{}}）

只输出 JSON，不要任何解释。"""

_EVOLVE_PROMPT = """现有技能「{name}」的工作流程：
{workflow}

以下是 {n} 次使用该技能但出现偏差的案例：
{deviations}

请更新工作流程以融入这些新经验，输出 JSON：
- workflow: 更新后的工作流程（保留原有内容，融入新经验）
- reason: 本次进化的原因（一句话）
- parameters: 更新后的可变参数字典

只输出 JSON，不要任何解释。"""


class SkillCrystallizer:
    def __init__(self, store: MemoryStore, client: AsyncOpenAI):
        self.store = store
        self.client = client

    async def process_episode(self, episode: Episode):
        """Episode 写入后检查是否可以提炼或进化 Skill。"""
        if episode.outcome != "success":
            return

        # 1. 检查是否匹配现有 Skill → 考虑进化
        existing = self.store.find_matching_skills(
            episode.intent, episode.agents_used, episode.systems
        )
        if existing:
            await self._maybe_evolve(existing[0], episode)
            return

        # 2. 没有匹配 Skill → 检查是否达到提炼阈值
        similar = self.store.find_success_episodes(
            episode.agents_used, episode.systems
        )
        # 排除自身
        similar = [e for e in similar if e.id != episode.id]

        if len(similar) >= _CRYSTALLIZE_THRESHOLD:
            candidates = similar[:4] + [episode]
            ep_ids = [e.id for e in candidates]
            if not self.store.skill_covers_episodes(ep_ids):
                await self._crystallize(candidates)

    async def _crystallize(self, episodes: list[Episode]):
        """从一批相似成功 Episode 提炼新 Skill。"""
        summaries = "\n".join(
            f"{i+1}. 任务：{e.intent}\n   执行：{e.execution_summary[:200]}"
            for i, e in enumerate(episodes)
        )
        prompt = _CRYSTALLIZE_PROMPT.format(n=len(episodes), summaries=summaries)

        resp = await self.client.chat.completions.create(
            model=config.AGENT_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _extract_json(resp.choices[0].message.content)
        if not raw:
            return

        skill = Skill(
            name=raw.get("name", "未命名技能"),
            description=raw.get("description", ""),
            workflow=raw.get("workflow", ""),
            trigger_patterns=raw.get("trigger_patterns", []),
            typical_agents=list({a for e in episodes for a in e.agents_used}),
            typical_systems=list({s for e in episodes for s in e.systems}),
            parameters=raw.get("parameters", {}),
            version=1,
            usage_count=len(episodes),
            success_count=len(episodes),
            source_episodes=[e.id for e in episodes],
            evolution_log=[EvolutionEntry(version=1, reason="初始提炼")],
        )
        self.store.save_skill(skill)

    async def _maybe_evolve(self, skill: Skill, new_episode: Episode):
        """检查新 Episode 是否带来了值得进化的偏差。"""
        # 统计该 Skill 被使用后出现用户纠正的次数
        # 简化：如果 new_episode 的执行摘要与 skill.workflow 差异较大，记录偏差
        # 当偏差积累到阈值时触发进化
        self.store.increment_skill_usage(skill.id, success=True)

        # 检查是否有足够的新 episode 来触发进化
        # 策略：source_episodes 之外的成功 episode 数量达到阈值
        all_similar = self.store.find_success_episodes(
            skill.typical_agents, skill.typical_systems
        )
        new_episodes = [
            e for e in all_similar
            if e.id not in skill.source_episodes and e.id != new_episode.id
        ]
        new_episodes.append(new_episode)

        if len(new_episodes) < _EVOLVE_THRESHOLD:
            return

        await self._evolve(skill, new_episodes[:4])

    async def _evolve(self, skill: Skill, new_episodes: list[Episode]):
        """用新 Episode 进化现有 Skill。"""
        deviations = "\n".join(
            f"{i+1}. {e.intent}：{e.execution_summary[:200]}"
            for i, e in enumerate(new_episodes)
        )
        prompt = _EVOLVE_PROMPT.format(
            name=skill.name,
            workflow=skill.workflow,
            n=len(new_episodes),
            deviations=deviations,
        )
        resp = await self.client.chat.completions.create(
            model=config.AGENT_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _extract_json(resp.choices[0].message.content)
        if not raw:
            return

        new_version = skill.version + 1
        skill.workflow = raw.get("workflow", skill.workflow)
        skill.parameters = raw.get("parameters", skill.parameters)
        skill.version = new_version
        skill.source_episodes = list(set(skill.source_episodes + [e.id for e in new_episodes]))
        skill.evolution_log.append(
            EvolutionEntry(version=new_version, reason=raw.get("reason", "自动进化"))
        )
        skill.updated_at = datetime.now().isoformat()
        self.store.save_skill(skill)


def _extract_json(text: str) -> dict | None:
    """从 Claude 输出中提取 JSON 对象。"""
    import re
    text = text.strip()
    # 去掉 markdown 代码块
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text.strip())
    except Exception:
        # 尝试找到第一个 { ... } 块
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None
