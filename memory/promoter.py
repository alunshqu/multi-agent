from anthropic import AsyncAnthropic
from memory.store import MemoryStore
from memory.layers import Episode, Pattern, Procedure
import config

# 同类失败达到此阈值才提炼为 Pattern
_PROMOTE_THRESHOLD = 3


class MemoryPromoter:
    def __init__(self, store: MemoryStore, client: AsyncAnthropic):
        self.store = store
        self.client = client

    async def process(self, episode: Episode):
        """Episode 写入后异步触发提炼检查。"""
        if episode.outcome == "failure" and episode.failure_reason:
            await self._check_failure_promotion(episode)

        if episode.outcome == "success":
            for system in episode.systems:
                if not self.store.procedure_exists(system):
                    self._create_procedure_stub(system, episode)

    async def _check_failure_promotion(self, episode: Episode):
        similar = self.store.find_failure_episodes(episode.systems)
        if len(similar) < _PROMOTE_THRESHOLD:
            return
        # 避免重复提炼：检查是否已有覆盖这些 episode 的 Pattern
        existing = self.store.load_all_patterns()
        existing_sources = {eid for p in existing for eid in p.source_episodes}
        new_episodes = [e for e in similar if e.id not in existing_sources]
        if len(new_episodes) < _PROMOTE_THRESHOLD:
            return
        await self._promote_to_pattern(new_episodes[:5])

    async def _promote_to_pattern(self, episodes: list[Episode]):
        summaries = "\n".join(
            f"- 任务：{e.intent}，失败原因：{e.failure_reason}"
            for e in episodes
        )
        resp = await self.client.messages.create(
            model=config.AGENT_MODEL,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": (
                    f"以下是多次类似失败的案例：\n{summaries}\n\n"
                    "请提炼出一条简洁的通用规律（一句话），帮助未来避免同类错误。"
                    "只输出规律本身，不要任何解释或前缀。"
                ),
            }],
        )
        content = resp.content[0].text.strip()
        pattern = Pattern(
            content=content,
            trigger_keywords=list({s for e in episodes for s in e.systems})[:4],
            confidence=0.7,
            source_episodes=[e.id for e in episodes],
        )
        self.store.save_pattern(pattern)

    def _create_procedure_stub(self, system: str, episode: Episode):
        proc = Procedure(
            system_name=system,
            content=episode.execution_summary[:400],
            description=f"自动提取自：{episode.intent[:60]}",
            success_count=1,
        )
        self.store.save_procedure(proc)
