from memory.store import MemoryStore
from memory.layers import MemoryContext


class MemoryRetriever:
    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve(self, intent: str, agents: list[str], systems: list[str]) -> MemoryContext:
        # Skills 优先：按 agent 序列 + 系统交集 + 触发词匹配
        skills = self.store.find_matching_skills(intent, agents, systems)

        # Pattern：无条件加载
        patterns = self.store.load_all_patterns()

        # Procedure：按系统名精确匹配
        procedures = self.store.find_procedures(systems)

        # Episode：只在没有匹配 Skill 时才检索（Skill 已经是提炼后的经验）
        episodes = []
        if not skills:
            episodes = self.store.find_similar_episodes(intent, systems, top_k=3)

        return MemoryContext(
            patterns=patterns,
            procedures=procedures,
            episodes=episodes,
            skills=skills,
        )
