from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


def _now() -> str:
    return datetime.now().isoformat()


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Pattern:
    """抽象规律，始终加载，不需要相似度检索。"""
    content: str
    trigger_keywords: list[str] = field(default_factory=list)
    confidence: float = 0.5
    source_episodes: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _uid("pat"))
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass
class Procedure:
    """特定系统的操作方法，按系统名精确匹配。"""
    system_name: str
    content: str
    description: str = ""
    success_count: int = 0
    failure_count: int = 0
    id: str = field(default_factory=lambda: _uid("proc"))
    last_used: str = field(default_factory=_now)
    created_at: str = field(default_factory=_now)


@dataclass
class Episode:
    """一次完整任务的执行记录，用于语义检索。"""
    intent: str
    agents_used: list[str]
    task_shape: str          # single / parallel / sequential
    systems: list[str]
    outcome: str             # success / failure / partial
    execution_summary: str
    user_feedback: str = "pending"   # pending / satisfied / corrected / no_response
    failure_reason: Optional[str] = None
    id: str = field(default_factory=lambda: _uid("ep"))
    created_at: str = field(default_factory=_now)


@dataclass
class EvolutionEntry:
    version: int
    reason: str
    changed_at: str = field(default_factory=_now)


@dataclass
class Skill:
    """从重复成功的工作流中提炼的可复用技能模板。"""
    name: str
    description: str
    workflow: str                        # 分步骤的工作流描述，含注意事项
    trigger_patterns: list[str] = field(default_factory=list)   # 触发关键词
    typical_agents: list[str] = field(default_factory=list)
    typical_systems: list[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)              # 可变参数模板
    version: int = 1
    usage_count: int = 0
    success_count: int = 0
    source_episodes: list[str] = field(default_factory=list)
    evolution_log: list[EvolutionEntry] = field(default_factory=list)
    id: str = field(default_factory=lambda: _uid("sk"))
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_prompt_block(self) -> str:
        params_str = ""
        if self.parameters:
            params_str = "\n可变参数：" + "、".join(
                f"{k}（{v}）" for k, v in self.parameters.items()
            )
        evol_str = ""
        if len(self.evolution_log) > 1:
            last = self.evolution_log[-1]
            evol_str = f"\n最近进化（v{last.version}）：{last.reason}"
        return (
            f"【技能：{self.name} v{self.version}】\n"
            f"{self.description}\n\n"
            f"工作流程：\n{self.workflow}"
            f"{params_str}{evol_str}"
        )


@dataclass
class MemoryContext:
    patterns: list[Pattern] = field(default_factory=list)
    procedures: list[Procedure] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([self.patterns, self.procedures, self.episodes, self.skills])

    def to_prompt(self) -> str:
        if self.is_empty():
            return ""
        parts = ["【记忆上下文】"]
        # Skills 优先展示，信息密度最高
        if self.skills:
            for s in self.skills:
                parts.append(s.to_prompt_block())
        if self.patterns:
            parts.append("通用规律（始终适用）：")
            for p in self.patterns:
                parts.append(f"  · {p.content}")
        if self.procedures:
            parts.append("已知系统操作方法：")
            for p in self.procedures:
                parts.append(f"  · {p.system_name}：{p.content[:200]}")
        if self.episodes:
            parts.append("相关历史经验：")
            for e in self.episodes:
                mark = "✓" if e.outcome == "success" else "✗"
                fb = "（用户已确认）" if e.user_feedback == "satisfied" else ""
                parts.append(f"  [{mark}] {e.intent}：{e.execution_summary[:150]}{fb}")
        return "\n".join(parts)
