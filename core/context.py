from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class SharedContext:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    data: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def set(self, key: str, value: Any):
        self.data[key] = value

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def add_event(self, agent: str, task: str, result: str):
        self.history.append({
            "agent": agent,
            "task": task[:120],
            "result": result[:300],
        })

    def recent_summary(self, n: int = 5) -> str:
        if not self.history:
            return "无历史记录"
        return "\n".join(
            f"[{h['agent']}] {h['task']}: {h['result']}"
            for h in self.history[-n:]
        )
