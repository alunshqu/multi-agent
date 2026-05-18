from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


class AgentType(str, Enum):
    BROWSER = "browser"
    CODE = "code"
    DATA = "data"
    API = "api"
    SYSTEM = "system"


@dataclass
class SubTask:
    id: str
    description: str
    agent: AgentType
    depends_on: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    agent: str
    success: bool
    output: str
    error: Optional[str] = None
