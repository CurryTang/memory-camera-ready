"""Shared types for the evaluation framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

@dataclass
class TaskResult:
    task: str
    status: str
    duration_sec: float
    summary: Dict[str, Any]
    artifacts: list[Path] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [str(path) for path in self.artifacts]
        return payload
