from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass(frozen=True)
class PlugMemStep:
    """Canonical PlugMem step."""

    index: int
    speaker: str
    action: str
    observation: str
    timestamp: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PlugMemQuestion:
    """Canonical PlugMem question."""

    index: int
    question: str
    answer: Optional[str]
    category: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class PlugMemSession:
    """Canonical PlugMem session payload."""

    session_id: str
    goal: str
    steps: list[PlugMemStep] = field(default_factory=list)
    questions: list[PlugMemQuestion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
