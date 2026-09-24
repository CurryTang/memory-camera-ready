"""Typed data and memory interfaces for the MemoryArena harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

@dataclass(frozen=True)
class MemoryArenaSession:
    """One subtask/session inside a MemoryArena task."""

    session_id: int
    question: str
    gold: Any | None = None
    background: str = ""

@dataclass(frozen=True)
class MemoryArenaTask:
    """Normalized MemoryArena task row.

    The HF dataset and local JSONL snapshots are row-oriented. This class gives
    the harness a stable interface while preserving the original row in ``raw``.
    """

    task_id: str
    config: str
    sessions: list[MemoryArenaSession]
    raw: dict[str, Any] = field(default_factory=dict)
    source_index: int | None = None

@dataclass
class MemoryRecord:
    """Persistent memory record written after a completed session."""

    task_id: str
    session_id: int
    kind: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

class MemoryInterface(Protocol):
    """Minimal MemoryArena memory protocol.

    Implementations retrieve once at the beginning of each session and update
    once at session end. Normal evaluation must not write the gold answer into
    memory; oracle-memory ablations should use a separate explicit interface.
    """

    def reset(self, task_id: str) -> None:
        """Start a new MemoryArena task and clear previous state."""

    def retrieve(self, query: str, k: int = 5) -> str:
        """Return memory text for the current session."""

    def update(
        self,
        *,
        session_id: int,
        question: str,
        prediction: str,
        feedback: dict[str, Any],
        trace: str | None = None,
        gold: Any | None = None,
    ) -> None:
        """Write session-observable state after the agent receives feedback."""
