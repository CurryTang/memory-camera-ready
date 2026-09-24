"""Multi-session memory interface for streaming benchmarks (MemoryArena, AMABench domains).

The existing single-shot ``MemoryInterface`` (build → answer) cannot express the
streaming pattern required by MemoryArena: enter session i with only the state
written by sessions 0..i-1, produce an answer, observe environment feedback,
write back without ever seeing the gold answer, carry that state into i+1.

This module defines the streaming protocol. Concrete adapters live in
``agentmem/eval/memoryarena_runner/adapters/`` and wrap each existing method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

@dataclass
class SessionFeedback:
    """All inputs the harness may pass to ``MultiSessionMemory.update``.

    The harness MUST construct this from agent-observable signals only:
    prediction, judge boolean, environment observations, agent reasoning trace.
    The gold answer must NEVER be written into ``judge_signal`` or ``observations``
    in normal evaluation. Oracle-memory ablations may opt in by passing a
    distinct ``oracle_gold`` field, but adapters must explicitly check for it.
    """

    session_id: int
    question: str
    prediction: str
    correct: bool | None = None
    judge_signal: dict[str, Any] = field(default_factory=dict)
    trace: str | None = None
    observations: list[str] = field(default_factory=list)

class MultiSessionMemory(Protocol):
    """Streaming memory protocol. See module docstring for invariants."""

    def reset(self, task_id: str, *, schema_hint: str | None = None) -> None:
        """Begin a new task. Clear all internal state.

        ``schema_hint`` is an optional domain label (e.g. "bundled_shopping",
        "amabench-text2sql") that lets schema-aware adapters pick a
        domain-specific belief structure.
        """

    def retrieve(self, query: str, *, session_id: int, k: int = 5) -> str:
        """Render the memory context for the agent at session ``session_id``.

        Returns free-form text that will be injected into the agent prompt.
        Must not raise if memory is empty; return a sentinel ("No prior memory.").
        """

    def update(self, fb: SessionFeedback) -> None:
        """Write back after session ``fb.session_id`` finishes.

        Adapters MUST NOT inspect any gold answer. The ``correct`` and
        ``judge_signal`` fields are derived from the harness's judge and are
        the only legitimate correctness signal.
        """
