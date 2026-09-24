"""
Backward-compatible type exports.

These are used by downstream code (SLIME hooks, eval methods, agents).
The original Mem-T code uses trajectory_logger.py dataclasses internally;
these types bridge the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class MemTToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None

@dataclass
class MemTTraceStep:
    step: int
    op_id: str
    thought: str
    tool_call: dict[str, Any]
    observation: str
    source_turn_ids: list[str] = field(default_factory=list)
    op_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class MemTRetrievalResult:
    answer: str
    traces: list[MemTTraceStep] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

@dataclass
class MemTMemTrajectory:
    sample_id: str
    phase: str
    op_id: str
    input_context: list[dict[str, Any]]
    llm_response: str
    parsed_output: list[dict[str, Any]]
    source_turn_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class MemTConstructionResult:
    sample_id: str
    num_sessions: int
    num_batches: int
    memory_counts: dict[str, int] = field(default_factory=dict)
    trajectories: list[MemTMemTrajectory] = field(default_factory=list)
