"""Unified method base classes and hardware-independent counters.

The Codex review drove the final shape of :class:`EfficiencyCounters`: only
universal fields live here; family-specific diagnostics (graph nodes/edges,
subgraph sizes, etc.) go into ``family_specific`` and are reported in paper
appendix tables, not in the main comparison.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping

class MethodKind(str, Enum):
    """Where a method falls in the token-reduction taxonomy."""

    LONG_CONTEXT = "long_context"
    LEXICAL = "lexical"
    DENSE = "dense"
    SUMMARY = "summary"
    STRUCTURED = "structured"
    AGENTIC = "agentic"
    PARAMETRIC = "parametric"
    ORACLE = "oracle"
    SHORT_CONTEXT = "short_context"

@dataclass
class EfficiencyCounters:
    """Hardware-independent counters collected during method execution.

    All fields are counts, token sums, or byte sums — never seconds. Wallclock
    and GPU-seconds are tracked separately (see ``wallclock_seconds``) and are
    reported only as a secondary HW-dependent block in the paper.

    Universal fields apply to every method family. ``family_specific`` holds
    graph node/edge counts, chunk statistics, etc., which are only meaningful
    for a subset of methods and are relegated to appendix tables.
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    retrieved_context_tokens: int = 0
    exact_prefix_reusable_tokens: int = 0
    token_overlap_tokens: int = 0
    retrieval_calls: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    persistent_store_bytes: int = 0
    candidate_units_scored: int = 0
    evidence_units_injected: int = 0

    build_input_tokens: int = 0
    build_output_tokens: int = 0
    query_input_tokens: int = 0
    query_output_tokens: int = 0

    wallclock_seconds: float = 0.0
    build_wallclock_seconds: float = 0.0

    seconds_by_phase: dict[str, float] = field(default_factory=dict)

    family_specific: dict[str, Any] = field(default_factory=dict)

    def record_llm_call(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        phase: str = "query",
    ) -> None:
        """Account a single LLM call. ``phase`` is ``"build"`` or ``"query"``."""
        self.llm_calls += 1
        self.total_input_tokens += int(prompt_tokens)
        self.total_output_tokens += int(completion_tokens)
        if phase == "build":
            self.build_input_tokens += int(prompt_tokens)
            self.build_output_tokens += int(completion_tokens)
        elif phase == "query":
            self.query_input_tokens += int(prompt_tokens)
            self.query_output_tokens += int(completion_tokens)
        else:
            raise ValueError(f"phase must be 'build' or 'query', got {phase!r}")

    def record_retrieval(
        self,
        *,
        candidates_scored: int,
        evidence_injected: int,
        context_tokens: int,
    ) -> None:
        self.retrieval_calls += 1
        self.candidate_units_scored += int(candidates_scored)
        self.evidence_units_injected += int(evidence_injected)
        self.retrieved_context_tokens += int(context_tokens)

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    @contextmanager
    def time_block(self, attr: str) -> Iterator[None]:
        """Accumulate wallclock into ``attr`` (kept separate from token metrics)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            setattr(self, attr, float(getattr(self, attr, 0.0)) + elapsed)

    def add_seconds(self, phase: str, seconds: float) -> None:
        """Record per-phase wallclock (the 7-metric ``w_llm`` / ``w_tool`` / ``w_llm_build`` slots).

        Adapters call this around LLM API calls (``add_seconds("w_llm", elapsed)``)
        and around retrieval / graph traversal (``add_seconds("w_tool", elapsed)``).
        Build-phase LLM time goes into ``w_llm_build``. Other phase keys are
        accepted but the canonical reporting only consumes the three above.
        """
        if seconds <= 0:
            return
        self.seconds_by_phase[phase] = float(self.seconds_by_phase.get(phase, 0.0)) + float(seconds)

    @property
    def w_llm(self) -> float:
        return float(self.seconds_by_phase.get("w_llm", 0.0))

    @property
    def w_tool(self) -> float:
        return float(self.seconds_by_phase.get("w_tool", 0.0))

    @property
    def w_llm_build(self) -> float:
        return float(self.seconds_by_phase.get("w_llm_build", 0.0))

    def to_dict(self) -> dict[str, Any]:
        out = {
            "total_input_tokens": int(self.total_input_tokens),
            "total_output_tokens": int(self.total_output_tokens),
            "retrieved_context_tokens": int(self.retrieved_context_tokens),
            "exact_prefix_reusable_tokens": int(self.exact_prefix_reusable_tokens),
            "token_overlap_tokens": int(self.token_overlap_tokens),
            "retrieval_calls": int(self.retrieval_calls),
            "tool_calls": int(self.tool_calls),
            "llm_calls": int(self.llm_calls),
            "persistent_store_bytes": int(self.persistent_store_bytes),
            "candidate_units_scored": int(self.candidate_units_scored),
            "evidence_units_injected": int(self.evidence_units_injected),
            "build_input_tokens": int(self.build_input_tokens),
            "build_output_tokens": int(self.build_output_tokens),
            "query_input_tokens": int(self.query_input_tokens),
            "query_output_tokens": int(self.query_output_tokens),
            "wallclock_seconds": float(self.wallclock_seconds),
            "build_wallclock_seconds": float(self.build_wallclock_seconds),
            "w_llm": self.w_llm,
            "w_tool": self.w_tool,
            "w_llm_build": self.w_llm_build,
        }
        if self.seconds_by_phase:
            out["seconds_by_phase"] = dict(self.seconds_by_phase)
        if self.family_specific:
            out["family_specific"] = dict(self.family_specific)
        return out

class _MemoryHandleProtocol:
    """Opaque marker for per-method memory handles returned from ``build``."""

class BaseMethod(ABC):
    """Interface for non-agentic memory methods.

    The method ``build`` digests a source trajectory into a memory handle,
    and ``answer`` uses the handle plus a question to produce a prediction.
    Every adapter must report counters for both phases.
    """

    kind: MethodKind = MethodKind.LONG_CONTEXT
    name: str = ""

    @abstractmethod
    def build(self, traj_text: str, *, task: str = "") -> Any:
        """Build a memory handle from the full source trajectory."""

    @abstractmethod
    def answer(self, memory: Any, question: str) -> str:
        """Answer a question against a prebuilt memory handle."""

    def persistent_store_bytes(self, memory: Any) -> int:
        return 0

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        path = Path(config_path)
        with path.open("r") as f:
            if path.suffix in (".yaml", ".yml"):
                import yaml

                return yaml.safe_load(f) or {}
            return json.load(f)

class BaseAgenticMethod(ABC):
    """Interface for ReAct-style methods (ALFWorld, agentic HotpotQA, AMA-Agent).

    The method receives an environment step callback and must drive the loop,
    recording every tool call through its :class:`EfficiencyCounters`.
    """

    kind: MethodKind = MethodKind.AGENTIC
    name: str = ""

    @abstractmethod
    def run_episode(
        self,
        *,
        task: str,
        step_fn: Any,
        max_steps: int = 50,
    ) -> Any:
        """Drive one agent episode. Returns the final prediction / trace."""

class MeteredMethod:
    """Mixin that owns an :class:`EfficiencyCounters` instance.

    Adapters compose this with :class:`BaseMethod` / :class:`BaseAgenticMethod`
    so all counter plumbing is shared. Use as::

        class BM25Method(MeteredMethod, BaseMethod):
            kind = MethodKind.LEXICAL
            ...

    A fresh counter object is created per ``reset_counters`` call (typically
    at the start of each episode).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._counters = EfficiencyCounters()

    @property
    def counters(self) -> EfficiencyCounters:
        return self._counters

    def reset_counters(self) -> EfficiencyCounters:
        self._counters = EfficiencyCounters()
        return self._counters

def exact_prefix_reusable_tokens(prompts: list[list[int]]) -> int:
    """Length of the longest token prefix common to every prompt.

    This is the strict lower bound Codex recommended: it matches exactly what a
    paged-attention KV cache can actually reuse when the first request primes
    the prefix cache. Returns 0 for an empty list or any empty prompt.
    """
    if not prompts or any(len(p) == 0 for p in prompts):
        return 0
    shortest = min(len(p) for p in prompts)
    limit = 0
    for i in range(shortest):
        ref = prompts[0][i]
        if all(p[i] == ref for p in prompts[1:]):
            limit = i + 1
        else:
            break
    return limit

def token_overlap_tokens(prompts: list[list[int]]) -> int:
    """Order-insensitive upper-ish bound on reusable tokens.

    Multiset intersection across all prompts, summed. Complements
    :func:`exact_prefix_reusable_tokens`: retrieval methods typically score low
    on the prefix metric (retrieved evidence changes) but high here.
    """
    if not prompts:
        return 0
    from collections import Counter

    counters = [Counter(p) for p in prompts]
    shared: Mapping[int, int] = counters[0]
    for other in counters[1:]:
        shared = {tok: min(shared[tok], other[tok]) for tok in shared if tok in other}
    return int(sum(shared.values()))
