"""Long-context baseline adapter for the 3-factor evaluation.

Zero-retrieval baseline: feeds the full (optionally truncated) trajectory
as context for every question. This is the upper-bound on token usage and
the lower-bound on retrieval complexity, used as the reference point in
Pareto comparisons.

Ported from ``agentmem.eval.amabench_runner.methods.longcontext`` to the
unified :class:`BaseMethod` / :class:`MeteredMethod` interface.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from agentmem.methods.base import (
    BaseMethod,
    EfficiencyCounters,
    MeteredMethod,
    MethodKind,
)

class _LongContextMemory:
    """Opaque handle holding the full trajectory text."""

    __slots__ = ("full_text",)

    def __init__(self, full_text: str) -> None:
        self.full_text = full_text

class LongContextMethod(MeteredMethod, BaseMethod):
    """Zero-overhead baseline: full trajectory as context.

    Instrumentation notes:

    - ``build()`` is essentially free -- it just stores the text.
    - ``answer()`` returns the (possibly truncated) trajectory and records
      it as a single retrieval with ``evidence_units_injected=1``.
    - No LLM calls happen inside the method itself; the benchmark runner
      runs the shared QA prompt.
    """

    kind = MethodKind.LONG_CONTEXT
    name = "longcontext"

    batch_mode = True

    def __init__(
        self,
        *,
        max_model_length: int = 16384,
        max_response_tokens: int = 4096,
        chars_per_token: int = 4,
        overflow_mode: str | None = None,
        search_top_k: int = 8,
        search_chunk_chars: int = 2200,
        search_chunk_overlap: int = 300,
        config_path: Optional[str] = None,
        token_counter: Optional[Any] = None,
        **_kw: Any,
    ) -> None:
        super().__init__()
        if config_path:
            cfg = self._load_config(config_path)
            max_model_length = cfg.get("max_model_length", max_model_length)
            max_response_tokens = cfg.get("max_response_tokens", max_response_tokens)
            chars_per_token = cfg.get("chars_per_token", chars_per_token)
            overflow_mode = cfg.get("overflow_mode", overflow_mode)
            search_top_k = cfg.get("search_top_k", search_top_k)
            search_chunk_chars = cfg.get("search_chunk_chars", search_chunk_chars)
            search_chunk_overlap = cfg.get("search_chunk_overlap", search_chunk_overlap)
        self.max_input_tokens = max_model_length - max_response_tokens
        self.chars_per_token = chars_per_token
        self._token_counter = token_counter
        self.overflow_mode = (overflow_mode or os.environ.get("LONGCONTEXT_OVERFLOW_MODE", "search")).lower()
        self.search_top_k = max(1, int(os.environ.get("LONGCONTEXT_SEARCH_TOP_K", search_top_k)))
        self.search_chunk_chars = max(200, int(os.environ.get("LONGCONTEXT_SEARCH_CHUNK_CHARS", search_chunk_chars)))
        self.search_chunk_overlap = max(0, int(os.environ.get("LONGCONTEXT_SEARCH_CHUNK_OVERLAP", search_chunk_overlap)))

    def build(self, traj_text: str, *, task: str = "") -> Any:
        with self._counters.time_block("build_wallclock_seconds"):
            full_text = (
                f"# Task\n{task}\n\n# Agent Trajectory\n{traj_text}"
                if task
                else traj_text
            )
        return _LongContextMemory(full_text)

    def memory_construction(self, traj_text: str, task: str = "") -> Any:
        return self.build(traj_text, task=task)

    def memory_retrieve(self, memory: Any, question: str) -> str:
        return self.answer(memory, question)

    def answer(self, memory: Any, question: str) -> str:
        context = self._select_context(memory.full_text, question)
        ctx_tokens = self._count_tokens(context)
        self._counters.record_retrieval(
            candidates_scored=0,
            evidence_injected=1,
            context_tokens=ctx_tokens,
        )
        return context

    def persistent_store_bytes(self, memory: Any) -> int:
        return 0

    def _select_context(self, text: str, question: str) -> str:
        est_tokens = len(text) / self.chars_per_token
        if est_tokens <= self.max_input_tokens:
            return text
        if self.overflow_mode in {"truncate", "middle_truncate", "middle-truncate"}:
            return self._truncate(text)
        return self._search_context(text, question)

    def _truncate(self, text: str) -> str:
        est_tokens = len(text) / self.chars_per_token
        if est_tokens <= self.max_input_tokens:
            return text
        target = int(self.max_input_tokens * self.chars_per_token)
        half = target // 2
        return (
            text[:half]
            + "\n\n... [middle section truncated] ...\n\n"
            + text[-half:]
        )

    def _search_context(self, text: str, question: str) -> str:
        """ReAct-style overflow path: search chunks, then pass observations."""
        target = int(self.max_input_tokens * self.chars_per_token)
        query_terms = self._query_terms(question)
        chunks = self._chunks(text)
        scored: list[tuple[float, int, str]] = []
        for idx, chunk in enumerate(chunks):
            lowered = chunk.lower()
            score = 0.0
            for term in query_terms:
                count = lowered.count(term)
                if count:
                    score += 1.0 + count
            if score:
                scored.append((score, idx, chunk))
        if not scored:

            return self._truncate(text)

        selected = sorted(scored, key=lambda item: (-item[0], item[1]))[: self.search_top_k]
        lines = [
            "# Long Context Overflow Search",
            "Action: Search the full context for question-relevant evidence.",
            f"Question: {question}",
            "",
        ]
        used = sum(len(line) + 1 for line in lines)
        budget = max(1000, target)
        for rank, (score, idx, chunk) in enumerate(selected, start=1):
            header = f"Observation {rank} (chunk={idx}, lexical_score={score:.1f}):\n"
            block = header + chunk.strip() + "\n"
            if used + len(block) > budget:
                remaining = budget - used - len(header) - 1
                if remaining <= 200:
                    break
                block = header + chunk.strip()[:remaining].rstrip() + "\n"
            lines.append(block)
            used += len(block)
        return "\n".join(lines).strip()

    def _chunks(self, text: str) -> list[str]:
        step = max(1, self.search_chunk_chars - min(self.search_chunk_overlap, self.search_chunk_chars - 1))
        return [text[i : i + self.search_chunk_chars] for i in range(0, len(text), step)]

    @staticmethod
    def _query_terms(question: str) -> list[str]:
        words = re.findall(r"[A-Za-z0-9_]+", question.lower())
        stop = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "how", "in", "is", "it", "of", "on", "or", "the", "to", "was",
            "were", "what", "when", "where", "which", "who", "why", "with",
        }
        return [word for word in words if len(word) > 2 and word not in stop]

    def _count_tokens(self, text: Any) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0
        return len(str(text or "").split())
