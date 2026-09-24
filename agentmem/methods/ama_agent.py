"""AMA-Agent adapter for the 3-factor evaluation.

Thin wrapper around the existing amabench adapter at
``agentmem.eval.amabench_runner.methods.ama_agent.AMAAgentMethod``,
re-exported as a unified :class:`BaseMethod` / :class:`MeteredMethod`
so AMA-Agent results are directly comparable to the rest of the method
fleet with hardware-independent counters.

Note: AMA-Agent is an agentic retrieval method (embedding retrieval +
tool-augmented search with a causality graph). The adapter delegates all
logic to the legacy implementation and only adds counter instrumentation.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from agentmem.methods.base import (
    BaseMethod,
    EfficiencyCounters,
    MeteredMethod,
    MethodKind,
)

def _load_ama_agent_cls() -> Any:
    """Import AMAAgentMethod lazily so the module is import-safe without deps."""
    try:
        from agentmem.eval.amabench_runner.methods.ama_agent import AMAAgentMethod
    except ImportError as exc:
        raise ImportError(
            "AMAAgentMethod is not importable. Ensure its dependencies "
            "(agentmem.retrieval.ama_agent, openai) are installed."
        ) from exc
    return AMAAgentMethod

class AMAAgentAdapterMethod(MeteredMethod, BaseMethod):
    """Thin metered adapter around the legacy AMAAgentMethod.

    Instrumentation notes:

    - ``build()`` delegates to ``AMAAgentMethod.memory_construction()``
      which builds a state memory with causality graph and embeddings.
    - ``answer()`` delegates to ``AMAAgentMethod.memory_retrieve()``
      which does embedding retrieval + optional tool-augmented search.
    - We wrap both calls with wallclock timing and retrieval counters.
    """

    kind = MethodKind.AGENTIC
    name = "ama_agent"

    def __init__(
        self,
        *,
        token_counter: Optional[Any] = None,
        **inner_kwargs: Any,
    ) -> None:
        super().__init__()
        self._token_counter = token_counter
        AMAAgentMethod = _load_ama_agent_cls()
        self._inner = AMAAgentMethod(**inner_kwargs)

    def build(self, traj_text: str, *, task: str = "") -> Any:
        with self._counters.time_block("build_wallclock_seconds"):
            memory = self._inner.memory_construction(traj_text, task=task)
        return memory

    def memory_construction(self, traj_text: str, task: str = "") -> Any:
        return self.build(traj_text, task=task)

    def memory_retrieve(self, memory: Any, question: str) -> str:
        return self.answer(memory, question)

    def answer(self, memory: Any, question: str) -> str:
        with self._counters.time_block("wallclock_seconds"):
            context = self._inner.memory_retrieve(memory, question)

        ctx_tokens = self._count_tokens(context)
        self._counters.record_retrieval(
            candidates_scored=0,
            evidence_injected=1 if context else 0,
            context_tokens=ctx_tokens,
        )
        return str(context)

    def persistent_store_bytes(self, memory: Any) -> int:
        return 0

    def last_trajectory(self) -> Optional[dict[str, Any]]:
        """Proxy to the inner method's debug trajectory."""
        return self._inner.last_trajectory()

    def _count_tokens(self, text: Any) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0
        return len(str(text or "").split())
