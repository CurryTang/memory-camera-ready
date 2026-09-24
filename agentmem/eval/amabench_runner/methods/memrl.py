"""MemRL adapter for AMABench.

Wraps :class:`agentmem.memrl.runtime.MemRLRuntimeEngine` (no PPO training
required — the engine is a runtime episodic store with similarity + utility
fused ranking + epsilon-greedy selection) into the AMABench
``memory_construction`` / ``memory_retrieve`` interface.

Build phase:
    parse the AMABench joined trajectory text back into per-turn
    ``{turn_idx, action, observation}`` records, then write each turn as an
    "experience" into an ``EpisodicMemoryStore`` exposed through the runtime.

Query phase:
    issue a domain-aware retrieval query (alfworld / webarena / spider2 hint),
    run the two-phase semantic + utility ranker, format the selected
    candidates as a context block. The downstream answer LLM is invoked by
    AMABench's shared answer prompt (``run_amabench.py:_qa_with_retrieved``),
    so this adapter only needs to return the *context string* — same shape
    as :class:`SimpleMemMethod`, :class:`PlugMemMethod`, etc.

Note on training: MemRL's runtime utility update is purely online via
``update_utilities`` after each QA. There is no upstream pretraining step
(unlike policy-gradient routers). To keep AMABench QA stateless across questions
within an episode (each question is independent), we do *not* call
``observe_outcome`` here — utilities stay at their q_init_pos defaults,
making this a 'similarity-anchored MemRL' baseline for AMABench. This
matches the LoCoMo single-pass evaluation pattern.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from agentmem.backends.episodic import EpisodicMemoryStore
from agentmem.eval.amabench_runner.methods.base import BaseMethod
from agentmem.memrl.runtime import MemRLConfig, MemRLRuntimeEngine
from agentmem.memrl.task_adaptation import (
    build_amabench_memrl_query,
    format_amabench_memory_context,
    parse_amabench_trajectory,
)

class MemRLAmaMemory:
    """Container holding a configured MemRL runtime after trajectory ingestion."""

    def __init__(
        self,
        runtime: MemRLRuntimeEngine,
        *,
        task: str,
        task_type: str,
        domain: str,
        num_turns: int,
    ) -> None:
        self.runtime = runtime
        self.task = task
        self.task_type = task_type
        self.domain = domain
        self.num_turns = num_turns

class MemRLMethod(BaseMethod):
    """AMABench-side MemRL adapter."""

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[str] = None,
        retrieval_topk: int = 5,
        phase1_topk: int = 20,
        alpha: float = 0.3,
        gamma: float = 0.0,
        epsilon: float = 0.1,
        similarity_weight: float = 0.5,
        utility_weight: float = 0.5,
        use_zscore_normalization: bool = True,

        **_extra: Any,
    ) -> None:
        super().__init__()
        if config is None and config_path:
            config = self._load_config(config_path)
        merged = dict(config) if config else {}

        retrieval_topk = int(merged.get("retrieval_topk", retrieval_topk))
        phase1_topk = int(merged.get("phase1_topk", phase1_topk))
        alpha = float(merged.get("alpha", alpha))
        gamma = float(merged.get("gamma", gamma))
        epsilon = float(merged.get("epsilon", epsilon))
        similarity_weight = float(merged.get("similarity_weight", similarity_weight))
        utility_weight = float(merged.get("utility_weight", utility_weight))
        use_zscore_normalization = bool(
            merged.get("use_zscore_normalization", use_zscore_normalization)
        )

        self._cfg = MemRLConfig(
            phase1_topk=max(1, phase1_topk),
            topk=max(1, retrieval_topk),
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            similarity_weight=similarity_weight,
            utility_weight=utility_weight,
            use_zscore_normalization=use_zscore_normalization,
        )

    def memory_construction(self, traj_text: str, task: str = "") -> MemRLAmaMemory:
        """Ingest AMABench trajectory turns into a fresh MemRL runtime."""
        store = EpisodicMemoryStore()
        runtime = MemRLRuntimeEngine(store=store, config=self._cfg)

        task_str = str(task or "").strip()

        task_type = ""
        domain = ""
        if "[" in task_str and "]" in task_str:
            head = task_str.split("]", 1)[0].lstrip("[").strip().lower()

            if head:
                domain = head

        turns = parse_amabench_trajectory(traj_text)
        for turn in turns:
            turn_idx = int(turn.get("turn_idx", 0))
            action = str(turn.get("action", "")).strip()
            observation = str(turn.get("observation", "")).strip()

            experience = (
                f"Action: {action}\nObservation: {observation}"
                if (action or observation)
                else ""
            )
            if not experience:
                continue
            intent = action or f"turn_{turn_idx}"
            runtime.add_experience(
                intent=intent,
                experience=experience,
                success=True,                                       
                metadata={
                    "source": "trajectory_turn",
                    "turn_idx": turn_idx,
                    "task": task_str,
                    "task_type": task_type,
                    "domain": domain,
                },
                task_id=f"turn_{turn_idx}",
            )
        return MemRLAmaMemory(
            runtime=runtime,
            task=task_str,
            task_type=task_type,
            domain=domain,
            num_turns=len(turns),
        )

    def memory_retrieve(self, memory: MemRLAmaMemory, question: str) -> str:
        if memory is None or memory.runtime is None:
            return ""
        query = build_amabench_memrl_query(
            question, task_type=memory.task_type, domain=memory.domain
        )
        retrieval = memory.runtime.retrieve(
            query, phase1_topk=self._cfg.phase1_topk, topk=self._cfg.topk
        )
        selected = retrieval.selected or retrieval.candidates[: self._cfg.topk]
        return format_amabench_memory_context(selected)
