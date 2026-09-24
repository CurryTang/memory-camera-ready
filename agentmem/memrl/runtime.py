from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from agentmem.backends.base import BaseMemoryStore, MemoryRecord
from agentmem.memrl.math_utils import safe_std, zscore

if TYPE_CHECKING:
    from agentmem.retrieval.base import Index, Retriever

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)

@dataclass
class MemRLConfig:
    """Runtime configuration for MemRL-style memory learning."""

    phase1_topk: int = 20
    topk: int = 5
    epsilon: float = 0.1
    alpha: float = 0.3
    gamma: float = 0.0
    q_init_pos: float = 0.0
    q_init_neg: float = 0.0
    q_floor: Optional[float] = None
    q_min_threshold: Optional[float] = None
    success_reward: float = 1.0
    failure_reward: float = -1.0
    similarity_threshold: float = 0.0
    unknown_threshold: Optional[float] = None
    similarity_weight: float = 0.5
    utility_weight: float = 0.5
    use_zscore_normalization: bool = True
    recency_boost: float = 0.0
    dedup_by_task_id: bool = False

@dataclass
class MemRLCandidate:
    """A scored memory candidate produced by two-phase retrieval."""

    memory_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: float = 0.0
    utility: float = 0.0
    similarity_z: float = 0.0
    utility_z: float = 0.0
    fused_score: float = 0.0
    task_id: Optional[str] = None

@dataclass
class MemRLRetrievalResult:
    """Full retrieval output: ranked candidates plus final selected set."""

    query: str
    candidates: list[MemRLCandidate] = field(default_factory=list)
    selected: list[MemRLCandidate] = field(default_factory=list)
    simmax: float = 0.0

    @property
    def selected_ids(self) -> list[str]:
        return [c.memory_id for c in self.selected]

class MemRLRuntimeEngine:
    """
    Native MemRL runtime learner for AgentMem.

    This class ports MemRL's core algorithmic loop into agentmem primitives:
    - Intent-Experience-Utility metadata on MemoryRecord
    - Two-phase retrieval (semantic recall -> value-aware ranking)
    - Utility update from runtime rewards
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        config: Optional[MemRLConfig] = None,
        *,
        rng: Optional[random.Random] = None,
        retriever: Optional["Retriever"] = None,
        retrieval_index: Optional["Index"] = None,
    ) -> None:
        self.store = store
        self.config = config or MemRLConfig()
        self._rng = rng or random.Random()
        self._retriever = retriever
        self._retrieval_index = retrieval_index

    def add_experience(
        self,
        *,
        intent: str,
        experience: str,
        success: bool,
        metadata: Optional[dict[str, Any]] = None,
        retrieved_memory_ids: Optional[list[str]] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Write one Intent-Experience-Utility memory item into the store."""
        meta = dict(metadata or {})
        base_q = self.config.q_init_pos if bool(success) else self.config.q_init_neg

        if "q_value" not in meta or meta.get("q_value") is None:
            meta["q_value"] = float(base_q)
        else:
            meta["q_value"] = _to_float(meta.get("q_value"), default=base_q)
        meta.setdefault("q_visits", 0)
        meta.setdefault("reward_ma", 0.0)
        meta.setdefault("success", bool(success))
        meta.setdefault("q_updated_at", _now_iso())
        meta.setdefault("last_used_at", _now_iso())
        meta.setdefault("intent", intent)
        meta.setdefault("experience", experience)
        meta.setdefault("full_content", f"Intent: {intent}\n\nExperience:\n{experience}")

        if retrieved_memory_ids:
            meta["related_memory_ids"] = [str(mid) for mid in retrieved_memory_ids if mid]
        if task_id is not None:
            meta.setdefault("task_id", str(task_id))

        record_content = str(meta.get("full_content") or f"Intent: {intent}\n\nExperience:\n{experience}")
        record = MemoryRecord(content=record_content, metadata=meta)
        return self.store.add(record)

    def retrieve(
        self,
        query: str,
        *,
        phase1_topk: Optional[int] = None,
        topk: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> MemRLRetrievalResult:
        """
        Two-phase retrieval:
        1) semantic recall from store.search
        2) value-aware ranking with epsilon-greedy selection
        """
        cfg = self.config
        stage1_k = max(int(phase1_topk or cfg.phase1_topk), int(topk or cfg.topk), 1)
        threshold = (
            cfg.similarity_threshold
            if similarity_threshold is None
            else float(similarity_threshold)
        )

        candidates: list[MemRLCandidate] = []
        if self._retriever is not None and self._retrieval_index is not None:
            retrieval_hits = self._retriever.retrieve(query, self._retrieval_index, k=stage1_k)
            if not retrieval_hits:
                return MemRLRetrievalResult(query=query)
            for rr in retrieval_hits:
                similarity = _to_float(rr.score, default=0.0)
                if similarity < threshold:
                    continue
                meta = dict(rr.metadata or {})
                q = _to_float(meta.get("q_value"), default=cfg.q_init_pos)
                if cfg.recency_boost > 0.0 and meta.get("last_used_at"):
                    q += cfg.recency_boost
                if cfg.q_floor is not None:
                    q = max(float(cfg.q_floor), q)
                if cfg.q_min_threshold is not None and q < float(cfg.q_min_threshold):
                    continue
                raw_task_id = meta.get("task_id") or meta.get("sample_index") or meta.get("id")
                task_id = str(raw_task_id) if raw_task_id is not None else None
                candidates.append(
                    MemRLCandidate(
                        memory_id=rr.doc_id,
                        content=rr.content,
                        metadata=meta,
                        similarity=similarity,
                        utility=q,
                        task_id=task_id,
                    )
                )
        else:
            base_results = self.store.search(query, k=stage1_k)
            if not base_results:
                return MemRLRetrievalResult(query=query)
            for result in base_results:
                similarity = _to_float(result.score, default=0.0)
                if similarity < threshold:
                    continue
                rec = result.record
                meta = dict(rec.metadata or {})
                q = _to_float(meta.get("q_value"), default=cfg.q_init_pos)
                if cfg.recency_boost > 0.0 and meta.get("last_used_at"):
                    q += cfg.recency_boost
                if cfg.q_floor is not None:
                    q = max(float(cfg.q_floor), q)
                if cfg.q_min_threshold is not None and q < float(cfg.q_min_threshold):
                    continue
                raw_task_id = meta.get("task_id") or meta.get("sample_index") or meta.get("id")
                task_id = str(raw_task_id) if raw_task_id is not None else None
                candidates.append(
                    MemRLCandidate(
                        memory_id=rec.id,
                        content=rec.content,
                        metadata=meta,
                        similarity=similarity,
                        utility=q,
                        task_id=task_id,
                    )
                )

        if not candidates:
            return MemRLRetrievalResult(query=query)

        simmax = max(c.similarity for c in candidates)
        if cfg.unknown_threshold is not None and simmax < float(cfg.unknown_threshold):
            return MemRLRetrievalResult(
                query=query,
                candidates=sorted(candidates, key=lambda c: c.similarity, reverse=True),
                selected=[],
                simmax=simmax,
            )

        sim_values = [c.similarity for c in candidates]
        q_values = [c.utility for c in candidates]
        sim_mean = float(statistics.fmean(sim_values))
        q_mean = float(statistics.fmean(q_values))
        sim_std = safe_std(sim_values, population=True)
        q_std = safe_std(q_values, population=True)

        for cand in candidates:
            if cfg.use_zscore_normalization:
                cand.similarity_z = zscore(cand.similarity, sim_mean, sim_std, clamp=3.0)
                cand.utility_z = zscore(cand.utility, q_mean, q_std, clamp=3.0)
                sim_component = cand.similarity_z
                utility_component = cand.utility_z
            else:
                cand.similarity_z = cand.similarity
                cand.utility_z = cand.utility
                sim_component = cand.similarity
                utility_component = cand.utility

            cand.fused_score = (
                float(cfg.similarity_weight) * sim_component
                + float(cfg.utility_weight) * utility_component
            )

        ranked = sorted(
            candidates,
            key=lambda c: (c.fused_score, c.utility, c.similarity),
            reverse=True,
        )
        selected = self._select(ranked, topk=int(topk or cfg.topk))

        return MemRLRetrievalResult(
            query=query,
            candidates=ranked,
            selected=selected,
            simmax=simmax,
        )

    def update_from_outcome(
        self,
        retrieval: MemRLRetrievalResult,
        *,
        outcome: bool | float,
        next_max_q: Optional[float] = None,
    ) -> dict[str, Optional[float]]:
        """
        Update selected memories using one-step utility update.

        `outcome` can be:
        - bool: mapped to config success/failure reward
        - float: treated as direct scalar reward
        """
        reward = self._reward_from_outcome(outcome)
        return self.update_utilities(
            memory_ids=retrieval.selected_ids,
            reward=reward,
            next_max_q=next_max_q,
        )

    def update_utilities(
        self,
        *,
        memory_ids: list[str],
        reward: float,
        next_max_q: Optional[float] = None,
    ) -> dict[str, Optional[float]]:
        """Batch-update utility values for a list of memory ids."""
        out: dict[str, Optional[float]] = {}
        for memory_id in memory_ids:
            out[str(memory_id)] = self.update_utility(
                memory_id=str(memory_id),
                reward=float(reward),
                next_max_q=next_max_q,
            )
        return out

    def update_utility(
        self,
        *,
        memory_id: str,
        reward: float,
        next_max_q: Optional[float] = None,
    ) -> Optional[float]:
        """Apply MemRL utility update to one memory item."""
        cfg = self.config
        rec = self.store.get(memory_id)
        if rec is None:
            return None

        meta = dict(rec.metadata or {})
        old_q = _to_float(meta.get("q_value"), default=cfg.q_init_pos)
        target = float(reward) + float(cfg.gamma) * float(next_max_q or 0.0)
        new_q = (1.0 - float(cfg.alpha)) * old_q + float(cfg.alpha) * target

        if cfg.q_floor is not None:
            new_q = max(float(cfg.q_floor), float(new_q))

        visits = int(_to_float(meta.get("q_visits"), default=0.0)) + 1
        reward_ma_old = _to_float(meta.get("reward_ma"), default=0.0)
        reward_ma = (1.0 - float(cfg.alpha)) * reward_ma_old + float(cfg.alpha) * float(reward)

        meta["q_value"] = float(new_q)
        meta["q_visits"] = int(visits)
        meta["last_reward"] = float(reward)
        meta["reward_ma"] = float(reward_ma)
        meta["q_updated_at"] = _now_iso()
        meta["last_used_at"] = _now_iso()

        self.store.add(MemoryRecord(content=rec.content, metadata=meta, id=rec.id))
        return float(new_q)

    def _reward_from_outcome(self, outcome: bool | float) -> float:
        if isinstance(outcome, bool):
            return (
                float(self.config.success_reward)
                if outcome
                else float(self.config.failure_reward)
            )
        return float(outcome)

    def _select(self, ranked: list[MemRLCandidate], topk: int) -> list[MemRLCandidate]:
        n = min(max(int(topk), 0), len(ranked))
        if n <= 0:
            return []

        explore = self._rng.random() < float(self.config.epsilon)
        pool = list(ranked)

        if self.config.dedup_by_task_id:
            if explore:
                self._rng.shuffle(pool)

            selected: list[MemRLCandidate] = []
            seen: set[str] = set()
            for cand in pool:
                key = cand.task_id or f"__missing_task_id__:{cand.memory_id}"
                if key in seen:
                    continue
                seen.add(key)
                selected.append(cand)
                if len(selected) >= n:
                    break
            return selected

        if explore:
            return self._rng.sample(pool, n)
        return pool[:n]
