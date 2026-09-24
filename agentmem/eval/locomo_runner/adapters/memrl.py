"""Repo-owned MemRL adapter for LoCoMo evaluation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from agentmem.backends.episodic import EpisodicMemoryStore
from agentmem.eval.locomo_runner.adapters.base import _BaselineLoCoMoAdapter
from agentmem.eval.locomo_runner.io_utils import _dump_memrl_store_records
from agentmem.eval.locomo_runner.prompts import _build_locomo_answer_prompt
from agentmem.eval.resource_metrics import estimate_trajectory_tokens, estimate_text_tokens
from agentmem.providers.base import Message
from agentmem.memrl.runtime import MemRLConfig, MemRLRuntimeEngine
from agentmem.memrl.task_adaptation import (
    build_locomo_memrl_query,
    build_locomo_qa_memory,
    compute_memrl_reward,
    format_locomo_memory_context,
)

class MemRLLoCoMoAdapter(_BaselineLoCoMoAdapter):
    """LoCoMo adapter backed by the repo-owned MemRL runtime.

    The adapter preserves:
    - per-question retrieval trajectories,
    - token usage for the answer-generation LLM call,
    - sample-level build/runtime information, and
    - a store dump artifact per sample.
    """

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,

        retrieval_topk: int = 10,
        phase1_topk: int = 50,
        alpha: float = 0.3,
        gamma: float = 0.0,
        epsilon: float = 0.05,
        similarity_weight: float = 0.7,
        utility_weight: float = 0.3,
        use_zscore_normalization: bool = True,
        max_tokens: Optional[int] = None,
        artifact_root: Optional[str | Path] = None,
        task: str = "memrl-locomo",
        token_counter: Optional[Any] = None,
        provider_kwargs: Optional[dict[str, Any]] = None,
        **_extra: Any,
    ) -> None:
        super().__init__(
            answer_model=answer_model,
            answer_api_key=answer_api_key,
            answer_base_url=answer_base_url,
            retrieval_topk=retrieval_topk,
            provider_kwargs=provider_kwargs,
        )
        self._cfg = MemRLConfig(
            phase1_topk=max(1, int(phase1_topk)),
            topk=max(1, int(retrieval_topk)),
            alpha=float(alpha),
            gamma=float(gamma),
            epsilon=float(epsilon),
            similarity_weight=float(similarity_weight),
            utility_weight=float(utility_weight),
            use_zscore_normalization=bool(use_zscore_normalization),
        )
        self._max_tokens = max_tokens
        self._artifact_root = Path(artifact_root or Path("results") / "locomo_artifacts")
        self._task = task
        self._token_counter = token_counter
        self._sample_id: Optional[str] = None
        self._sample_dir: Optional[Path] = None
        self._store: Optional[EpisodicMemoryStore] = None
        self._runtime: Optional[MemRLRuntimeEngine] = None
        self._sample_artifacts: list[Path] = []
        self._build_resource_usage: dict[str, Any] = {}
        self._last_resource_usage: dict[str, Any] = {}
        self._question_index = 0
        self._last_retrieval = None
        self._last_selected: list[Any] = []

    def reset(self, sample_id: Optional[str] = None) -> None:
        super().reset()
        self._sample_id = sample_id
        self._sample_dir = self._make_sample_dir(sample_id)
        self._sample_artifacts = []
        self._build_resource_usage = {}
        self._last_resource_usage = {}
        self._question_index = 0
        self._last_retrieval = None
        self._last_selected = []
        self._store = EpisodicMemoryStore()
        self._runtime = MemRLRuntimeEngine(store=self._store, config=self._cfg)

    def finalize(self, sample_id: Optional[str] = None) -> None:
        if sample_id is not None:
            self._sample_id = sample_id
        if self._runtime is None or self._store is None:
            self.reset(sample_id=self._sample_id)
        assert self._runtime is not None
        assert self._store is not None

        start = time.perf_counter()
        turns_payload: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(self._turns):
            content = str(getattr(turn, "content", turn))
            metadata = dict(getattr(turn, "metadata", {}) or {})
            turns_payload.append({"turn_idx": turn_index, "content": content})
            self._runtime.add_experience(
                intent=content,
                experience=content,
                success=True,
                metadata={
                    "source": "dialogue",
                    "sample_id": self._sample_id,
                    "turn_index": turn_index,
                    **metadata,
                },
                task_id=f"{self._sample_id or 'sample'}_turn_{turn_index}",
            )

        build_seconds = time.perf_counter() - start
        build_input_tokens = estimate_trajectory_tokens(
            turns_payload,
            token_counter=self._token_counter,
        )
        store_path = self._sample_dir_path / "memrl_store.jsonl"
        _dump_memrl_store_records(self._store, store_path)
        self._sample_artifacts.append(store_path)
        self._build_resource_usage = {
            "construction_input_tokens": int(build_input_tokens),
            "build_input_tokens": int(build_input_tokens),
            "build_output_tokens": 0,
            "build_wallclock_seconds": float(build_seconds),
            "persistent_store_bytes": int(store_path.stat().st_size if store_path.exists() else 0),
            "sample_id": self._sample_id,
            "num_turns": len(self._turns),
        }

    def ask(self, question: str, category: Optional[int] = None) -> str:
        if self._runtime is None:
            self.finalize(sample_id=self._sample_id)
        assert self._runtime is not None

        retrieve_start = time.perf_counter()
        retrieval_query = build_locomo_memrl_query(question, category)
        retrieval = self._runtime.retrieve(
            retrieval_query,
            phase1_topk=self._cfg.phase1_topk,
            topk=self._cfg.topk,
        )
        retrieve_seconds = time.perf_counter() - retrieve_start

        selected = retrieval.selected or retrieval.candidates[: self._cfg.topk]
        self._last_selected = list(selected)
        context = format_locomo_memory_context(selected)

        prompt = _build_locomo_answer_prompt(
            question=question,
            context=context,
            category=category,
        )
        llm_start = time.perf_counter()
        resp = self._answer_provider.chat(
            [Message(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        llm_seconds = time.perf_counter() - llm_start
        prediction = resp.content or ""
        usage = dict(resp.usage or {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens))

        self._last_trajectory = {
            "question": question,
            "retrieval_query": retrieval_query,
            "retrieved": [
                {
                    "memory_id": cand.memory_id,
                    "score": round(float(cand.fused_score), 6),
                    "similarity": round(float(cand.similarity), 6),
                    "utility": round(float(cand.utility), 6),
                    "content_chars": len(cand.content),
                }
                for cand in selected
            ],
            "candidate_count": len(retrieval.candidates),
            "selected_count": len(selected),
            "selected_ids": retrieval.selected_ids,
            "retrieved_context": context,
        }
        self._last_retrieval = retrieval
        self._last_resource_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "retrieved_context_tokens": estimate_text_tokens(
                context,
                token_counter=self._token_counter,
            ),
            "retrieval_calls": 1,
            "llm_calls": 1,
            "retrieve_seconds": float(retrieve_seconds),
            "llm_seconds": float(llm_seconds),
            "latency_seconds": float(retrieve_seconds + llm_seconds),
        }

        self._write_question_artifact(question=question, prediction=prediction, usage=usage)
        self._question_index += 1
        return prediction

    def observe_outcome(
        self,
        *,
        question: str,
        prediction: str,
        gold: Optional[str],
        category: Optional[int],
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._runtime is None or self._last_retrieval is None or gold is None:
            return
        reward = compute_memrl_reward(
            prediction,
            gold,
            scheme="token-f1",
        )
        self._runtime.update_utilities(
            memory_ids=[str(cand.memory_id) for cand in self._last_selected],
            reward=reward,
        )
        intent, experience, metadata = build_locomo_qa_memory(
            question=question,
            prediction=prediction,
            reference=gold,
            reward=reward,
            category=category,
            selected=self._last_selected,
        )
        if metrics:
            metadata["metrics"] = dict(metrics)
        self._runtime.add_experience(
            intent=intent,
            experience=experience,
            success=bool(reward >= 0.999),
            metadata=metadata,
            retrieved_memory_ids=self._last_retrieval.selected_ids,
            task_id=f"{self._sample_id or 'sample'}_qa_{self._question_index - 1}",
        )

    def sample_resource_usage(self) -> dict[str, Any]:
        return dict(self._build_resource_usage)

    def last_resource_usage(self) -> Optional[dict[str, Any]]:
        return dict(self._last_resource_usage) if self._last_resource_usage else None

    def sample_artifacts(self) -> list[Path]:
        return list(self._sample_artifacts)

    def shutdown(self) -> None:
        super().shutdown()
        self._store = None
        self._runtime = None
        self._sample_id = None
        self._sample_dir = None
        self._sample_artifacts = []
        self._build_resource_usage = {}
        self._last_resource_usage = {}
        self._question_index = 0
        self._last_retrieval = None
        self._last_selected = []

    @property
    def _sample_dir_path(self) -> Path:
        sample_id = self._sample_id or "sample_0000"
        path = self._artifact_root / self._task / sample_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _make_sample_dir(self, sample_id: Optional[str]) -> Path:
        path = self._artifact_root / self._task / (sample_id or "sample_0000")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_question_artifact(self, *, question: str, prediction: str, usage: dict[str, Any]) -> None:
        artifact = self._sample_dir_path / f"question_{self._question_index:03d}.json"
        payload = {
            "sample_id": self._sample_id,
            "question_index": self._question_index,
            "question": question,
            "prediction": prediction,
            "trajectory": self.last_trajectory(),
            "resource_usage": dict(self._last_resource_usage),
            "usage": usage,
        }
        artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._sample_artifacts.append(artifact)
