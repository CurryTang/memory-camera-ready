"""Base adapter class for C1-C9 LoCoMo retrieval baselines."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Mapping, Optional

def canonical_efficiency_from_usage(
    build_usage: Optional[Mapping[str, Any]],
    query_usage: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Map adapter-local resource-usage dicts to the canonical 7-metric schema.

    Every per-question row carries: ``t_build_input``, ``t_build_output``,
    ``t_query_input``, ``t_query_output``, ``w_llm``, ``w_tool``,
    ``gpu_util_mean``. ``w_llm_build`` is also exported when the
    adapter records it. Missing values default to 0; ``gpu_util_mean`` stays
    ``None`` unless a sampler is attached.
    """
    build = dict(build_usage or {})
    query = dict(query_usage or {})
    return {
        "t_build_input": int(
            build.get("build_input_tokens",
                      build.get("construction_input_tokens", 0)) or 0
        ),
        "t_build_output": int(build.get("build_output_tokens", 0) or 0),
        "t_query_input": int(query.get("prompt_tokens", 0) or 0),
        "t_query_output": int(query.get("completion_tokens", 0) or 0),

        "w_llm": float(query.get("llm_seconds", 0.0) or 0.0),
        "w_tool": float(query.get("retrieve_seconds", 0.0) or 0.0),
        "w_llm_build": float(build.get("build_wallclock_seconds", 0.0) or 0.0),
        "gpu_util_mean": query.get("gpu_util_mean", build.get("gpu_util_mean")),
    }

from agentmem.providers.openai_compat import OpenAICompatibleProvider
from agentmem.providers.base import Message
from agentmem.eval.resource_metrics import estimate_text_tokens
from agentmem.eval.locomo_runner.prompts import _build_locomo_answer_prompt

class _BaselineLoCoMoAdapter:
    """Base class for C1-C9 LoCoMo adapters.

    Subclasses provide _build_index() and _get_retriever().
    All adapters share the same observe/finalize/ask/reset pattern.

    Index caching: subclasses that share an IndexBuilder with another config
    should override ``_index_cache_key()`` and pass ``index_cache``.
    """

    def __init__(
        self,
        answer_model: str,
        answer_api_key: Optional[str],
        answer_base_url: Optional[str],
        retrieval_topk: int,
        provider_kwargs: Optional[dict[str, Any]] = None,
        index_cache: Optional[Any] = None,
    ) -> None:
        self._answer_provider = OpenAICompatibleProvider(
            api_key=(answer_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"),
            model=answer_model,
            base_url=answer_base_url,
            **dict(provider_kwargs or {}),
        )
        self._topk = max(1, retrieval_topk)
        self._turns: list[Any] = []
        self._index_cache = index_cache
        self._build_resource_usage: dict[str, Any] = {}
        self._last_resource_usage: dict[str, Any] = {}
        self._last_trajectory: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self._turns = []
        self._index = None
        self._build_resource_usage = {}
        self._last_resource_usage = {}
        self._last_trajectory = None

    def observe(self, content: str, timestamp: Optional[str] = None) -> None:
        from agentmem.retrieval.base import Document
        turn_id = f"t{len(self._turns)}"
        meta: dict[str, Any] = {}
        if timestamp:
            meta["timestamp"] = timestamp
        self._turns.append(Document(id=turn_id, content=content, metadata=meta))

    def _index_cache_key(self) -> Optional[str]:
        return None

    def finalize(self) -> None:
        cache_key_prefix = self._index_cache_key()
        if self._index_cache is not None and cache_key_prefix is not None:
            full_key = self._index_cache.make_key(
                builder_name=cache_key_prefix,
                params={},
                documents=self._turns,
            )
            self._index = self._index_cache.get_or_build(
                full_key, lambda: self._build_index(self._turns)
            )
        else:
            self._index = self._build_index(self._turns)
        self._build_resource_usage = {
            "construction_input_tokens": estimate_text_tokens(
                "\n".join(str(getattr(turn, "content", "")) for turn in self._turns),
                model=self._answer_provider.model,
            ),
            "build_input_tokens": 0,
            "build_output_tokens": 0,
            "llm_calls": 0,
        }

    def _build_index(self, turns: list[Any]) -> Any:
        raise NotImplementedError

    def _get_retriever(self) -> Any:
        raise NotImplementedError

    def ask(self, question: str, category: Optional[int] = None) -> str:
        if self._index is None:
            self.finalize()
        retrieve_start = time.perf_counter()
        results = self._get_retriever().retrieve(question, self._index, k=self._topk)
        retrieve_seconds = time.perf_counter() - retrieve_start
        self._last_trajectory = {
            "retrieved": [
                {"doc_id": r.doc_id, "score": round(r.score, 6), "chars": len(r.content)}
                for r in results
            ],
            "top_k": self._topk,
            "num_candidates": len(self._index.units) if self._index else 0,
        }
        context = "\n\n".join(r.content for r in results) if results else ""
        prompt = _build_locomo_answer_prompt(question=question, context=context, category=category)
        msgs = [Message(role="user", content=prompt)]
        llm_start = time.perf_counter()
        resp = self._answer_provider.chat(msgs)
        llm_seconds = time.perf_counter() - llm_start
        usage = dict(resp.usage or {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens))
        self._last_resource_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "retrieved_context_tokens": estimate_text_tokens(context, model=self._answer_provider.model),
            "retrieval_calls": 1,
            "llm_calls": 1,
            "retrieve_seconds": float(retrieve_seconds),
            "llm_seconds": float(llm_seconds),
            "latency_seconds": float(retrieve_seconds + llm_seconds),
        }
        if usage:
            self._last_resource_usage["usage"] = usage
        return resp.content or ""

    def last_trajectory(self) -> Optional[Dict[str, Any]]:
        return getattr(self, "_last_trajectory", None)

    def sample_resource_usage(self) -> dict[str, Any]:
        return dict(self._build_resource_usage)

    def last_resource_usage(self) -> Optional[dict[str, Any]]:
        return dict(self._last_resource_usage) if self._last_resource_usage else None

    def shutdown(self) -> None:
        self._turns = []
        self._index = None
        self._build_resource_usage = {}
        self._last_resource_usage = {}
        self._last_trajectory = None

    def canonical_efficiency(self) -> dict[str, Any]:
        return canonical_efficiency_from_usage(
            self._build_resource_usage, self._last_resource_usage
        )
