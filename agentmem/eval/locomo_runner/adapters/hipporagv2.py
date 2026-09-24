"""HippoRAGv2 adapter using the official hipporag pip package."""

from __future__ import annotations

import os
import sys
import tempfile
import types
from typing import Any, Dict, Optional

from agentmem.providers.openai_compat import OpenAICompatibleProvider
from agentmem.providers.base import Message
from agentmem.eval.locomo_runner.prompts import _build_locomo_answer_prompt
from agentmem.eval.resource_metrics import estimate_text_tokens

class HippoRAGv2LoCoMoAdapter:
    """HippoRAGv2 adapter using the official hipporag pip package.

    Uses the same ``_build_locomo_answer_prompt`` as C1-C9 for consistency.
    Retrieval via HippoRAG's entity graph + PPR.

    Requires: ``pip install hipporag``
    """

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        retrieval_topk: int = 15,
        save_dir: Optional[str] = None,
        provider_kwargs: Optional[dict[str, Any]] = None,
        **_extra: Any,
    ) -> None:
        self._answer_provider = OpenAICompatibleProvider(
            api_key=(answer_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"),
            model=answer_model,
            base_url=answer_base_url,
            **dict(provider_kwargs or {}),
        )
        self._embedding_model = embedding_model
        self._embedding_base_url = embedding_base_url
        self._llm_model = llm_model or answer_model
        self._llm_base_url = llm_base_url or answer_base_url
        self._topk = max(1, retrieval_topk)
        self._save_dir = save_dir or os.path.join(tempfile.gettempdir(), "hipporag_locomo")
        self._turns: list[str] = []
        self._hipporag = None
        self._sample_counter = 0

    def reset(self) -> None:
        self._turns = []
        self._hipporag = None
        self._sample_counter += 1

    def observe(self, content: str, timestamp: Optional[str] = None) -> None:
        self._turns.append(content)

    def finalize(self) -> None:

        if "vllm" not in sys.modules:
            _vllm = types.ModuleType("vllm")
            _vllm.SamplingParams = type("SamplingParams", (), {})
            _vllm.LLM = type("LLM", (), {})
            sys.modules["vllm"] = _vllm

        from hipporag import HippoRAG

        try:
            import hipporag.embedding_model as _emb_mod
            from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
            _orig_get = _emb_mod._get_embedding_model_class

            def _patched_get(model_name=None, **kwargs):
                model_name = model_name or kwargs.get("embedding_model_name")
                try:
                    return _orig_get(model_name)
                except (AssertionError, KeyError):
                    return OpenAIEmbeddingModel
            _emb_mod._get_embedding_model_class = _patched_get

            for mod in list(sys.modules.values()):
                if hasattr(mod, '_get_embedding_model_class') and mod is not _emb_mod:
                    mod._get_embedding_model_class = _patched_get
            HippoRAG.__init__.__globals__['_get_embedding_model_class'] = _patched_get
        except Exception:
            pass

        sample_dir = os.path.join(
            self._save_dir, f"sample_{self._sample_counter:04d}"
        )
        kwargs: dict[str, Any] = {
            "save_dir": sample_dir,
            "llm_model_name": self._llm_model,
            "embedding_model_name": self._embedding_model,
        }
        if self._llm_base_url:
            kwargs["llm_base_url"] = self._llm_base_url
        if self._embedding_base_url:
            kwargs["embedding_base_url"] = self._embedding_base_url
        self._hipporag = HippoRAG(**kwargs)
        self._hipporag.index(docs=self._turns)

    def ask(self, question: str, category: Optional[int] = None) -> str:
        import time as _time
        if self._hipporag is None:
            self.finalize()

        retrieve_start = _time.perf_counter()
        query_solutions = self._hipporag.retrieve(queries=[question])
        sol = query_solutions[0]
        retrieved_docs = sol.docs[: self._topk] if sol.docs else []
        retrieve_seconds = _time.perf_counter() - retrieve_start

        context = "\n\n".join(
            doc.strip() for doc in retrieved_docs
        ) if retrieved_docs else ""

        self._last_trajectory = {
            "retrieved": [{"doc": d[:100], "chars": len(d)} for d in retrieved_docs],
            "top_k": self._topk,
        }

        prompt = _build_locomo_answer_prompt(
            question=question, context=context, category=category
        )
        msgs = [Message(role="user", content=prompt)]
        llm_start = _time.perf_counter()
        resp = self._answer_provider.chat(msgs)
        llm_seconds = _time.perf_counter() - llm_start
        usage = dict(getattr(resp, "usage", None) or {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
        self._last_resource_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)),
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

    def canonical_efficiency(self) -> dict:
        from agentmem.eval.locomo_runner.adapters.base import canonical_efficiency_from_usage
        return canonical_efficiency_from_usage(
            getattr(self, "_build_resource_usage", None),
            getattr(self, "_last_resource_usage", None),
        )

    def shutdown(self) -> None:
        self._turns = []
        self._hipporag = None
