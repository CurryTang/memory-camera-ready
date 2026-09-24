"""C1-C9 baseline retrieval adapters for LoCoMo evaluation.

All adapters extend _BaselineLoCoMoAdapter and use the same QA prompt.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from agentmem.providers.openai_compat import OpenAICompatibleProvider
from agentmem.providers.base import Message
from agentmem.eval.resource_metrics import estimate_text_tokens
from agentmem.eval.locomo_runner.prompts import _build_locomo_answer_prompt
from agentmem.eval.locomo_runner.adapters.base import _BaselineLoCoMoAdapter

def _make_compress_provider(
    model: str,
    api_key: Optional[str],
    provider_kwargs: Optional[dict[str, Any]],
    base_url: Optional[str] = None,
) -> Any:
    """Create an OpenAICompatibleProvider for LLMCompressor with call cache."""
    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
    kwargs = dict(provider_kwargs or {})
    resolved_base_url = base_url or kwargs.pop("base_url", None)
    kwargs.pop("history_metadata", None)
    kwargs["history_metadata"] = {"role": "compressor", "model": model}
    return OpenAICompatibleProvider(
        api_key=resolved_key,
        model=model,
        base_url=resolved_base_url,
        **kwargs,
    )

class C1LoCoMoAdapter(_BaselineLoCoMoAdapter):
    """C1: PlainTextIndex + BM25Retriever."""

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        retrieval_topk: int = 10,
        provider_kwargs: Optional[dict[str, Any]] = None,
        index_cache: Optional[Any] = None,
    ) -> None:
        super().__init__(
            answer_model=answer_model, answer_api_key=answer_api_key,
            answer_base_url=answer_base_url, retrieval_topk=retrieval_topk,
            provider_kwargs=provider_kwargs, index_cache=index_cache,
        )
        from agentmem.retrieval.indexing.plain import PlainTextIndex
        from agentmem.retrieval.retrievers.bm25 import BM25Retriever
        self._index_builder = PlainTextIndex()
        self._retriever = BM25Retriever()

    def _index_cache_key(self) -> Optional[str]:
        return "PlainTextIndex"

    def _build_index(self, turns):
        return self._index_builder.build(turns)

    def _get_retriever(self):
        return self._retriever

class C2LoCoMoAdapter(_BaselineLoCoMoAdapter):
    """C2: PlainTextIndex + SubstringMatcher."""

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        retrieval_topk: int = 10,
        provider_kwargs: Optional[dict[str, Any]] = None,
        index_cache: Optional[Any] = None,
    ) -> None:
        super().__init__(
            answer_model=answer_model, answer_api_key=answer_api_key,
            answer_base_url=answer_base_url, retrieval_topk=retrieval_topk,
            provider_kwargs=provider_kwargs, index_cache=index_cache,
        )
        from agentmem.retrieval.indexing.plain import PlainTextIndex
        from agentmem.retrieval.retrievers.substring import SubstringMatcher
        self._index_builder = PlainTextIndex()
        self._retriever = SubstringMatcher()

    def _index_cache_key(self) -> Optional[str]:
        return "PlainTextIndex"

    def _build_index(self, turns):
        return self._index_builder.build(turns)

    def _get_retriever(self):
        return self._retriever

class C3LoCoMoAdapter(_BaselineLoCoMoAdapter):
    """C3: FixedSizeChunker + DenseRetriever (sentence-transformers)."""

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        retrieval_topk: int = 10,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        provider_kwargs: Optional[dict[str, Any]] = None,
        index_cache: Optional[Any] = None,
    ) -> None:
        super().__init__(
            answer_model=answer_model, answer_api_key=answer_api_key,
            answer_base_url=answer_base_url, retrieval_topk=retrieval_topk,
            provider_kwargs=provider_kwargs, index_cache=index_cache,
        )
        from agentmem.retrieval.indexing.chunker import FixedSizeChunker
        from agentmem.retrieval.retrievers.dense import DenseRetriever
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._index_builder = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self._retriever = DenseRetriever(embedding_model=embedding_model)

    def _index_cache_key(self) -> Optional[str]:
        return f"FixedSizeChunker_cs{self._chunk_size}_ol{self._chunk_overlap}"

    def _build_index(self, turns):
        return self._index_builder.build(turns)

    def _get_retriever(self):
        return self._retriever

class C4LoCoMoAdapter(_BaselineLoCoMoAdapter):
    """C4: LLMCompressor + DenseRetriever."""

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        retrieval_topk: int = 10,
        compress_model: str = "gpt-4o-mini",
        compress_api_key: Optional[str] = None,
        compress_base_url: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        provider_kwargs: Optional[dict[str, Any]] = None,
        index_cache: Optional[Any] = None,
    ) -> None:
        super().__init__(
            answer_model=answer_model, answer_api_key=answer_api_key,
            answer_base_url=answer_base_url, retrieval_topk=retrieval_topk,
            provider_kwargs=provider_kwargs, index_cache=index_cache,
        )
        from agentmem.retrieval.indexing.compressor import LLMCompressor
        from agentmem.retrieval.retrievers.dense import DenseRetriever
        self._compress_model = compress_model
        self._index_builder = LLMCompressor(
            provider=_make_compress_provider(
                compress_model,
                compress_api_key,
                provider_kwargs,
                base_url=compress_base_url or answer_base_url,
            ),
        )
        self._retriever = DenseRetriever(embedding_model=embedding_model)

    def _index_cache_key(self) -> Optional[str]:
        return f"LLMCompressor_model_{self._compress_model}"

    def _build_index(self, turns):
        return self._index_builder.build(turns)

    def _get_retriever(self):
        return self._retriever

class C5LoCoMoAdapter(_BaselineLoCoMoAdapter):
    """C5: LLMCompressor + ScoreFusion (dense + BM25, RRF)."""

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        retrieval_topk: int = 10,
        compress_model: str = "gpt-4o-mini",
        compress_api_key: Optional[str] = None,
        compress_base_url: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        provider_kwargs: Optional[dict[str, Any]] = None,
        index_cache: Optional[Any] = None,
    ) -> None:
        super().__init__(
            answer_model=answer_model, answer_api_key=answer_api_key,
            answer_base_url=answer_base_url, retrieval_topk=retrieval_topk,
            provider_kwargs=provider_kwargs, index_cache=index_cache,
        )
        from agentmem.retrieval.indexing.compressor import LLMCompressor
        from agentmem.retrieval.retrievers.dense import DenseRetriever
        from agentmem.retrieval.retrievers.bm25 import BM25Retriever
        from agentmem.retrieval.fusion.score_fusion import ScoreFusion
        self._compress_model = compress_model
        self._index_builder = LLMCompressor(
            provider=_make_compress_provider(
                compress_model,
                compress_api_key,
                provider_kwargs,
                base_url=compress_base_url or answer_base_url,
            ),
        )
        dense = DenseRetriever(embedding_model=embedding_model)
        bm25 = BM25Retriever()
        self._compositor = ScoreFusion(retrievers=[(dense, 0.5), (bm25, 0.5)], mode="rrf")

    def _index_cache_key(self) -> Optional[str]:
        return f"LLMCompressor_model_{self._compress_model}"

    def _build_index(self, turns):
        return self._index_builder.build(turns)

    def _get_retriever(self):
        raise NotImplementedError("C5 uses compositor; call ask() directly")

    def ask(self, question: str, category: Optional[int] = None) -> str:
        if self._index is None:
            self.finalize()
        results = self._compositor.compose(question, self._index, k=self._topk)
        self._last_trajectory = {
            "retrieved": [
                {"doc_id": r.doc_id, "score": round(r.score, 6), "chars": len(r.content)}
                for r in results
            ],
            "top_k": self._topk,
            "method": "score_fusion",
        }
        context = "\n\n".join(r.content for r in results) if results else ""
        prompt = _build_locomo_answer_prompt(question=question, context=context, category=category)
        msgs = [Message(role="user", content=prompt)]
        resp = self._answer_provider.chat(msgs)
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
        }
        if usage:
            self._last_resource_usage["usage"] = usage
        return resp.content or ""

class C6LoCoMoAdapter(_BaselineLoCoMoAdapter):
    """C6: CausalGraphBuilder + StateMemoryBuilder + LLMGraphReasoner.

    3-tier retrieval with A1/A2 adaptations for LoCoMo.
    """

    _DEFAULT_STATE_SUFFICIENCY_THRESHOLD = 0.05

    _REASONER_SYSTEM = (
        "You are a graph reasoning assistant. Given a list of facts/passages and a query, "
        "select the most relevant entry IDs that help answer the query.\n\n"
        "Each entry is formatted as: ID: CONTENT\n\n"
        "Output a JSON array of selected entry IDs (strings), most relevant first. "
        "Output only valid JSON, no commentary."
    )

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        retrieval_topk: int = 10,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        state_snapshot_every: int = 10,
        state_sufficiency_threshold: float = _DEFAULT_STATE_SUFFICIENCY_THRESHOLD,
        provider_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            answer_model=answer_model, answer_api_key=answer_api_key,
            answer_base_url=answer_base_url, retrieval_topk=retrieval_topk,
            provider_kwargs=provider_kwargs,
        )
        from agentmem.retrieval.indexing.causal_graph import CausalGraphBuilder
        from agentmem.retrieval.indexing.state_memory import StateMemoryBuilder
        self._causal_builder = CausalGraphBuilder(model=llm_model, api_key=llm_api_key)
        self._state_builder = StateMemoryBuilder(
            model=llm_model, api_key=llm_api_key, snapshot_every=state_snapshot_every
        )
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key
        self._state_index = None
        self._raw_turn_index = None
        self._state_sufficiency_threshold = state_sufficiency_threshold

    def _build_index(self, turns):
        from agentmem.retrieval.base import Index, IndexUnit
        self._state_index = self._state_builder.build(turns)
        causal_index = self._causal_builder.build(turns)
        raw_units = [
            IndexUnit(
                id=f"raw_{doc.id}", content=doc.content,
                metadata={"source": "raw_turn", "edge_type": "raw_turn", **doc.metadata},
                source_doc_id=doc.id,
            )
            for doc in turns
        ]
        self._raw_turn_index = Index(units=raw_units, metadata={"builder": "RawTurnIndex"})
        return causal_index

    def _get_retriever(self):
        return None

    def _llm_graph_reason_a2(self, question: str) -> tuple[str, bool]:
        import json as _json
        from agentmem.retrieval.retrievers.bm25 import BM25Retriever

        if not self._index or not self._index.units:
            return "", False

        unit_map = {u.id: u for u in self._index.units}
        if len(self._index.units) > 50:
            bm25 = BM25Retriever()
            pre = bm25.retrieve(question, self._index, k=50)
            candidate_units = [unit_map[r.doc_id] for r in pre if r.doc_id in unit_map]
        else:
            candidate_units = list(self._index.units)

        if not candidate_units:
            return "", False

        facts_text = "\n".join(f"{u.id}: {u.content[:250]}" for u in candidate_units[:50])

        try:
            from openai import OpenAI
            client = OpenAI(api_key=(self._llm_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"))
            resp = client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": self._REASONER_SYSTEM},
                    {"role": "user", "content": f"Query: {question}\n\nFacts:\n{facts_text}\n\nSelect top-{self._topk} relevant entry IDs:"},
                ],
                max_tokens=256, temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception:
            return "", False

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            selected_ids = _json.loads(raw)
        except Exception:
            return "", False

        if not isinstance(selected_ids, list):
            return "", False

        parts, used_a2 = [], False
        for doc_id in selected_ids[:self._topk]:
            unit = unit_map.get(str(doc_id))
            if unit is None:
                continue
            parts.append(unit.content.strip())
            if unit.metadata.get("edge_type") == "raw_turn":
                used_a2 = True

        return "\n\n---\n\n".join(parts), used_a2

    def ask(self, question: str, category: Optional[int] = None) -> str:
        if self._index is None:
            self.finalize()

        from agentmem.retrieval.retrievers.bm25 import BM25Retriever
        context = ""
        tier_used = None

        if self._state_index is not None and self._state_index.units:
            state_bm25 = BM25Retriever()
            state_results = state_bm25.retrieve(question, self._state_index, k=self._topk)
            if state_results and state_results[0].score >= self._state_sufficiency_threshold:
                context = "\n\n".join(r.content for r in state_results[:self._topk])
                tier_used = "state_memory"
                self._last_trajectory = {
                    "retrieved": [{"doc_id": r.doc_id, "score": round(r.score, 6), "chars": len(r.content)} for r in state_results[:self._topk]],
                    "tier": "state_memory", "top_k": self._topk,
                }

        if not context:
            context, _used_a2 = self._llm_graph_reason_a2(question)
            if context:
                tier_used = "causal_graph"
                self._last_trajectory = {"retrieved": [], "tier": "causal_graph", "context_chars": len(context), "top_k": self._topk}

        if not context and self._raw_turn_index and self._raw_turn_index.units:
            bm25 = BM25Retriever()
            results = bm25.retrieve(question, self._raw_turn_index, k=self._topk)
            if results:
                context = "\n\n---\n\n".join(r.content.strip() for r in results[:self._topk])
                tier_used = "raw_bm25"
                self._last_trajectory = {
                    "retrieved": [{"doc_id": r.doc_id, "score": round(r.score, 6), "chars": len(r.content)} for r in results[:self._topk]],
                    "tier": "raw_bm25", "top_k": self._topk,
                }

        if tier_used is None:
            self._last_trajectory = {"retrieved": [], "tier": "none", "top_k": self._topk}

        prompt = _build_locomo_answer_prompt(question=question, context=context, category=category)
        msgs = [Message(role="user", content=prompt)]
        resp = self._answer_provider.chat(msgs)
        return resp.content or ""

class C7LoCoMoAdapter(_BaselineLoCoMoAdapter):
    """C7: PropositionExtractor + ConceptGraphBuilder + MultiHopTraverser."""

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        retrieval_topk: int = 10,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        max_hops: int = 2,
        provider_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            answer_model=answer_model, answer_api_key=answer_api_key,
            answer_base_url=answer_base_url, retrieval_topk=retrieval_topk,
            provider_kwargs=provider_kwargs,
        )
        from agentmem.retrieval.indexing.proposition import PropositionExtractor
        from agentmem.retrieval.indexing.concept_graph import ConceptGraphBuilder
        from agentmem.retrieval.retrievers.multi_hop import MultiHopTraverser
        self._proposition_extractor = PropositionExtractor(model=llm_model, api_key=llm_api_key)
        self._concept_builder = ConceptGraphBuilder()
        self._retriever = MultiHopTraverser(max_hops=max_hops)

    def _build_index(self, turns):
        prop_index = self._proposition_extractor.build(turns)
        return self._concept_builder.build_from_index(prop_index)

    def _get_retriever(self):
        return self._retriever

class C8LoCoMoAdapter(_BaselineLoCoMoAdapter):
    """C8: KnowledgeGraphBuilder + PersonalizedPageRank (HippoRAG2-inspired)."""

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        retrieval_topk: int = 10,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        provider_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            answer_model=answer_model, answer_api_key=answer_api_key,
            answer_base_url=answer_base_url, retrieval_topk=retrieval_topk,
            provider_kwargs=provider_kwargs,
        )
        from agentmem.retrieval.indexing.knowledge_graph import KnowledgeGraphBuilder
        from agentmem.retrieval.retrievers.pagerank import PersonalizedPageRank
        self._index_builder = KnowledgeGraphBuilder(model=llm_model, api_key=llm_api_key)
        self._retriever = PersonalizedPageRank()

    def _build_index(self, turns):
        return self._index_builder.build(turns)

    def _get_retriever(self):
        return self._retriever

class C9LoCoMoAdapter(_BaselineLoCoMoAdapter):
    """C9: LLMCompressor + ColBERTRetriever."""

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        retrieval_topk: int = 10,
        compress_model: str = "gpt-4o-mini",
        compress_api_key: Optional[str] = None,
        compress_base_url: Optional[str] = None,
        model_name: str = "colbert-ir/colbertv2.0",
        provider_kwargs: Optional[dict[str, Any]] = None,
        index_cache: Optional[Any] = None,
    ) -> None:
        super().__init__(
            answer_model=answer_model, answer_api_key=answer_api_key,
            answer_base_url=answer_base_url, retrieval_topk=retrieval_topk,
            provider_kwargs=provider_kwargs, index_cache=index_cache,
        )
        from agentmem.retrieval.indexing.compressor import LLMCompressor
        from agentmem.retrieval.retrievers.colbert import ColBERTRetriever
        self._compress_model = compress_model
        self._index_builder = LLMCompressor(
            provider=_make_compress_provider(
                compress_model,
                compress_api_key,
                provider_kwargs,
                base_url=compress_base_url or answer_base_url,
            ),
        )
        self._retriever = ColBERTRetriever(model_name=model_name, max_query_length=32)

    def _index_cache_key(self) -> Optional[str]:
        return f"LLMCompressor_model_{self._compress_model}"

    def _build_index(self, turns):
        return self._index_builder.build(turns)

    def _get_retriever(self):
        return self._retriever
