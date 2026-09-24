"""
QueryDecomposer — decomposes a complex query into sub-queries via LLM.

Used by C5. The LLM decomposes the query into semantic (paraphrase for dense
retrieval), lexical (keyword list for BM25), and optional symbolic (entity/
relation constraint) sub-queries, plus a depth hint.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from agentmem.retrieval.base import Compositor, Index, RetrievalResult

logger = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = (
    "You decompose search queries for a memory retrieval system. "
    "Given a question, output a JSON object with these keys:\n"
    "  semantic: a paraphrased question optimized for semantic/dense retrieval\n"
    "  lexical: a short list of key terms optimized for keyword/BM25 search\n"
    "  symbolic: (optional) entity or relation constraint, e.g. 'person:Alice date:2023'\n"
    "  depth: integer 1-3 for retrieval depth (1=simple, 2=moderate, 3=complex multi-hop)\n\n"
    "Output only valid JSON, no commentary."
)

@dataclass
class DecomposedQuery:
    """A decomposed query with semantic, lexical, and symbolic sub-queries."""
    semantic: str
    lexical: str
    symbolic: Optional[str] = None
    depth: int = 1

class QueryDecomposer(Compositor):
    """
    Decomposes a query into (q_sem, q_lex, q_sym, depth) components,
    then routes each to the appropriate retriever.

    Args:
        semantic_retriever: Dense retriever for semantic sub-query.
        lexical_retriever: BM25 retriever for lexical sub-query.
        fusion: Compositor to merge results.
        provider: OpenAICompatibleProvider instance.
    """

    def __init__(
        self,
        semantic_retriever: Optional[Any] = None,
        lexical_retriever: Optional[Any] = None,
        fusion: Optional[Any] = None,
        provider: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._semantic = semantic_retriever
        self._lexical = lexical_retriever
        self._fusion = fusion
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    def _get_provider(self) -> Any:
        if self._provider is not None:
            return self._provider
        import os
        from agentmem.providers.openai_compat import OpenAICompatibleProvider
        api_key = self._api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self._provider = OpenAICompatibleProvider(
            api_key=api_key,
            model=self._model,
            base_url=self._base_url,
        )
        return self._provider

    def _decompose(self, query: str) -> DecomposedQuery:
        """Call LLM to decompose query into (semantic, lexical, symbolic, depth).

        Falls back to DecomposedQuery(semantic=query, lexical=query) on any failure.
        """
        from agentmem.providers.base import Message
        try:
            provider = self._get_provider()
            messages = [
                Message(role="system", content=_DECOMPOSE_SYSTEM),
                Message(role="user", content=f"Query: {query}"),
            ]
            response = provider.chat(messages, max_tokens=200)
            raw = (response.content or "").strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            data = json.loads(raw)
            return DecomposedQuery(
                semantic=str(data.get("semantic") or query),
                lexical=str(data.get("lexical") or query),
                symbolic=data.get("symbolic") or None,
                depth=int(data.get("depth") or 1),
            )
        except Exception as exc:
            logger.debug("QueryDecomposer: LLM decomposition failed (%s); using original query.", exc)
            return DecomposedQuery(semantic=query, lexical=query)

    def compose(
        self,
        query: str,
        index: Index,
        k: int = 10,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        try:
            decomposed = self._decompose(query)
        except NotImplementedError:
            decomposed = DecomposedQuery(semantic=query, lexical=query)

        results: list[RetrievalResult] = []

        if self._semantic is not None:
            sem_results = self._semantic.retrieve(decomposed.semantic, index, k=k)
            results.extend(sem_results)

        if self._lexical is not None:
            lex_results = self._lexical.retrieve(decomposed.lexical, index, k=k)
            results.extend(lex_results)

        if self._fusion is not None:
            return self._fusion.compose(query, index, k=k)

        best: dict[str, RetrievalResult] = {}
        for r in results:
            if r.doc_id not in best or r.score > best[r.doc_id].score:
                best[r.doc_id] = r

        ranked = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return ranked[:k]
