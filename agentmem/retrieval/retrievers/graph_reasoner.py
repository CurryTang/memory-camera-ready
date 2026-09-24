"""
LLMGraphReasoner — traverses a causal graph via LLM reasoning.

Used by C6. Returns empty list with a warning when LLM is unavailable.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agentmem.retrieval.base import Index, RetrievalResult, Retriever

logger = logging.getLogger(__name__)

class LLMGraphReasoner(Retriever):
    """
    Uses LLM to reason over causal graph edges to answer a query.

    Given a causal graph index (from CausalGraphBuilder), identifies relevant
    edges via LLM-guided traversal. Returns empty list with a warning when
    the LLM is unavailable (does NOT silently fall back to BM25).

    Args:
        provider: OpenAICompatibleProvider instance.
        model: LLM model name.
        api_key: API key.
        base_url: Base URL override.
        max_hops: Maximum reasoning hops (currently single-hop with LLM selection).
    """

    _REASON_SYSTEM = (
        "You are a graph reasoning assistant. Given a list of causal facts and a query, "
        "select the most relevant fact IDs that help answer the query.\n\n"
        "Each fact is formatted as: ID: CAUSE → EFFECT\n\n"
        "Output a JSON array of selected fact IDs (strings), most relevant first. "
        "Output only valid JSON, no commentary."
    )

    def __init__(
        self,
        provider: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_hops: int = 3,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_hops = max_hops

    def _get_provider(self):
        if self._provider is not None:
            return self._provider
        import os
        api_key = self._api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        from agentmem.providers.openai_compat import OpenAICompatibleProvider
        self._provider = OpenAICompatibleProvider(
            api_key=api_key,
            model=self._model,
            base_url=self._base_url,
        )
        return self._provider

    def retrieve(self, query: str, index: Index, k: int = 10) -> list[RetrievalResult]:
        """Retrieve causal graph edges relevant to the query via LLM reasoning.

        Attempts LLM-guided selection of causal edges. On LLM failure or
        unavailability, logs a warning and returns an empty list.
        Does NOT silently fall back to BM25 — callers must handle empty results.

        Args:
            query: The question or search query.
            index: Causal graph Index (from CausalGraphBuilder).
            k: Number of results to return.

        Returns:
            List of RetrievalResult objects ranked by relevance, or [] on LLM failure.
        """
        import json
        from agentmem.providers.base import Message

        if not index.units:
            return []

        unit_map = {u.id: u for u in index.units}

        causal_units = [u for u in index.units if u.metadata.get("edge_type") == "causal"]
        if not causal_units:

            logger.warning(
                "LLMGraphReasoner: no causal-edge units in index (all units have edge_type != 'causal'); "
                "returning empty results. C6 results for this query will be empty. "
                "Ensure CausalGraphBuilder successfully extracted edges."
            )
            return []

        if len(causal_units) > 50:
            from agentmem.retrieval.retrievers.bm25 import BM25Retriever
            pre_results = BM25Retriever().retrieve(query, index, k=50)
            causal_units = [unit_map[r.doc_id] for r in pre_results if r.doc_id in unit_map]

        facts_text = "\n".join(
            f"{u.id}: {u.content}" for u in causal_units[:50]
        )

        try:
            provider = self._get_provider()
        except Exception as exc:
            logger.warning(
                "LLMGraphReasoner: could not initialize LLM provider (%s); returning empty results. "
                "C6 results for this query will be empty.",
                exc,
            )
            return []

        messages = [
            Message(role="system", content=self._REASON_SYSTEM),
            Message(role="user", content=f"Query: {query}\n\nFacts:\n{facts_text}\n\nSelect top-{k} relevant fact IDs:"),
        ]
        try:
            response = provider.chat(messages, max_tokens=256)
            raw = (response.content or "").strip()
        except Exception as exc:
            logger.warning(
                "LLMGraphReasoner: LLM call failed (%s); returning empty results. "
                "C6 results for this query will be empty.",
                exc,
            )
            return []

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            selected_ids = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "LLMGraphReasoner: could not parse LLM JSON response (%s); returning empty results.",
                exc,
            )
            return []

        if not isinstance(selected_ids, list):
            logger.warning(
                "LLMGraphReasoner: LLM returned non-list response; returning empty results."
            )
            return []

        results: list[RetrievalResult] = []
        for rank, doc_id in enumerate(selected_ids[:k]):
            unit = unit_map.get(str(doc_id))
            if unit is None:
                continue
            results.append(RetrievalResult(
                doc_id=unit.id,
                content=unit.content,
                score=1.0 / (rank + 1),
                metadata=dict(unit.metadata or {}),
            ))

        return results
