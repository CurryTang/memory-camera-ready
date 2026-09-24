"""
MetadataFilter — filters retrieval results by metadata fields.

Used by C5 as a post-filter step.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from agentmem.retrieval.base import Index, RetrievalResult, Retriever

class MetadataFilter(Retriever):
    """
    Wraps another retriever and filters results by metadata constraints.

    Args:
        base_retriever: The retriever to wrap.
        filters: Dict of {metadata_key: expected_value} constraints.
            A result passes if all constraints match.
        filter_fn: Optional callable(metadata) -> bool for custom filtering.
    """

    def __init__(
        self,
        base_retriever: Any,
        filters: Optional[Dict[str, Any]] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> None:
        self._base = base_retriever
        self._filters = filters or {}
        self._filter_fn = filter_fn

    def _passes(self, result: RetrievalResult) -> bool:
        meta = result.metadata or {}
        for key, val in self._filters.items():
            if meta.get(key) != val:
                return False
        if self._filter_fn is not None:
            return self._filter_fn(meta)
        return True

    def retrieve(self, query: str, index: Index, k: int = 10) -> list[RetrievalResult]:

        candidates = self._base.retrieve(query, index, k=k * 3)
        filtered = [r for r in candidates if self._passes(r)]
        return filtered[:k]
