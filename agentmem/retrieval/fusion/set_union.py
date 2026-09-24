"""
SetUnionFusion — merges results from multiple retrievers via set union + dedup.

Used by C5 (multi-view parallel retrieval).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from agentmem.retrieval.base import Compositor, Index, RetrievalResult

class SetUnionFusion(Compositor):
    """
    Merges results from multiple retrievers, deduplicates by doc_id,
    and assigns a final score as the max score across all retrievers.

    Args:
        retrievers: List of retriever instances (or (retriever, weight) pairs).
    """

    def __init__(self, retrievers: list) -> None:
        self._retrievers = retrievers

    def compose(
        self,
        query: str,
        index: Index,
        k: int = 10,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        best: dict[str, RetrievalResult] = {}

        for item in self._retrievers:
            retriever = item[0] if isinstance(item, (list, tuple)) else item
            results = retriever.retrieve(query, index, k=k * 2)
            for r in results:
                if r.doc_id not in best or r.score > best[r.doc_id].score:
                    best[r.doc_id] = r

        ranked = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return ranked[:k]
