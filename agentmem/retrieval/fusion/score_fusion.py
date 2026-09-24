"""
ScoreFusion — combines results from multiple retrievers by score.

Supports weighted linear combination and Reciprocal Rank Fusion (RRF).
Used as an optional helper for C3 hybrid variant and C5.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

from agentmem.retrieval.base import Compositor, Index, RetrievalResult

class ScoreFusion(Compositor):
    """
    Merge and re-rank results from multiple retrievers.

    Modes:
    - "linear": weighted sum of normalized scores.
    - "rrf": Reciprocal Rank Fusion (score = sum of 1/(k+rank)).

    Args:
        retrievers: List of (retriever, weight) pairs.
        mode: Fusion mode — "linear" or "rrf".
        rrf_k: RRF smoothing constant (default 60).
        alpha: For 2-retriever linear fusion: score = alpha * r1 + (1-alpha) * r2.
               Ignored when explicit weights are provided.
    """

    def __init__(
        self,
        retrievers: list,
        mode: str = "linear",
        rrf_k: int = 60,
        alpha: float = 0.5,
    ) -> None:
        self._retrievers = retrievers                                                 
        self._mode = mode
        self._rrf_k = rrf_k
        self._alpha = alpha

    def _normalize_scores(self, results: list[RetrievalResult]) -> dict[str, float]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return {}
        scores = [r.score for r in results]
        lo, hi = min(scores), max(scores)
        span = hi - lo if hi != lo else 1.0
        return {r.doc_id: (r.score - lo) / span for r in results}

    def compose(
        self,
        query: str,
        index: Index,
        k: int = 10,
        **kwargs,
    ) -> list[RetrievalResult]:
        retriever_results: list[tuple[list[RetrievalResult], float]] = []
        for item in self._retrievers:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                retriever, weight = item
            else:
                retriever, weight = item, 1.0
            res = retriever.retrieve(query, index, k=k * 2)
            retriever_results.append((res, weight))

        doc_info: dict[str, tuple[str, dict]] = {}
        for results, _ in retriever_results:
            for r in results:
                if r.doc_id not in doc_info:
                    doc_info[r.doc_id] = (r.content, r.metadata)

        fused_scores: dict[str, float] = defaultdict(float)

        if self._mode == "rrf":
            for results, weight in retriever_results:
                for rank, r in enumerate(results):
                    fused_scores[r.doc_id] += weight / (self._rrf_k + rank + 1)
        else:          
            for results, weight in retriever_results:
                norm = self._normalize_scores(results)
                for doc_id, score in norm.items():
                    fused_scores[doc_id] += weight * score

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        output: list[RetrievalResult] = []
        for doc_id, score in ranked[:k]:
            content, metadata = doc_info.get(doc_id, ("", {}))
            output.append(
                RetrievalResult(
                    doc_id=doc_id,
                    content=content,
                    score=score,
                    metadata=metadata,
                )
            )
        return output
