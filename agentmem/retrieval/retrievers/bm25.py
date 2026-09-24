"""
BM25Retriever — wraps rank_bm25.BM25Okapi for sparse lexical retrieval.

Used by C1 and as a sub-retriever in C5.
"""
from __future__ import annotations

import re
from typing import Optional

from agentmem.retrieval.base import Index, RetrievalResult, Retriever

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being",
        "by", "did", "do", "does", "doing", "for", "from", "had",
        "has", "have", "having", "he", "her", "him", "his", "i",
        "if", "in", "into", "is", "it", "its", "itself", "just",
        "me", "my", "of", "on", "or", "our", "she", "so", "than",
        "that", "the", "their", "them", "then", "there", "these",
        "they", "this", "those", "to", "up", "us", "was", "we",
        "were", "what", "when", "where", "which", "who", "will",
        "with", "would", "you", "your",
    }
)

def _tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens or [""]

class BM25Retriever(Retriever):
    """
    BM25 retriever using rank_bm25.BM25Okapi.

    Builds a BM25 index from Index.units at retrieval time (lazy build),
    caching the index when the same Index object is passed again.
    """

    def __init__(self, remove_stopwords: bool = True) -> None:
        self._remove_stopwords = remove_stopwords
        self._cached_index_id: Optional[int] = None
        self._cached_bm25: Optional[object] = None
        self._cached_units: list = []

    def _build_bm25(self, index: Index):
        try:
            from rank_bm25 import BM25Okapi                
        except ImportError as exc:
            raise RuntimeError(
                "BM25Retriever requires rank-bm25. Install: pip install rank-bm25"
            ) from exc

        corpus = [_tokenize(u.content, self._remove_stopwords) for u in index.units]
        return BM25Okapi(corpus)

    def retrieve(self, query: str, index: Index, k: int = 10) -> list[RetrievalResult]:
        if not index.units:
            return []

        index_obj_id = id(index)
        if self._cached_index_id != index_obj_id:
            self._cached_bm25 = self._build_bm25(index)
            self._cached_index_id = index_obj_id
            self._cached_units = list(index.units)

        q_tokens = _tokenize(query, self._remove_stopwords)
        scores = self._cached_bm25.get_scores(q_tokens)                              

        ranked = sorted(
            zip(self._cached_units, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[RetrievalResult] = []
        for unit, score in ranked[:k]:
            results.append(
                RetrievalResult(
                    doc_id=unit.id,
                    content=unit.content,
                    score=float(score),
                    metadata=dict(unit.metadata or {}),
                )
            )
        return results
