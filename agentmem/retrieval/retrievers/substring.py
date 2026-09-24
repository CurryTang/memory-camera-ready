"""
SubstringMatcher — case-insensitive keyword substring retrieval.

Used by C2. Scores turns by keyword coverage and exact phrase bonus.
"""
from __future__ import annotations

import re
from typing import Optional

from agentmem.retrieval.base import Index, RetrievalResult, Retriever

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by",
        "did", "do", "does", "for", "from", "had", "has", "have",
        "he", "her", "him", "his", "i", "if", "in", "is", "it",
        "its", "just", "me", "my", "of", "on", "or", "she", "so",
        "than", "that", "the", "their", "them", "then", "there",
        "they", "this", "those", "to", "us", "was", "we", "were",
        "what", "when", "where", "which", "who", "will", "with",
        "would", "you", "your",
    }
)

def _extract_keywords(query: str) -> list[str]:
    """Extract content words from query (remove stopwords, lowercase)."""
    tokens = re.findall(r"[A-Za-z0-9]+", (query or "").lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]

def _score_unit(unit_lower: str, keywords: list[str], query_lower: str) -> float:
    """
    Score a unit by:
    - keyword hit count (normalized by num keywords)
    - coverage ratio (fraction of distinct keywords that appear)
    - exact phrase bonus (if normalized query phrase is a substring)
    """
    if not keywords:
        return 0.0

    hits = sum(1 for kw in keywords if kw in unit_lower)
    coverage = hits / len(keywords)

    phrase = " ".join(keywords)
    exact_bonus = 0.3 if phrase in unit_lower else 0.0

    return coverage + exact_bonus

class SubstringMatcher(Retriever):
    """
    Retrieves index units via keyword substring match.

    All scoring is done in-memory; suitable for turn-level plain text indices.
    """

    def retrieve(self, query: str, index: Index, k: int = 10) -> list[RetrievalResult]:
        if not index.units:
            return []

        keywords = _extract_keywords(query)
        if not keywords:

            return [
                RetrievalResult(
                    doc_id=u.id,
                    content=u.content,
                    score=0.0,
                    metadata=dict(u.metadata or {}),
                )
                for u in index.units[:k]
            ]

        query_lower = " ".join(keywords)
        scored: list[tuple[object, float]] = []
        for unit in index.units:
            unit_lower = (unit.content or "").lower()
            score = _score_unit(unit_lower, keywords, query_lower)
            scored.append((unit, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[RetrievalResult] = []
        for unit, score in scored[:k]:
            results.append(
                RetrievalResult(
                    doc_id=unit.id,                              
                    content=unit.content,                              
                    score=score,
                    metadata=dict(unit.metadata or {}),                              
                )
            )
        return results
