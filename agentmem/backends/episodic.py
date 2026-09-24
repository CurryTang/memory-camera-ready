"""
Episodic memory store: ordered, temporal, append-only.

Preserves insertion order. Suitable for conversation history,
event streams, or any memory where recency and ordering matter.
Search is linear scan with basic string matching — override for richer retrieval.
"""

from __future__ import annotations

import math
import re
from typing import Optional

from agentmem.backends.base import BaseMemoryStore, MemoryRecord, SearchResult

try:
    from nltk.stem import PorterStemmer
except Exception:                                          
    PorterStemmer = None

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "would",
}

_STEMMER = PorterStemmer() if PorterStemmer is not None else None

def _tokenize(text: str, *, drop_stopwords: bool) -> list[str]:
    tokens = [tok for tok in re.findall(r"[a-z0-9_]+", str(text).lower()) if tok]
    out: list[str] = []
    for token in tokens:
        if drop_stopwords and token in _STOPWORDS:
            continue
        out.append(token)
        if _STEMMER is not None:
            stem = _STEMMER.stem(token)
            if stem and stem != token:
                out.append(stem)
        if len(token) > 3 and token.endswith("s"):
            singular = token[:-1]
            if singular and singular != token:
                out.append(singular)
    return out

class EpisodicMemoryStore(BaseMemoryStore):
    """
    In-memory ordered buffer. Records are stored in insertion order.

    Designed for dialogue history and event streams where temporal
    ordering is meaningful and the working set fits in memory.
    """

    def __init__(self, max_records: Optional[int] = None) -> None:
        """
        Args:
            max_records: If set, oldest records are evicted when the cap is exceeded.
        """
        self.max_records = max_records
        self._records: list[MemoryRecord] = []
        self._index: dict[str, int] = {}                              

    def add(self, record: MemoryRecord) -> str:
        if record.id in self._index:
            pos = self._index[record.id]
            self._records[pos] = record
            return record.id

        self._records.append(record)
        self._index[record.id] = len(self._records) - 1

        if self.max_records is not None and self.max_records > 0:
            while len(self._records) > self.max_records:
                oldest = self._records.pop(0)
                self._index.pop(oldest.id, None)
                self._rebuild_index()
        return record.id

    def get(self, id: str) -> Optional[MemoryRecord]:
        pos = self._index.get(id)
        if pos is None:
            return None
        if pos >= len(self._records):
            return None
        return self._records[pos]

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        q = query.strip().lower()
        if not q:
            return []
        q_terms = set(_tokenize(q, drop_stopwords=True))
        if not q_terms:
            q_terms = set(_tokenize(q, drop_stopwords=False))
        if not q_terms:
            return []

        doc_freq: dict[str, int] = {}
        tokenized_records: list[list[str]] = []
        for rec in self._records:
            tokens = _tokenize(rec.content, drop_stopwords=False)
            tokenized_records.append(tokens)
            for term in set(tokens):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        total_docs = max(len(self._records), 1)
        idf = {
            term: math.log((1.0 + total_docs) / (1.0 + freq)) + 1.0
            for term, freq in doc_freq.items()
        }

        scored: list[SearchResult] = []
        total = len(self._records)
        for idx, rec in enumerate(self._records):
            content = rec.content.lower()
            tokens = tokenized_records[idx]
            token_set = set(tokens)
            overlap_terms = q_terms & token_set
            overlap = len(overlap_terms)
            contains = q in content
            if overlap == 0 and not contains:
                continue

            overlap_weight = sum(idf.get(term, 1.0) for term in overlap_terms)
            coverage = overlap / max(len(q_terms), 1)
            recency_weight = (idx + 1) / max(total, 1)

            score = (
                float(overlap_weight)
                + (1.5 if contains else 0.0)
                + (0.75 * coverage)
                + (0.15 * recency_weight)
            )

            metadata = rec.metadata or {}
            speaker = str(metadata.get("speaker") or "").strip().lower()
            if speaker and speaker in q_terms:
                score += 0.5

            if str(metadata.get("source") or "") == "qa_feedback":
                score *= 0.35

            scored.append(SearchResult(record=rec, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: max(k, 0)]

    def delete(self, id: str) -> None:
        pos = self._index.get(id)
        if pos is None:
            return
        self._records.pop(pos)
        self._index.pop(id, None)
        self._rebuild_index()

    def clear(self) -> None:
        self._records = []
        self._index = {}

    def count(self) -> int:
        return len(self._records)

    def iter_chronological(self) -> list[MemoryRecord]:
        """Return all records in insertion order."""
        return list(self._records)

    def _rebuild_index(self) -> None:
        self._index = {rec.id: i for i, rec in enumerate(self._records)}
