"""
Key-value memory store.

Stores records keyed by explicit string keys (not uuid).
Useful for named facts, structured knowledge, or agent working memory
where keys are meaningful (e.g. "user_preference_food", "task_status").
"""

from __future__ import annotations

from typing import Optional

from agentmem.backends.base import BaseMemoryStore, MemoryRecord, SearchResult

class KVMemoryStore(BaseMemoryStore):
    """
    Flat string-keyed in-memory store.

    Retrieval is exact-match by key or linear scan over values.
    """

    def __init__(self) -> None:
        self._store: dict[str, MemoryRecord] = {}

    def add(self, record: MemoryRecord) -> str:

        self._store[record.id] = record
        return record.id

    def get(self, id: str) -> Optional[MemoryRecord]:
        return self._store.get(id)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        q = query.strip().lower()
        if not q:
            return []

        results: list[SearchResult] = []
        q_terms = set(q.split())
        for key, rec in self._store.items():
            key_l = key.lower()
            content_l = rec.content.lower()
            if q == key_l:
                score = 3.0
            else:
                overlap = len(q_terms & set(content_l.split()))
                if overlap == 0 and q not in content_l and q not in key_l:
                    continue
                score = 1.0 + float(overlap)
                if q in content_l:
                    score += 0.5
                if q in key_l:
                    score += 0.5
            results.append(SearchResult(record=rec, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: max(k, 0)]

    def delete(self, id: str) -> None:
        self._store.pop(id, None)

    def clear(self) -> None:
        self._store.clear()

    def count(self) -> int:
        return len(self._store)

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def records(self) -> list[MemoryRecord]:
        """Return a snapshot of all records."""
        return list(self._store.values())
