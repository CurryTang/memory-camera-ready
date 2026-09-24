"""
Storage layer base interface.

A MemoryStore is a pure data layer: add, get, search, delete, clear.
It has no knowledge of LLMs, agents, or operations.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class MemoryRecord:
    """
    A single unit of stored memory.

    Attributes:
        id: Unique identifier. Auto-generated if not provided.
        content: The text content of this memory.
        metadata: Arbitrary key-value metadata (timestamps, speaker, source, etc.).
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class SearchResult:
    """
    A ranked search result from a memory store.

    Attributes:
        record: The matched memory record.
        score: Relevance score (higher = more relevant). Semantics depend on backend.
    """

    record: MemoryRecord
    score: float

class BaseMemoryStore(ABC):
    """
    Abstract base for all storage backends.

    All implementations must be safe to call after clear() or on a fresh instance.
    Search should return results sorted by descending score.
    """

    @abstractmethod
    def add(self, record: MemoryRecord) -> str:
        """
        Persist a record and return its id.
        If record.id already exists, behavior is implementation-defined (upsert or error).
        """

    @abstractmethod
    def get(self, id: str) -> Optional[MemoryRecord]:
        """Return the record with the given id, or None if not found."""

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """
        Return up to k records most relevant to query, sorted by descending score.
        """

    @abstractmethod
    def delete(self, id: str) -> None:
        """Remove the record with the given id. No-op if not found."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all records from this store."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored records."""

    def add_many(self, records: list[MemoryRecord]) -> list[str]:
        """Convenience wrapper to add multiple records. Returns list of ids."""
        return [self.add(r) for r in records]
