"""
Base abstractions for the agentmem.retrieval module.

Defines composable primitives for C1-C8 index/retrieval configurations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Document:
    """A raw input document to be indexed."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

@dataclass
class IndexUnit:
    """An atomic unit stored in an index (may be a chunk, turn, proposition, etc.)."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_doc_id: Optional[str] = None
    embeddings: Optional[List[float]] = None

@dataclass
class Index:
    """A built index containing units and associated metadata."""

    units: List[IndexUnit] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    """A single retrieval result with a relevance score."""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalConfig:
    """Configuration for a composed retrieval pipeline."""

    index_builder: Optional[Any] = None                          
    retriever: Optional[Any] = None                           
    compositor: Optional[Any] = None                           
    params: Dict[str, Any] = field(default_factory=dict)

class IndexBuilder(ABC):
    """Abstract base for index builders.

    Takes raw Documents and produces an Index.
    """

    @abstractmethod
    def build(self, documents: List[Document]) -> Index:
        """Build an index from the given documents."""
        ...

class Retriever(ABC):
    """Abstract base for retrievers.

    Takes a query and an Index, returns ranked RetrievalResults.
    """

    @abstractmethod
    def retrieve(self, query: str, index: Index, k: int = 10) -> List[RetrievalResult]:
        """Retrieve the top-k results for the given query."""
        ...

class Compositor(ABC):
    """Abstract base for compositors.

    Orchestrates multiple retrievers or adds LLM-in-the-loop logic.
    """

    @abstractmethod
    def compose(
        self,
        query: str,
        index: Index,
        k: int = 10,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Compose a final ranked list from sub-retrievers or reasoning steps."""
        ...
