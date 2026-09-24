"""
PlainTextIndex — stores each document as a single IndexUnit (no chunking).

Used by C1 (BM25) and C2 (substring match).
"""
from __future__ import annotations

from agentmem.retrieval.base import Document, Index, IndexBuilder, IndexUnit

class PlainTextIndex(IndexBuilder):
    """
    Trivial index builder: one IndexUnit per Document.

    No chunking, no embedding — just preserves text and metadata.
    """

    def build(self, documents: list[Document]) -> Index:
        units: list[IndexUnit] = []
        for doc in documents:
            units.append(
                IndexUnit(
                    id=doc.id,
                    content=doc.content,
                    metadata=dict(doc.metadata or {}),
                    source_doc_id=doc.id,
                )
            )
        return Index(units=units, metadata={"builder": "PlainTextIndex", "n_docs": len(documents)})
