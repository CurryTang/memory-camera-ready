"""
FixedSizeChunker — splits documents into fixed-token-sized overlapping chunks.

Used by C3 (RAG chunk + dense embedding).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agentmem.retrieval.base import Document, Index, IndexBuilder, IndexUnit

def _simple_tokenize(text: str) -> list[str]:
    """Whitespace tokenization used for chunk sizing (fast, no dependencies)."""
    return text.split()

def _detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)

class FixedSizeChunker(IndexBuilder):
    """
    Splits each document into fixed-size overlapping token chunks.

    Args:
        chunk_size: Target chunk size in whitespace tokens.
        overlap: Number of tokens to overlap between adjacent chunks.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                f"overlap must be in [0, chunk_size), got overlap={overlap}, chunk_size={chunk_size}"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def build(self, documents: list[Document]) -> Index:
        units: list[IndexUnit] = []
        for doc in documents:
            tokens = _simple_tokenize(doc.content)
            if not tokens:

                units.append(
                    IndexUnit(
                        id=f"{doc.id}_c0",
                        content=doc.content,
                        metadata={"chunk_index": 0, "source_doc_id": doc.id, **doc.metadata},
                        source_doc_id=doc.id,
                    )
                )
                continue

            step = self.chunk_size - self.overlap
            chunk_index = 0
            pos = 0
            while pos < len(tokens):
                chunk_tokens = tokens[pos : pos + self.chunk_size]
                chunk_text = _detokenize(chunk_tokens)
                meta = {
                    "chunk_index": chunk_index,
                    "token_start": pos,
                    "token_end": pos + len(chunk_tokens),
                    "source_doc_id": doc.id,
                    **doc.metadata,
                }
                units.append(
                    IndexUnit(
                        id=f"{doc.id}_c{chunk_index}",
                        content=chunk_text,
                        metadata=meta,
                        source_doc_id=doc.id,
                    )
                )
                chunk_index += 1
                pos += step

        return Index(
            units=units,
            metadata={
                "builder": "FixedSizeChunker",
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "n_chunks": len(units),
            },
        )
