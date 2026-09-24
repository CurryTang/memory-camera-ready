"""
DenseRetriever — cosine similarity retrieval using sentence-transformers embeddings.

Used by C3 (chunked RAG), C4, C5, C7, C8.
Lazy-builds numpy embedding matrix at first retrieve call (cached per Index object).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from agentmem.retrieval.base import Index, IndexUnit, RetrievalResult, Retriever

class DenseRetriever(Retriever):
    """
    Dense retrieval using sentence-transformers.

    Embeds all index units at first call, then uses cosine similarity.

    Args:
        embedding_model: sentence-transformers model name.
        batch_size: Embedding batch size for document encoding.
        normalize: If True, normalize embeddings to unit length before dot product.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 128,
        normalize: bool = True,
    ) -> None:
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.normalize = normalize

        self._model: Optional[object] = None
        self._cached_index_id: Optional[int] = None
        self._cached_matrix: Optional[np.ndarray] = None
        self._cached_units: list[IndexUnit] = []

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer                
            except ImportError as exc:
                raise RuntimeError(
                    "DenseRetriever requires sentence-transformers. "
                    "Install: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.embedding_model, device="cpu")
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        vecs = model.encode(                              
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return np.array(vecs, dtype=np.float32)

    def _build_matrix(self, index: Index) -> tuple[np.ndarray, list[IndexUnit]]:
        units = index.units
        texts = [u.content for u in units]
        matrix = self._embed(texts)
        return matrix, units

    def retrieve(self, query: str, index: Index, k: int = 10) -> list[RetrievalResult]:
        if not index.units:
            return []

        index_obj_id = id(index)
        if self._cached_index_id != index_obj_id:
            self._cached_matrix, self._cached_units = self._build_matrix(index)
            self._cached_index_id = index_obj_id

        q_vec = self._embed([query])[0]                
        scores: np.ndarray = self._cached_matrix @ q_vec                           

        top_k = min(k, len(self._cached_units))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            unit = self._cached_units[idx]
            results.append(
                RetrievalResult(
                    doc_id=unit.id,
                    content=unit.content,
                    score=float(scores[idx]),
                    metadata=dict(unit.metadata or {}),
                )
            )
        return results
