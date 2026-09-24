"""
ColBERTRetriever — late interaction MaxSim retrieval.

Implements ColBERT's MaxSim scoring:
    Score(q, d) = Σᵢ max_j (qᵢ · dⱼ)

where qᵢ are query token embeddings and dⱼ are document token embeddings.
Unlike dense retrieval (single vector per doc), each document is represented
as a matrix of token-level embeddings. At query time, each query token finds
its best-matching document token; scores are summed across query tokens.

Used by C9.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from agentmem.retrieval.base import Index, IndexUnit, RetrievalResult, Retriever

class ColBERTRetriever(Retriever):
    """
    Late interaction retriever using ColBERT-style MaxSim scoring.

    Each document chunk is encoded into a matrix of per-token embeddings
    at index build time (lazily cached). At query time, query tokens are
    encoded and MaxSim is computed against every document.

    Score(q, d) = Σᵢ max_j sim(qᵢ, dⱼ)

    Args:
        model_name: HuggingFace model name. Default: colbert-ir/colbertv2.0
            (128-dim token embeddings, purpose-built for late interaction).
            Any encoder-only transformer works as a drop-in.
        max_doc_length: Maximum tokens per document chunk (including special tokens).
        max_query_length: Maximum tokens per query.
        batch_size: Number of documents encoded per forward pass.
        normalize: If True, L2-normalize token embeddings before MaxSim
            (converts dot product to cosine similarity per token).
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        max_doc_length: int = 256,
        max_query_length: int = 32,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.batch_size = batch_size
        self.normalize = normalize

        self._tokenizer: Optional[object] = None
        self._model: Optional[object] = None

        self._cached_index_id: Optional[int] = None
        self._cached_token_matrices: Optional[list[np.ndarray]] = None
        self._cached_units: list[IndexUnit] = []

    def _load_model(self):
        if self._model is None:
            try:
                from transformers import AutoModel, AutoTokenizer                
            except ImportError as exc:
                raise RuntimeError(
                    "ColBERTRetriever requires transformers. "
                    "Install: pip install transformers"
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()                              
        return self._tokenizer, self._model

    def _encode_tokens(self, texts: list[str], max_length: int) -> list[np.ndarray]:
        """Encode a list of texts into per-token embedding matrices.

        Returns a list of numpy arrays, one per text, each of shape
        (n_valid_tokens, hidden_dim). Padding tokens are excluded.
        """
        import torch                

        tokenizer, model = self._load_model()
        all_matrices: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = tokenizer(                          
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = model(**inputs)                          

            token_embs = outputs.last_hidden_state.cpu().numpy()
            mask = inputs["attention_mask"].cpu().numpy().astype(bool)

            for j in range(len(batch)):
                emb = token_embs[j][mask[j]]                       
                if self.normalize:
                    norms = np.linalg.norm(emb, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-8, 1.0, norms)
                    emb = emb / norms
                all_matrices.append(emb.astype(np.float32))

        return all_matrices

    def _maxsim(self, query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """MaxSim score between one query and one document.

        query_emb: (q_len, dim)
        doc_emb:   (d_len, dim)
        Returns scalar: sum over query tokens of max similarity to any doc token.
        """

        sim = query_emb @ doc_emb.T                  
        return float(sim.max(axis=1).sum())

    def retrieve(self, query: str, index: Index, k: int = 10) -> list[RetrievalResult]:
        if not index.units:
            return []

        index_obj_id = id(index)
        if self._cached_index_id != index_obj_id:
            texts = [u.content for u in index.units]
            self._cached_token_matrices = self._encode_tokens(texts, self.max_doc_length)
            self._cached_units = list(index.units)
            self._cached_index_id = index_obj_id

        query_emb = self._encode_tokens([query], self.max_query_length)[0]

        scores = np.array(
            [self._maxsim(query_emb, doc_emb) for doc_emb in self._cached_token_matrices]
        )

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
