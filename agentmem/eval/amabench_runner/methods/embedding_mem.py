"""Embedding-based retrieval method for AMAbench."""

from __future__ import annotations

from typing import Any, List

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from agentmem.eval.amabench_runner.methods.base import BaseMethod

class EmbeddingMemory:
    def __init__(self, documents: List[str], embeddings: np.ndarray, index: Any = None):
        self.documents = documents
        self.embeddings = embeddings
        self.index = index

class EmbeddingMethod(BaseMethod):
    def __init__(
        self,
        top_k: int = 5,
        use_faiss: bool = True,
        config_path: str = None,
        embedding_engine: Any = None,
        **_kw,
    ):
        if config_path:
            cfg = self._load_config(config_path)
            top_k = cfg.get("top_k", top_k)
            use_faiss = cfg.get("use_faiss", use_faiss)
        self.top_k = top_k
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        if embedding_engine is None:
            raise ValueError("EmbeddingMethod requires an embedding_engine")
        self.embedding_engine = embedding_engine

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.embedding_engine.encode(texts)

    def memory_construction(self, traj_text: str, task: str = "") -> EmbeddingMemory:
        full_text = f"Task: {task}\n\n{traj_text}" if task else traj_text
        documents = _split_turns(full_text)
        embeddings = self._encode(documents)

        index = None
        if self.use_faiss:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)

        return EmbeddingMemory(documents, embeddings, index)

    def memory_retrieve(self, memory: EmbeddingMemory, question: str) -> str:
        q_emb = self._encode([question])

        if memory.index is not None:
            faiss.normalize_L2(q_emb)
            _, indices = memory.index.search(q_emb, self.top_k)
            top_idx = indices[0].tolist()
        else:
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            d_norm = memory.embeddings / (np.linalg.norm(memory.embeddings, axis=1, keepdims=True) + 1e-9)
            sims = np.dot(d_norm, q_norm.T).flatten()
            top_idx = np.argsort(sims)[::-1][: self.top_k].tolist()

        return "\n\n".join(memory.documents[i] for i in top_idx if i < len(memory.documents))

def _split_turns(text: str) -> List[str]:
    documents: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.strip().startswith(("Turn ", "Step ")):
            if current:
                documents.append("\n".join(current))
                current = []
        current.append(line)
    if current:
        documents.append("\n".join(current))
    if not documents:
        chunk_size = 500
        documents = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return documents
