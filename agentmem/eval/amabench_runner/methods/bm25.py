"""BM25 retrieval method for AMAbench."""

from __future__ import annotations

from typing import Any, List

from rank_bm25 import BM25Okapi

from agentmem.eval.amabench_runner.methods.base import BaseMethod

class BM25Memory:
    def __init__(self, documents: List[str], bm25_index: BM25Okapi):
        self.documents = documents
        self.bm25_index = bm25_index

class BM25Method(BaseMethod):
    def __init__(self, top_k: int = 10, config_path: str = None, **_kw):
        if config_path:
            cfg = self._load_config(config_path)
            top_k = cfg.get("top_k", top_k)
        self.top_k = top_k

    def memory_construction(self, traj_text: str, task: str = "") -> BM25Memory:
        documents = _split_turns(traj_text)
        corpus_tokens = [doc.lower().split() for doc in documents]
        return BM25Memory(documents, BM25Okapi(corpus_tokens))

    def memory_retrieve(self, memory: BM25Memory, question: str) -> str:
        scores = memory.bm25_index.get_scores(question.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]
        return "\n\n".join(memory.documents[i] for i in top_idx)

def _split_turns(text: str) -> List[str]:
    """Split trajectory text into per-turn documents."""
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
    return documents or [text]
