"""
PersonalizedPageRank — PPR graph walk over knowledge graph for associative retrieval.

Used by C8 (HippoRAG2-inspired). Builds an entity co-occurrence graph from
KnowledgeGraphBuilder triples, adds passage nodes, and runs PPR seeded from
query entities matched via **embedding similarity** (cosine).

Returns passage-level results ranked by aggregated PPR score.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, List, Optional

import numpy as np

from agentmem.retrieval.base import Index, IndexUnit, RetrievalResult, Retriever

logger = logging.getLogger(__name__)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a and matrix b (each row a vector)."""
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm

class PersonalizedPageRank(Retriever):
    """
    Personalized PageRank over a KnowledgeGraphBuilder index.

    Seeds the PPR walk from query-relevant entities found via embedding
    similarity (cosine ≥ threshold).  Returns passage-level results.

    Args:
        embed_fn: Callable that takes a list of strings and returns np.ndarray
                  of shape (n, dim).  Required for embedding-based seed matching.
        damping: PPR damping factor (0.5 per HippoRAGv2 paper).
        max_iter: Maximum PPR iterations.
        n_seed_entities: How many top-scoring query-entity matches to use as seeds.
        seed_threshold: Minimum cosine similarity to qualify as a seed.
        passage_weight: Weight for passage-entity "contains" edges.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        damping: float = 0.5,
        max_iter: int = 50,
        n_seed_entities: int = 10,
        seed_threshold: float = 0.5,
        passage_weight: float = 0.05,
    ) -> None:
        self._embed_fn = embed_fn
        self._damping = damping
        self._max_iter = max_iter
        self._n_seeds = n_seed_entities
        self._seed_threshold = seed_threshold
        self._passage_weight = passage_weight

    def _run_ppr(
        self,
        nodes: list[str],
        adj: dict[str, dict[str, float]],
        seeds: list[str],
    ) -> dict[str, float]:
        if not nodes or not seeds:
            return {}

        seed_set = set(s for s in seeds if s in set(nodes))
        if not seed_set:
            return {}

        p = {n: (1.0 / len(seed_set) if n in seed_set else 0.0) for n in nodes}
        r = dict(p)
        out_weights = {n: sum(adj.get(n, {}).values()) for n in nodes}

        for _ in range(self._max_iter):
            new_r: dict[str, float] = {}
            for node in nodes:
                incoming = sum(
                    r.get(src, 0.0) * (w / max(out_weights[src], 1e-9))
                    for src, nbrs in adj.items()
                    for nbr, w in nbrs.items()
                    if nbr == node
                )
                new_r[node] = (1.0 - self._damping) * p.get(node, 0.0) + self._damping * incoming

            delta = sum(abs(new_r.get(nd, 0.0) - r.get(nd, 0.0)) for nd in nodes)
            r = new_r
            if delta < 1e-6:
                break

        return r

    def _find_seeds_embedding(
        self, query: str, entities: list[str]
    ) -> list[tuple[str, float]]:
        """Match query to KG entities via embedding cosine similarity."""
        if self._embed_fn is None:
            raise RuntimeError(
                "PersonalizedPageRank requires embed_fn for seed matching. "
                "Pass an embedding function when constructing the retriever."
            )
        if not entities:
            return []

        all_texts = [query] + entities
        embeddings = self._embed_fn(all_texts)              
        query_emb = embeddings[0]
        entity_embs = embeddings[1:]

        sims = _cosine_sim(query_emb, entity_embs)

        scored = [
            (entities[i], float(sims[i]))
            for i in range(len(entities))
            if sims[i] >= self._seed_threshold
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:self._n_seeds]

    def retrieve(self, query: str, index: Index, k: int = 10) -> list[RetrievalResult]:
        """Retrieve passages via PPR on the knowledge graph with passage nodes."""
        if not index.units:
            return []

        triple_units = [u for u in index.units if u.metadata.get("triple_type") == "openie"]

        if not triple_units:
            logger.warning(
                "PersonalizedPageRank: no OpenIE triples in index; returning empty."
            )
            return []

        entity_to_triples: dict[str, list[IndexUnit]] = defaultdict(list)
        adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        passage_nodes: set[str] = set()

        for unit in triple_units:
            subj = unit.metadata.get("subject", "").lower().strip()
            obj = unit.metadata.get("object", "").lower().strip()
            src_doc = unit.metadata.get("source_doc_id", "")

            if subj:
                entity_to_triples[subj].append(unit)
            if obj:
                entity_to_triples[obj].append(unit)
            if subj and obj:
                adj[subj][obj] += 1.0
                adj[obj][subj] += 1.0
            if src_doc:
                p_node = f"_passage_{src_doc}"
                passage_nodes.add(p_node)
                if subj:
                    adj[p_node][subj] += self._passage_weight
                    adj[subj][p_node] += self._passage_weight
                if obj:
                    adj[p_node][obj] += self._passage_weight
                    adj[obj][p_node] += self._passage_weight

        if not entity_to_triples:
            logger.warning("PersonalizedPageRank: no entities in triples.")
            return []

        all_entities = list(entity_to_triples.keys())
        seed_candidates = self._find_seeds_embedding(query, all_entities)

        if not seed_candidates:
            logger.warning(
                "PersonalizedPageRank: no entities matched query via embedding (threshold=%.2f).",
                self._seed_threshold,
            )
            return []

        seeds = [e for e, _ in seed_candidates]

        all_nodes = list(set(entity_to_triples.keys()) | passage_nodes)
        ppr_scores = self._run_ppr(all_nodes, dict(adj), seeds)

        doc_scores: dict[str, float] = defaultdict(float)
        for p_node in passage_nodes:
            score = ppr_scores.get(p_node, 0.0)
            if score > 0:
                doc_scores[p_node.replace("_passage_", "")] += score
        for entity, score in ppr_scores.items():
            if entity in passage_nodes or score <= 0:
                continue
            for unit in entity_to_triples.get(entity, []):
                src_doc = unit.metadata.get("source_doc_id", "")
                if src_doc:
                    doc_scores[src_doc] += score

        ranked = sorted(doc_scores.items(), key=lambda x: -x[1])[:k]

        triple_by_doc: dict[str, IndexUnit] = {}
        for u in triple_units:
            src = u.metadata.get("source_doc_id", u.source_doc_id or "")
            if src and src not in triple_by_doc:
                triple_by_doc[src] = u

        results: list[RetrievalResult] = []
        for doc_id, score in ranked:
            unit = triple_by_doc.get(doc_id)
            if unit is None:
                continue
            results.append(RetrievalResult(
                doc_id=doc_id,
                content="",
                score=score,
                metadata=dict(unit.metadata or {}),
            ))

        return results
