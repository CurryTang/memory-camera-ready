"""
MultiHopTraverser — multi-hop traversal through concept/proposition graph.

Used by C7. Traverses the bipartite concept graph built by ConceptGraphBuilder,
starting from query-relevant concept nodes. Returns empty list with a warning
when the graph structure is absent (does NOT silently fall back to BM25).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agentmem.retrieval.base import Index, IndexUnit, RetrievalResult, Retriever

logger = logging.getLogger(__name__)

class MultiHopTraverser(Retriever):
    """
    Multi-hop graph traversal over a concept graph index.

    Starting from query-relevant concept nodes (identified by keyword overlap),
    traverses linked propositions and expands to neighboring concepts up to
    max_hops hops away.

    Returns empty list with a warning when the index contains no concept nodes.
    Does NOT silently fall back to BM25.

    Args:
        max_hops: Maximum traversal depth.
        beam_width: Number of concept nodes to expand at each hop.
    """

    def __init__(self, max_hops: int = 2, beam_width: int = 5) -> None:
        self._max_hops = max_hops
        self._beam_width = beam_width

    def _tokenize(self, text: str) -> set[str]:
        import re
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def retrieve(self, query: str, index: Index, k: int = 10) -> list[RetrievalResult]:
        """Retrieve propositions via multi-hop concept graph traversal.

        Steps:
        1. Find concept nodes with highest keyword overlap with the query.
        2. Collect linked proposition units from those concepts.
        3. Expand to neighboring concepts via shared propositions (up to max_hops).
        4. Return all collected propositions ranked by hop distance then overlap score.

        Returns empty list with a warning if no concept nodes are present.
        Does NOT silently fall back to BM25.

        Args:
            query: The question or search query.
            index: Concept graph Index (from ConceptGraphBuilder).
            k: Number of results to return.

        Returns:
            List of RetrievalResult objects, or [] if graph structure is absent.
        """
        if not index.units:
            return []

        concept_units = [u for u in index.units if u.metadata.get("node_type") == "concept"]
        prop_units = [u for u in index.units if u.metadata.get("node_type") != "concept"]

        if not concept_units:

            logger.warning(
                "MultiHopTraverser: no concept nodes in index; C7 results for this query will be empty. "
                "Ensure ConceptGraphBuilder ran successfully before calling MultiHopTraverser."
            )
            return []

        prop_map: dict[str, IndexUnit] = {u.id: u for u in prop_units}
        concept_map: dict[str, IndexUnit] = {u.id: u for u in concept_units}

        concept_to_props: dict[str, list[str]] = {}
        prop_to_concepts: dict[str, list[str]] = {}
        for cu in concept_units:
            linked = cu.metadata.get("linked_prop_ids", [])
            cname = cu.id
            concept_to_props[cname] = list(linked)
            for pid in linked:
                prop_to_concepts.setdefault(pid, []).append(cname)

        query_tokens = self._tokenize(query)

        concept_scores: dict[str, float] = {}
        for cu in concept_units:
            concept_text = cu.metadata.get("concept", "") + " " + cu.content
            overlap = len(query_tokens & self._tokenize(concept_text))
            if overlap > 0:
                concept_scores[cu.id] = overlap

        if not concept_scores:
            logger.warning(
                "MultiHopTraverser: no query-concept overlap found; C7 results for this query will be empty."
            )
            return []

        seed_concepts = sorted(concept_scores, key=lambda x: -concept_scores[x])[: self._beam_width]

        collected_prop_ids: dict[str, float] = {}                    
        frontier: set[str] = set(seed_concepts)
        visited_concepts: set[str] = set()

        for hop in range(self._max_hops):
            hop_score = 1.0 / (hop + 1)
            next_frontier: set[str] = set()
            for cid in frontier:
                if cid in visited_concepts:
                    continue
                visited_concepts.add(cid)
                for pid in concept_to_props.get(cid, []):
                    if pid not in collected_prop_ids:
                        collected_prop_ids[pid] = hop_score

                    for neighbor_cid in prop_to_concepts.get(pid, []):
                        if neighbor_cid not in visited_concepts:
                            next_frontier.add(neighbor_cid)
            frontier = next_frontier
            if not frontier:
                break

        results: list[RetrievalResult] = []
        for pid, score in sorted(collected_prop_ids.items(), key=lambda x: -x[1]):
            unit = prop_map.get(pid)
            if unit is None:
                continue
            results.append(RetrievalResult(
                doc_id=unit.id,
                content=unit.content,
                score=score,
                metadata=dict(unit.metadata or {}),
            ))
            if len(results) >= k:
                break

        return results
