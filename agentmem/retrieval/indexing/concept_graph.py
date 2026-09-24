"""
ConceptGraphBuilder — builds a bipartite graph (propositions ↔ concepts).

Used by C7. Builds on top of PropositionExtractor output.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from agentmem.retrieval.base import Document, Index, IndexBuilder, IndexUnit

class ConceptGraphBuilder(IndexBuilder):
    """
    Builds a bipartite concept graph from PropositionExtractor output.

    Expects Index.units to already contain proposition units with
    metadata["concepts"] from PropositionExtractor.

    Adds concept-node IndexUnits that aggregate all propositions
    sharing a concept.

    Args:
        min_concept_occurrences: Skip concepts that appear fewer times.
    """

    def __init__(self, min_concept_occurrences: int = 1) -> None:
        self._min_occurrences = max(1, min_concept_occurrences)

    def build_from_index(self, prop_index: Index) -> Index:
        """Build concept graph from an existing proposition index."""
        concept_to_props: dict[str, list[IndexUnit]] = defaultdict(list)

        for unit in prop_index.units:
            concepts = unit.metadata.get("concepts", [])
            for concept in concepts:
                concept_to_props[concept.lower().strip()].append(unit)

        units: list[IndexUnit] = list(prop_index.units)                     
        concept_units: list[IndexUnit] = []

        for concept, prop_units in concept_to_props.items():
            if len(prop_units) < self._min_occurrences:
                continue

            agg_text = f"CONCEPT: {concept}\nPropositions:\n" + "\n".join(
                f"- {u.content}" for u in prop_units
            )
            concept_units.append(
                IndexUnit(
                    id=f"concept_{concept.replace(' ', '_')}",
                    content=agg_text,
                    metadata={
                        "node_type": "concept",
                        "concept": concept,
                        "linked_prop_ids": [u.id for u in prop_units],
                        "occurrence_count": len(prop_units),
                    },
                    source_doc_id=None,
                )
            )

        all_units = units + concept_units
        return Index(
            units=all_units,
            metadata={
                "builder": "ConceptGraphBuilder",
                "n_propositions": len(units),
                "n_concepts": len(concept_units),
            },
        )

    def build(self, documents: list[Document]) -> Index:
        """Fallback: treat documents as raw units and extract concepts from content."""
        from agentmem.retrieval.indexing.proposition import PropositionExtractor
        extractor = PropositionExtractor()
        prop_index = extractor.build(documents)
        return self.build_from_index(prop_index)
