"""
CausalGraphBuilder — extracts causal event edges from trajectory turns via LLM.

Used by C6 (AMA-Agent inspired). When the LLM is unavailable, logs a warning
and stores raw turns as units (no causal edges extracted).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from agentmem.retrieval.base import Document, Index, IndexBuilder, IndexUnit

logger = logging.getLogger(__name__)

@dataclass
class CausalEdge:
    """A directed causal edge: cause → effect."""
    cause: str
    effect: str
    evidence: str                    
    confidence: float = 1.0

class CausalGraphBuilder(IndexBuilder):
    """
    Builds a causal graph from dialogue turns.

    Each IndexUnit in the output represents a causal edge (cause → effect),
    stored as text and with graph metadata.

    Args:
        provider: OpenAICompatibleProvider instance.
        model: LLM model name.
        api_key: API key.
        base_url: Base URL override.
    """

    def __init__(
        self,
        provider: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    _EXTRACT_SYSTEM = (
        "You are an event-graph extraction assistant. "
        "Given a dialogue turn, extract all causal relationships as JSON.\n\n"
        "Output a JSON array of objects with keys:\n"
        "  cause (str): the cause event or state\n"
        "  effect (str): the resulting event or state\n"
        "  confidence (float 0-1): your confidence in this causal link\n\n"
        "If there are no causal relationships, output an empty array: []\n"
        "Output only valid JSON, no commentary."
    )

    def _get_provider(self):
        if self._provider is not None:
            return self._provider
        import os
        api_key = self._api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        from agentmem.providers.openai_compat import OpenAICompatibleProvider
        self._provider = OpenAICompatibleProvider(
            api_key=api_key,
            model=self._model,
            base_url=self._base_url,
        )
        return self._provider

    def _extract_causal_edges(self, provider: Any, turn: str) -> list[CausalEdge]:
        """Extract causal edges from a single turn via LLM.

        Calls the LLM with a structured prompt requesting JSON output of
        (cause, effect, confidence) triples.

        Args:
            provider: OpenAICompatibleProvider instance (may be None, will be
                      auto-initialized from self._model/api_key/base_url).
            turn: Raw turn text to analyze.

        Returns:
            List of CausalEdge objects extracted from the turn.
        """
        import json
        from agentmem.providers.base import Message

        try:
            if provider is None:
                provider = self._get_provider()
        except Exception:
            return []

        messages = [
            Message(role="system", content=self._EXTRACT_SYSTEM),
            Message(role="user", content=f"Dialogue turn:\n{turn}"),
        ]
        try:
            response = provider.chat(messages, max_tokens=512)
            raw = (response.content or "").strip()
        except Exception:
            return []

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []

        if not isinstance(data, list):
            return []

        edges: list[CausalEdge] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            cause = str(item.get("cause", "")).strip()
            effect = str(item.get("effect", "")).strip()
            if not cause or not effect:
                continue
            confidence = float(item.get("confidence", 1.0))
            edges.append(CausalEdge(cause=cause, effect=effect, evidence=turn, confidence=confidence))
        return edges

    def build(self, documents: list[Document]) -> Index:
        units: list[IndexUnit] = []
        graph_edges: list[dict] = []

        for doc in documents:
            try:
                edges = self._extract_causal_edges(None, doc.content)
            except NotImplementedError:
                logger.warning(
                    "CausalGraphBuilder: LLM stub raised NotImplementedError for doc %s; "
                    "storing raw turn (no causal edges). C6 will degrade to empty retrieval.",
                    doc.id,
                )
                edges = []

            if not edges:

                units.append(
                    IndexUnit(
                        id=f"cg_{doc.id}",
                        content=doc.content,
                        metadata={"edge_type": "raw_turn", "source_doc_id": doc.id, **doc.metadata},
                        source_doc_id=doc.id,
                    )
                )
                continue

            for i, edge in enumerate(edges):
                content = f"CAUSE: {edge.cause}\nEFFECT: {edge.effect}\nEVIDENCE: {edge.evidence}"
                units.append(
                    IndexUnit(
                        id=f"cg_{doc.id}_e{i}",
                        content=content,
                        metadata={
                            "edge_type": "causal",
                            "cause": edge.cause,
                            "effect": edge.effect,
                            "confidence": edge.confidence,
                            "source_doc_id": doc.id,
                            **doc.metadata,
                        },
                        source_doc_id=doc.id,
                    )
                )
                graph_edges.append({"cause": edge.cause, "effect": edge.effect, "doc_id": doc.id})

        return Index(
            units=units,
            metadata={
                "builder": "CausalGraphBuilder",
                "n_units": len(units),
                "n_edges": len(graph_edges),
                "edges": graph_edges,
            },
        )
