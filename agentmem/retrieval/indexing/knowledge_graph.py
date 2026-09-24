"""
KnowledgeGraphBuilder — OpenIE triple extraction → Knowledge Graph.

Used by C8 (HippoRAG2-inspired). When the LLM is unavailable, logs a warning
and stores raw turns as units (no triples extracted), causing PersonalizedPageRank
to return empty results with a warning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from agentmem.retrieval.base import Document, Index, IndexBuilder, IndexUnit

logger = logging.getLogger(__name__)

@dataclass
class Triple:
    """An OpenIE-style subject-predicate-object triple."""
    subject: str
    predicate: str
    obj: str
    source_doc_id: Optional[str] = None

class KnowledgeGraphBuilder(IndexBuilder):
    """
    Extracts OpenIE triples from turns and builds a knowledge graph.

    Each triple becomes an IndexUnit. Graph structure is stored in Index.metadata.

    Args:
        provider: OpenAICompatibleProvider instance.
        model: LLM model name.
        max_triples_per_turn: Hard cap on triples per turn.
    """

    def __init__(
        self,
        provider: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_triples_per_turn: int = 15,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_triples = max(1, max_triples_per_turn)

    _EXTRACT_SYSTEM = (
        "You are an OpenIE triple extraction assistant. "
        "Given a dialogue turn, extract factual (subject, predicate, object) triples.\n\n"
        "Output a JSON array of objects with keys:\n"
        "  subject (str): entity or concept\n"
        "  predicate (str): relation verb phrase\n"
        "  object (str): entity, concept, or value\n\n"
        "If no triples can be extracted, output: []\n"
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

    def _extract_triples(self, provider: Any, turn: str) -> list[Triple]:
        """Extract OpenIE (subject, predicate, object) triples from a turn via LLM.

        Args:
            provider: OpenAICompatibleProvider instance (may be None for lazy init).
            turn: Raw dialogue turn text.

        Returns:
            List of Triple objects (up to max_triples_per_turn).
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

        triples: list[Triple] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            subj = str(item.get("subject", "")).strip()
            pred = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            if not subj or not pred or not obj:
                continue
            triples.append(Triple(subject=subj, predicate=pred, obj=obj))
        return triples

    def build(self, documents: list[Document], max_workers: int = 2) -> Index:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        units: list[IndexUnit] = []
        all_triples: list[dict] = []

        def _extract_for_doc(doc: Document) -> tuple[Document, list[Triple]]:
            try:
                return doc, self._extract_triples(None, doc.content)
            except NotImplementedError:
                logger.warning(
                    "KnowledgeGraphBuilder: LLM stub raised NotImplementedError for doc %s; "
                    "storing raw turn (no triples). C8 PersonalizedPageRank will return empty results.",
                    doc.id,
                )
                return doc, []

        results: list[tuple[Document, list[Triple]]] = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(documents))) as ex:
            future_to_doc = {ex.submit(_extract_for_doc, doc): doc for doc in documents}
            for future in as_completed(future_to_doc):
                results.append(future.result())

        doc_order = {doc.id: i for i, doc in enumerate(documents)}
        results.sort(key=lambda r: doc_order.get(r[0].id, 0))

        for doc, triples in results:
            if not triples:
                units.append(
                    IndexUnit(
                        id=f"kg_{doc.id}",
                        content=doc.content,
                        metadata={"triple_type": "raw", "source_doc_id": doc.id, **doc.metadata},
                        source_doc_id=doc.id,
                    )
                )
                continue

            for i, triple in enumerate(triples[: self._max_triples]):
                text = f"{triple.subject} {triple.predicate} {triple.obj}"
                units.append(
                    IndexUnit(
                        id=f"kg_{doc.id}_t{i}",
                        content=text,
                        metadata={
                            "triple_type": "openie",
                            "subject": triple.subject,
                            "predicate": triple.predicate,
                            "object": triple.obj,
                            "source_doc_id": doc.id,
                            **doc.metadata,
                        },
                        source_doc_id=doc.id,
                    )
                )
                all_triples.append({
                    "s": triple.subject,
                    "p": triple.predicate,
                    "o": triple.obj,
                    "doc_id": doc.id,
                })

        return Index(
            units=units,
            metadata={
                "builder": "KnowledgeGraphBuilder",
                "n_triples": len(all_triples),
                "triples": all_triples,
            },
        )
