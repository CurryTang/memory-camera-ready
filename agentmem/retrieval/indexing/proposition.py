"""
PropositionExtractor — extracts atomic propositions + concept tags from turns via LLM.

Used by C7 (PlugMem-inspired). When the LLM is unavailable, logs a warning and
stores the full turn as a single proposition (concept graph structure will be absent,
causing MultiHopTraverser to return empty results with a warning).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from agentmem.retrieval.base import Document, Index, IndexBuilder, IndexUnit

logger = logging.getLogger(__name__)

@dataclass
class Proposition:
    """An atomic proposition with associated concept tags."""
    text: str
    concepts: list[str] = field(default_factory=list)
    source_doc_id: Optional[str] = None

class PropositionExtractor(IndexBuilder):
    """
    Extracts atomic propositions from dialogue turns.

    Each proposition becomes an IndexUnit. Concept tags are stored in metadata.

    Args:
        provider: OpenAICompatibleProvider instance.
        model: LLM model name.
        max_propositions_per_turn: Hard cap per turn.
    """

    def __init__(
        self,
        provider: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_propositions_per_turn: int = 10,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_props = max(1, max_propositions_per_turn)

    _EXTRACT_SYSTEM = (
        "You are a proposition extraction assistant. "
        "Given a dialogue turn, decompose it into atomic propositions — "
        "simple, self-contained factual statements.\n\n"
        "Output a JSON array of objects with keys:\n"
        "  text (str): the atomic proposition as a complete sentence\n"
        "  concepts (list[str]): up to 5 key concept tags (nouns/noun phrases)\n\n"
        "If there are no useful propositions, output: []\n"
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

    def _extract_propositions(self, provider: Any, turn: str) -> list[Proposition]:
        """Extract atomic propositions from a single turn via LLM.

        Calls the LLM to decompose the turn into atomic, self-contained
        factual statements with associated concept tags.

        Args:
            provider: OpenAICompatibleProvider instance (may be None for lazy init).
            turn: Raw dialogue turn text.

        Returns:
            List of Proposition objects (up to max_propositions_per_turn).
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
            response = provider.chat(messages, max_tokens=768)
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

        props: list[Proposition] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            concepts_raw = item.get("concepts", [])
            concepts = [str(c) for c in concepts_raw if c] if isinstance(concepts_raw, list) else []
            props.append(Proposition(text=text, concepts=concepts))
        return props

    def build(self, documents: list[Document]) -> Index:
        units: list[IndexUnit] = []

        for doc in documents:
            try:
                props = self._extract_propositions(None, doc.content)
            except NotImplementedError:
                logger.warning(
                    "PropositionExtractor: LLM stub raised NotImplementedError for doc %s; "
                    "storing full turn as single proposition (no concept tags). "
                    "C7 MultiHopTraverser will return empty results.",
                    doc.id,
                )
                props = []

            if not props:

                props = [Proposition(text=doc.content, concepts=[], source_doc_id=doc.id)]

            for i, prop in enumerate(props[: self._max_props]):
                units.append(
                    IndexUnit(
                        id=f"prop_{doc.id}_{i}",
                        content=prop.text,
                        metadata={
                            "concepts": prop.concepts,
                            "prop_index": i,
                            "source_doc_id": doc.id,
                            **doc.metadata,
                        },
                        source_doc_id=doc.id,
                    )
                )

        return Index(
            units=units,
            metadata={
                "builder": "PropositionExtractor",
                "n_propositions": len(units),
            },
        )
