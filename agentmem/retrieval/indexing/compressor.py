"""
LLMCompressor — density-gated LLM compression of dialogue turns.

Ports the SimpleMem-style compression: coref resolution + time normalization +
density gating. Used by C4 and C5.
"""
from __future__ import annotations

from typing import Any, Optional

from agentmem.retrieval.base import Document, Index, IndexBuilder, IndexUnit

class LLMCompressor(IndexBuilder):
    """
    Compresses each document via an LLM before indexing.

    For each dialogue turn, calls the LLM to:
    1. Resolve coreferences (replace pronouns with entity names)
    2. Normalize time references (convert relative to absolute timestamps)
    3. Density-gate: skip turns below a salience threshold

    Args:
        provider: OpenAICompatibleProvider instance for LLM calls.
        model: Model name (used if provider is None).
        api_key: API key (used if provider is None).
        base_url: Base URL override (used if provider is None).
        density_threshold: Salience threshold 0-1; turns below this are skipped.
        max_tokens: Max tokens for compression output.
    """

    _INIT_PARAM_NAMES = frozenset([
        "provider", "model", "api_key", "base_url",
        "density_threshold", "max_tokens",
    ])

    @classmethod
    def _init_params(cls) -> frozenset:
        return cls._INIT_PARAM_NAMES

    def __init__(
        self,
        provider: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        density_threshold: float = 0.3,
        max_tokens: int = 512,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._density_threshold = density_threshold
        self._max_tokens = max_tokens

    def _get_provider(self):
        if self._provider is not None:
            return self._provider
        if not self._api_key:
            import os
            self._api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        from agentmem.providers.openai_compat import OpenAICompatibleProvider
        self._provider = OpenAICompatibleProvider(
            api_key=self._api_key,
            model=self._model,
            base_url=self._base_url,
        )
        return self._provider

    _COMPRESS_SYSTEM = (
        "You are a memory compression assistant. Given a dialogue turn, perform:\n"
        "1. Coreference resolution: replace pronouns/demonstratives with their referents.\n"
        "2. Time normalization: replace relative time expressions with the provided "
        "absolute timestamp when available.\n"
        "3. Density gating: if the turn contains no factual content worth remembering, "
        "output exactly: SKIP\n\n"
        "Otherwise output the compressed, self-contained version of the turn. "
        "Preserve speaker labels. Be concise."
    )

    def _compress_turn(self, provider: Any, content: str, metadata: dict) -> Optional[str]:
        """Compress a single turn via LLM; return None if below density threshold.

        Calls the LLM to:
        1. Resolve coreferences (replace pronouns with entity names).
        2. Normalize time references using the turn's timestamp metadata.
        3. Density-gate: if the LLM returns "SKIP", this turn is dropped.

        Args:
            provider: OpenAICompatibleProvider instance (pre-initialized).
            content: Raw turn text.
            metadata: Turn metadata (may include 'timestamp').

        Returns:
            Compressed turn string, or None if the turn should be dropped.
        """
        from agentmem.providers.base import Message

        try:
            if provider is None:
                provider = self._get_provider()
        except Exception:
            return content

        timestamp = metadata.get("timestamp", "unknown")
        user_prompt = (
            f"Turn timestamp: {timestamp}\n\n"
            f"Turn text:\n{content}\n\n"
            "Compressed version (or SKIP):"
        )
        messages = [
            Message(role="system", content=self._COMPRESS_SYSTEM),
            Message(role="user", content=user_prompt),
        ]
        try:
            response = provider.chat(messages, max_tokens=self._max_tokens)
            compressed = (response.content or "").strip()
        except Exception:

            return content

        if not compressed or compressed.upper() == "SKIP":
            return None

        if self._density_threshold > 0:
            raw_word_count = len(content.split())
            compressed_word_count = len(compressed.split())
            if raw_word_count > 0:
                salience = compressed_word_count / max(raw_word_count, 1)

                if compressed_word_count < 3 and salience < self._density_threshold:
                    return None

        return compressed

    def build(self, documents: list[Document]) -> Index:
        try:
            provider = self._get_provider()
        except Exception:
            provider = None
        units: list[IndexUnit] = []
        for doc in documents:
            try:
                compressed = self._compress_turn(provider, doc.content, doc.metadata)
            except NotImplementedError:

                compressed = doc.content
            if compressed is None:
                continue
            units.append(
                IndexUnit(
                    id=doc.id,
                    content=compressed,
                    metadata={**doc.metadata, "compressed": True},
                    source_doc_id=doc.id,
                )
            )
        return Index(
            units=units,
            metadata={"builder": "LLMCompressor", "n_units": len(units)},
        )
