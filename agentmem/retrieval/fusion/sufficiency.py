"""
SufficiencyChecker — LLM judges if retrieved context is sufficient to answer query.

Used by C6. Iteratively retrieves and asks the LLM if the context is sufficient
before committing to an answer. Falls back to single-round retrieval on LLM failure.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agentmem.retrieval.base import Compositor, Index, RetrievalResult

logger = logging.getLogger(__name__)

_SUFFICIENCY_SYSTEM = (
    "You are a retrieval assistant. Given a question and retrieved context, "
    "decide if the context contains enough information to answer the question.\n\n"
    "Reply with exactly one word: YES or NO."
)

class SufficiencyChecker(Compositor):
    """
    Iteratively retrieves context and checks if it is sufficient to answer the query.

    If the initial retrieval is insufficient (per LLM), fetches more context
    until satisfied or max_rounds is reached.

    Args:
        base_retriever: Initial retriever.
        provider: OpenAICompatibleProvider instance.
        max_rounds: Maximum sufficiency check rounds.
    """

    def __init__(
        self,
        base_retriever: Optional[Any] = None,
        provider: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_rounds: int = 2,
    ) -> None:
        self._base = base_retriever
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_rounds = max_rounds

    def _get_provider(self) -> Any:
        if self._provider is not None:
            return self._provider
        import os
        from agentmem.providers.openai_compat import OpenAICompatibleProvider
        api_key = self._api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self._provider = OpenAICompatibleProvider(
            api_key=api_key,
            model=self._model,
            base_url=self._base_url,
        )
        return self._provider

    def _is_sufficient(self, provider: Any, query: str, context: str) -> bool:
        """Ask LLM if the context is sufficient to answer the query.

        Returns True if YES, False if NO. Falls back to False on any error so
        the caller keeps fetching more context.
        """
        from agentmem.providers.base import Message
        try:
            prov = provider if provider is not None else self._get_provider()
            messages = [
                Message(role="system", content=_SUFFICIENCY_SYSTEM),
                Message(
                    role="user",
                    content=(
                        f"Question: {query}\n\n"
                        f"Context:\n{context[:2000]}\n\n"
                        "Is the context sufficient? (YES/NO)"
                    ),
                ),
            ]
            response = prov.chat(messages, max_tokens=5)
            answer = (response.content or "").strip().upper()
            return answer.startswith("YES")
        except Exception as exc:
            logger.debug("SufficiencyChecker: LLM call failed (%s); assuming insufficient.", exc)
            return False

    def compose(
        self,
        query: str,
        index: Index,
        k: int = 10,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        if self._base is None:
            return []

        results = self._base.retrieve(query, index, k=k)

        for round_i in range(1, self._max_rounds + 1):
            context = "\n".join(r.content for r in results)
            try:
                if self._is_sufficient(None, query, context):
                    break
            except NotImplementedError:

                break

            results = self._base.retrieve(query, index, k=k * (round_i + 1))

        return results[:k]
