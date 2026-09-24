"""
StateMemoryBuilder — compresses trajectory into a running state summary via LLM.

Used by C6. Stub: structure correct, LLM call raises NotImplementedError.
"""
from __future__ import annotations

from typing import Any, Optional

from agentmem.retrieval.base import Document, Index, IndexBuilder, IndexUnit

class StateMemoryBuilder(IndexBuilder):
    """
    Compresses a sequence of turns into a state-memory summary.

    Maintains a rolling state that is updated after each turn by the LLM.
    The final state summary plus intermediate snapshots become IndexUnits.

    Args:
        provider: OpenAICompatibleProvider instance.
        model: LLM model name.
        snapshot_every: Create a snapshot IndexUnit every N turns.
    """

    def __init__(
        self,
        provider: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        snapshot_every: int = 10,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._snapshot_every = max(1, snapshot_every)

    _UPDATE_SYSTEM = (
        "You maintain a concise running state summary of a conversation. "
        "Given the current state and a new dialogue turn, produce an updated state summary.\n\n"
        "Rules:\n"
        "- Keep the summary concise (under 300 words)\n"
        "- Include key facts, user preferences, ongoing topics, and notable events\n"
        "- Update or supersede outdated information from the current state\n"
        "- Preserve important historical context\n"
        "Output only the updated state summary text, no preamble."
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

    def _update_state(self, provider: Any, current_state: str, new_turn: str) -> str:
        """Update the running state summary with a new dialogue turn.

        Calls the LLM to incrementally update the state summary by incorporating
        new information from the latest turn.

        Args:
            provider: OpenAICompatibleProvider instance (may be None for lazy init).
            current_state: The current state summary string (may be empty).
            new_turn: The new dialogue turn to integrate.

        Returns:
            Updated state summary string.
        """
        from agentmem.providers.base import Message

        try:
            if provider is None:
                provider = self._get_provider()
        except Exception:
            return (current_state + " " + new_turn).strip()

        if current_state:
            user_content = (
                f"Current state:\n{current_state}\n\n"
                f"New turn:\n{new_turn}\n\n"
                "Updated state:"
            )
        else:
            user_content = f"New turn:\n{new_turn}\n\nInitial state summary:"

        messages = [
            Message(role="system", content=self._UPDATE_SYSTEM),
            Message(role="user", content=user_content),
        ]
        try:
            response = provider.chat(messages, max_tokens=400)
            updated = (response.content or "").strip()
            return updated if updated else (current_state + " " + new_turn).strip()
        except Exception:
            return (current_state + " " + new_turn).strip()

    def build(self, documents: list[Document]) -> Index:
        units: list[IndexUnit] = []
        state = ""

        for i, doc in enumerate(documents):
            try:
                state = self._update_state(None, state, doc.content)
            except NotImplementedError:

                state = (state + " " + doc.content).strip()

            if (i + 1) % self._snapshot_every == 0 or i == len(documents) - 1:
                units.append(
                    IndexUnit(
                        id=f"sm_snap_{i}",
                        content=state,
                        metadata={
                            "snapshot_at_turn": i,
                            "turn_count": i + 1,
                        },
                        source_doc_id=doc.id,
                    )
                )

        return Index(
            units=units,
            metadata={
                "builder": "StateMemoryBuilder",
                "n_snapshots": len(units),
                "n_turns": len(documents),
            },
        )
