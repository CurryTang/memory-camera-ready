"""LongContext multi-session adapter — reference implementation.

Stores every session trace verbatim and renders the concatenation as memory
context for the next session. No retrieval, no compression: the upper bound
on what the LLM can ingest in its context window.
"""

from __future__ import annotations

from agentmem.methods.multi_session import MultiSessionMemory, SessionFeedback

class LongContextAdapter(MultiSessionMemory):
    def __init__(self, *, max_chars: int | None = None) -> None:
        self._task_id: str | None = None
        self._records: list[str] = []
        self._max_chars = max_chars

    def reset(self, task_id: str, *, schema_hint: str | None = None) -> None:
        self._task_id = task_id
        self._records = []

    def retrieve(self, query: str, *, session_id: int, k: int = 5) -> str:
        if not self._records:
            return "No prior memory."
        text = "\n\n".join(self._records)
        if self._max_chars is not None and len(text) > self._max_chars:
            text = text[-self._max_chars :]
        return text

    def update(self, fb: SessionFeedback) -> None:
        record = (
            f"[Session {fb.session_id}] Q: {fb.question}\n"
            f"Predicted: {fb.prediction}\n"
            f"Correct: {fb.correct}\n"
            f"Trace: {fb.trace or '(no trace)'}"
        )
        self._records.append(record)
