"""Simple session-buffer memory for MemoryArena."""

from __future__ import annotations

from typing import Any

from agentmem.eval.memoryarena_runner.types import MemoryRecord
from agentmem.methods.multi_session import MultiSessionMemory, SessionFeedback

class BufferMemory(MultiSessionMemory):
    """Long-context buffer baseline.

    This is the minimal reproducible memory system for the offline
    multi-session text-agent setup: retrieve recent completed sessions, then
    append one record after the current session receives feedback.
    """

    def __init__(self, *, max_records: int | None = None, max_chars: int | None = None) -> None:
        self.task_id: str | None = None
        self.records: list[MemoryRecord] = []
        self.max_records = max_records
        self.max_chars = max_chars

    def reset(self, task_id: str, *, schema_hint: str | None = None) -> None:
        self.task_id = str(task_id)
        self.records = []

    def retrieve(self, query: str, *, session_id: int, k: int = 5) -> str:
        if not self.records:
            return "No previous memory."
        limit = self.max_records if self.max_records is not None else k
        selected = self.records[-limit:]
        text = "\n\n".join(
            f"[Session {r.session_id} | {r.kind}]\n{r.content}" for r in selected
        )
        if self.max_chars is not None and len(text) > self.max_chars:
            text = text[-self.max_chars :]
        return text

    def update(self, fb: SessionFeedback | None = None, **kwargs: Any) -> None:
        if fb is None:
            fb = SessionFeedback(
                session_id=int(kwargs["session_id"]),
                question=str(kwargs["question"]),
                prediction=str(kwargs["prediction"]),
                correct=bool((kwargs.get("feedback") or {}).get("correct")),
                judge_signal=dict(kwargs.get("feedback") or {}),
                trace=kwargs.get("trace"),
            )
        content = (
            f"Question:\n{fb.question}\n\n"
            f"Prediction:\n{fb.prediction}\n\n"
            f"Feedback:\n{{'correct': {fb.correct}, 'signal': {fb.judge_signal}}}\n\n"
            f"Trace:\n{fb.trace or ''}"
        )
        self.records.append(
            MemoryRecord(
                task_id=str(self.task_id or ""),
                session_id=fb.session_id,
                kind="raw_session_trace",
                content=content,
                metadata={"correct": fb.correct},
            )
        )
