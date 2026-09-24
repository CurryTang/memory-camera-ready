"""Structured belief-state memory for MemoryArena Progressive Web Search."""

from __future__ import annotations

import json
import re
from typing import Any

from agentmem.eval.memoryarena_runner import judges
from agentmem.eval.memoryarena_runner.types import MemoryRecord
from agentmem.methods.multi_session import MultiSessionMemory, SessionFeedback

_BRACKET_REF_RE = re.compile(r"\[[0-9,\s]+\]")

def _shorten(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."

class ProgressiveSearchStructuredMemory(MultiSessionMemory):
    """Maintain a compact task state for Progressive Web Search.

    The schema mirrors the paper's POMDP framing: retain accumulated
    constraints, candidate answers, evidence snippets, and rejected/wrong
    hypotheses as sufficient statistics for the next session. This class is
    intentionally deterministic and does not call an LLM; LLM summarizers can
    be layered on top later without changing the harness interface.
    """

    def __init__(self, *, max_evidence_per_candidate: int = 3) -> None:
        self.task_id: str | None = None
        self.max_evidence_per_candidate = max_evidence_per_candidate
        self.state: dict[str, Any] = {}
        self.raw_records: list[MemoryRecord] = []

    def reset(self, task_id: str, *, schema_hint: str | None = None) -> None:
        self.task_id = str(task_id)
        self.state = {
            "domain": schema_hint or "progressive_search",
            "accumulated_constraints": [],
            "candidate_entities": [],
            "rejected_entities": [],
            "session_outcomes": [],
        }
        self.raw_records = []

    def retrieve(self, query: str, *, session_id: int, k: int = 5) -> str:
        if not self.raw_records:
            return "No previous memory."
        return "Structured task memory:\n" + json.dumps(
            self.state,
            ensure_ascii=False,
            indent=2,
        )

    def update(self, fb: SessionFeedback | None = None, **kwargs: Any) -> None:
        if fb is None:
            feedback = dict(kwargs.get("feedback") or {})
            fb = SessionFeedback(
                session_id=int(kwargs["session_id"]),
                question=str(kwargs["question"]),
                prediction=str(kwargs["prediction"]),
                correct=bool(feedback.get("correct")),
                judge_signal=feedback,
                trace=kwargs.get("trace"),
            )
        self._update_state(fb)
        self.raw_records.append(
            MemoryRecord(
                task_id=str(self.task_id or ""),
                session_id=fb.session_id,
                kind="structured_session",
                content=fb.trace or fb.prediction,
                metadata={"correct": fb.correct},
            )
        )

    def _update_state(self, fb: SessionFeedback) -> None:
        constraint = _shorten(fb.question, 300)
        if constraint and constraint not in self.state["accumulated_constraints"]:
            self.state["accumulated_constraints"].append(constraint)

        exact = (
            fb.judge_signal.get("pred_exact")
            or judges.extract_exact_answer(fb.prediction)
            or _fallback_answer(fb.prediction)
        )
        evidence = _extract_evidence(fb.prediction or fb.trace or "")
        if exact:
            bucket = "candidate_entities" if fb.correct is not False else "rejected_entities"
            self._upsert_entity(bucket, exact, fb.session_id, constraint, evidence)

        self.state["session_outcomes"].append(
            {
                "session": fb.session_id,
                "correct": fb.correct,
                "predicted_answer": exact,
            }
        )

    def _upsert_entity(
        self,
        bucket: str,
        name: str,
        session_id: int,
        constraint: str,
        evidence: list[str],
    ) -> None:
        entities = self.state[bucket]
        normalized = name.lower()
        existing = next(
            (item for item in entities if str(item.get("name", "")).lower() == normalized),
            None,
        )
        if existing is None:
            existing = {
                "name": name,
                "first_seen_session": session_id,
                "satisfies": [],
                "evidence": [],
            }
            entities.append(existing)
        if constraint not in existing["satisfies"]:
            existing["satisfies"].append(constraint)
        for item in evidence:
            if item not in existing["evidence"]:
                existing["evidence"].append(item)
        existing["evidence"] = existing["evidence"][: self.max_evidence_per_candidate]

def _fallback_answer(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    tail = lines[-1]
    tail = re.sub(r"^(?:final answer|answer)\s*:\s*", "", tail, flags=re.IGNORECASE)
    return _shorten(tail, 160)

def _extract_evidence(text: str) -> list[str]:
    evidence: list[str] = []
    for raw in re.split(r"(?<=[.!?])\s+", text):
        sentence = _shorten(raw, 240)
        if not sentence:
            continue
        if _BRACKET_REF_RE.search(sentence) or "because" in sentence.lower():
            evidence.append(sentence)
        if len(evidence) >= 3:
            break
    return evidence
