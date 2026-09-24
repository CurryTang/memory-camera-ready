from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable, Optional

from agentmem.plugmem.session import PlugMemQuestion, PlugMemSession, PlugMemStep

_QA_TYPE_TO_CATEGORY = {"A": 1, "B": 2, "C": 3, "D": 4}

def _sample_metadata(sample: Mapping[str, Any], *, exclude: set[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in sample.items():
        if key in exclude:
            continue
        metadata[str(key)] = value
    return metadata

def _session_sort_key(key: str) -> tuple[int, str]:
    parts = key.split("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1]), key
        except ValueError:
            pass
    return (10**9, key)

def _normalize_text(text: Any) -> str:
    return str(text or "").strip()

def _normalize_locomo_turn(turn: Mapping[str, Any]) -> str:
    text = str(turn.get("text", "") or "")
    caption = turn.get("blip_caption")
    has_image = "img_url" in turn and caption
    if has_image:
        text = f"[Image: {caption}] {text}".strip()
    return text.strip()

def _final_answer_from_qa(qa: Mapping[str, Any], category: Optional[int]) -> Optional[str]:
    if category == 5:
        adversarial = qa.get("adversarial_answer")
        if adversarial is not None:
            return str(adversarial)
    answer = qa.get("answer")
    if answer is None:
        return None
    return str(answer)

def _locomo_goal(sample: Mapping[str, Any]) -> str:
    task = _normalize_text(sample.get("task"))
    if task:
        return task
    return "Answer questions grounded in the conversation."

def _locomo_session_id(sample: Mapping[str, Any]) -> str:
    for key in ("sample_id", "id", "episode_id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "locomo-sample"

def _iter_locomo_turns(sample: Mapping[str, Any]) -> Iterable[tuple[str, Any, Optional[str]]]:
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, Mapping):
        return []

    session_keys = sorted(
        [
            key
            for key, value in conversation.items()
            if str(key).startswith("session_") and isinstance(value, list)
        ],
        key=_session_sort_key,
    )

    ordered: list[tuple[str, Any, Optional[str]]] = []
    for session_key in session_keys:
        timestamp = conversation.get(f"{session_key}_date_time")
        turns = conversation.get(session_key, [])
        for turn in turns:
            ordered.append((str(session_key), turn, timestamp))
    return ordered

def build_plugmem_session_from_locomo(sample: Mapping[str, Any]) -> PlugMemSession:
    """Translate a LoCoMo sample into a canonical PlugMem session."""

    sample_map = dict(sample)
    steps: list[PlugMemStep] = []
    for step_index, (session_key, turn, timestamp) in enumerate(_iter_locomo_turns(sample_map)):
        if not isinstance(turn, Mapping):
            continue
        steps.append(
            PlugMemStep(
                index=step_index,
                speaker=str(turn.get("speaker", "Unknown")),
                action="dialogue_turn",
                observation=_normalize_locomo_turn(turn),
                timestamp=str(timestamp) if timestamp is not None else None,
                metadata={
                    "session_key": session_key,
                    "turn_index": step_index,
                    "source_turn": dict(turn),
                },
            )
        )

    questions: list[PlugMemQuestion] = []
    for question_index, qa in enumerate(sample_map.get("qa", [])):
        if not isinstance(qa, Mapping):
            continue
        category: Optional[int]
        try:
            category = int(qa["category"]) if qa.get("category") is not None else None
        except Exception:
            category = None
        questions.append(
            PlugMemQuestion(
                index=question_index,
                question=_normalize_text(qa.get("question")),
                answer=_final_answer_from_qa(qa, category),
                category=category,
                metadata={
                    "benchmark": "locomo",
                    "question_uuid": str(qa.get("question_uuid", "")),
                    "evidence": [str(item) for item in (qa.get("evidence") or [])],
                    "adversarial_answer": qa.get("adversarial_answer"),
                    "source_question": dict(qa),
                },
            )
        )

    metadata = _sample_metadata(sample_map, exclude={"conversation", "qa"})
    metadata["benchmark"] = "locomo"

    return PlugMemSession(
        session_id=_locomo_session_id(sample_map),
        goal=_locomo_goal(sample_map),
        steps=steps,
        questions=questions,
        metadata=metadata,
    )

def _amabench_session_id(sample: Mapping[str, Any]) -> str:
    episode_id = sample.get("episode_id")
    if episode_id is not None and str(episode_id).strip():
        return f"amabench-{episode_id}"
    for key in ("sample_id", "id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "amabench-sample"

def _amabench_goal(sample: Mapping[str, Any]) -> str:
    task = _normalize_text(sample.get("task"))
    if task:
        return task
    task_type = _normalize_text(sample.get("task_type"))
    if task_type:
        return task_type
    return "Answer questions grounded in the episode trajectory."

def _iter_amabench_trajectory(sample: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    trajectory = sample.get("trajectory", [])
    if isinstance(trajectory, list):
        for turn in trajectory:
            if isinstance(turn, Mapping):
                yield turn

def _iter_amabench_conversation(sample: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any], Optional[str]]]:
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, Mapping):
        return []

    session_keys = sorted(
        [
            key
            for key, value in conversation.items()
            if str(key).startswith("session_") and isinstance(value, list)
        ],
        key=_session_sort_key,
    )
    ordered: list[tuple[str, Mapping[str, Any], Optional[str]]] = []
    for session_key in session_keys:
        timestamp = conversation.get(f"{session_key}_date_time")
        turns = conversation.get(session_key, [])
        for turn in turns:
            if isinstance(turn, Mapping):
                ordered.append((str(session_key), turn, timestamp))
    return ordered

def _qa_type_to_category(qa_type: Any, category: Any) -> Optional[int]:
    if category is not None:
        try:
            return int(category)
        except Exception:
            return None
    if qa_type is None:
        return None
    return _QA_TYPE_TO_CATEGORY.get(str(qa_type).strip().upper())

def build_plugmem_session_from_amabench(sample: Mapping[str, Any]) -> PlugMemSession:
    """Translate an AMABench episode into a canonical PlugMem session."""

    sample_map = dict(sample)
    steps: list[PlugMemStep] = []
    if sample_map.get("trajectory"):
        for step_index, turn in enumerate(_iter_amabench_trajectory(sample_map)):
            steps.append(
                PlugMemStep(
                    index=step_index,
                    speaker=str(turn.get("speaker", "agent")),
                    action=_normalize_text(turn.get("action")),
                    observation=_normalize_text(turn.get("observation")),
                    timestamp=str(turn.get("timestamp")) if turn.get("timestamp") is not None else None,
                    metadata={
                        "turn_idx": turn.get("turn_idx"),
                        "source_turn": dict(turn),
                    },
                )
            )
    else:
        for step_index, (session_key, turn, timestamp) in enumerate(_iter_amabench_conversation(sample_map)):
            steps.append(
                PlugMemStep(
                    index=step_index,
                    speaker=str(turn.get("speaker", "Unknown")),
                    action="dialogue_turn",
                    observation=_normalize_locomo_turn(turn),
                    timestamp=str(timestamp) if timestamp is not None else None,
                    metadata={
                        "session_key": session_key,
                        "turn_index": step_index,
                        "source_turn": dict(turn),
                    },
                )
            )

    questions: list[PlugMemQuestion] = []
    qa_pairs = sample_map.get("qa_pairs", [])
    if not isinstance(qa_pairs, list) or not qa_pairs:
        qa_pairs = sample_map.get("qa", [])
    for question_index, qa in enumerate(qa_pairs):
        if not isinstance(qa, Mapping):
            continue
        qa_type = qa.get("type", qa.get("qa_type"))
        category = _qa_type_to_category(qa_type, qa.get("category"))
        questions.append(
            PlugMemQuestion(
                index=question_index,
                question=_normalize_text(qa.get("question")),
                answer=_final_answer_from_qa(qa, category),
                category=category,
                metadata={
                    "benchmark": "amabench",
                    "qa_type": str(qa_type) if qa_type is not None else None,
                    "question_uuid": str(qa.get("question_uuid", "")),
                    "domain": sample_map.get("domain"),
                    "task_type": sample_map.get("task_type"),
                    "task": sample_map.get("task"),
                    "success": sample_map.get("success"),
                    "source_question": dict(qa),
                },
            )
        )

    metadata = _sample_metadata(sample_map, exclude={"trajectory", "qa_pairs", "qa"})
    metadata["benchmark"] = "amabench"

    return PlugMemSession(
        session_id=_amabench_session_id(sample_map),
        goal=_amabench_goal(sample_map),
        steps=steps,
        questions=questions,
        metadata=metadata,
    )
