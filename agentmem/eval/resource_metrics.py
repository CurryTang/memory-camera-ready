"""Utilities for summarizing runtime and token usage from eval trajectories.

These helpers are intentionally tolerant of partial / heterogeneous trajectory
payloads. They prefer explicit token-usage dicts when present, then fall back
to estimating tokens from saved prompt/response text, and finally report a
lower-bound estimate from question/prediction text when no better signal is
available.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

TokenCounter = Callable[[str], int]

_USAGE_KEYS = {
    "usage",
    "token_usage",
    "llm_usage",
    "answer_usage",
    "model_usage",
}
_PROMPT_KEYS = (
    "qa_prompt",
    "prompt",
    "input_context",
    "user_prompt",
)
_COMPLETION_KEYS = (
    "llm_response",
    "response",
    "completion",
    "completion_text",
)
_LOWER_BOUND_COMPLETION_KEYS = (
    "prediction",
    "answer",
    "content",
)
_QUESTION_TRAJ_KEYS = {
    "question",
    "prediction",
    "mode",
    "retrieve_sec",
    "llm_sec",
    "latency_seconds",
}
_SOURCE_TRAJ_KEYS = {
    "action",
    "observation",
    "turn_idx",
    "step",
}

def estimate_text_tokens(
    text: Any,
    *,
    model: Optional[str] = None,
    encoding_name: Optional[str] = None,
    chars_per_token: float = 4.0,
    token_counter: Optional[TokenCounter] = None,
) -> int:
    """Estimate token count for arbitrary text.

    Order of preference:
    1. Caller-provided ``token_counter``.
    2. ``tiktoken`` with the requested model / encoding.
    3. Character-length fallback using ``chars_per_token``.
    """
    raw = "" if text is None else str(text)
    if not raw:
        return 0
    if token_counter is not None:
        return max(0, int(token_counter(raw)))

    try:
        import tiktoken

        if encoding_name:
            encoding = tiktoken.get_encoding(encoding_name)
        elif model:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(raw))
    except Exception:
        ratio = max(float(chars_per_token), 1e-6)
        return int(math.ceil(len(raw) / ratio))

def estimate_trajectory_tokens(
    trajectory: Any,
    *,
    model: Optional[str] = None,
    encoding_name: Optional[str] = None,
    chars_per_token: float = 4.0,
    token_counter: Optional[TokenCounter] = None,
) -> int:
    """Estimate tokens required to serialize a source trajectory."""
    if isinstance(trajectory, str):
        return estimate_text_tokens(
            trajectory,
            model=model,
            encoding_name=encoding_name,
            chars_per_token=chars_per_token,
            token_counter=token_counter,
        )
    if not isinstance(trajectory, Sequence):
        return 0

    lines: list[str] = []
    for idx, step in enumerate(trajectory):
        if isinstance(step, Mapping):
            turn_idx = step.get("turn_idx", step.get("step", idx))
            action = step.get("action")
            observation = step.get("observation")
            text = step.get("text")
            content = step.get("content")
            if action is not None or observation is not None:
                lines.append(f"Turn {turn_idx}:")
                if action not in (None, ""):
                    lines.append(f"Action: {action}")
                if observation not in (None, ""):
                    lines.append(f"Observation: {observation}")
                continue
            if text not in (None, ""):
                lines.append(f"Turn {turn_idx}: {text}")
                continue
            if content not in (None, ""):
                lines.append(f"Turn {turn_idx}: {content}")
                continue
        elif step not in (None, ""):
            lines.append(str(step))

    return estimate_text_tokens(
        "\n".join(lines),
        model=model,
        encoding_name=encoding_name,
        chars_per_token=chars_per_token,
        token_counter=token_counter,
    )

def summarize_resource_usage(
    record: Any,
    *,
    model: Optional[str] = None,
    encoding_name: Optional[str] = None,
    chars_per_token: float = 4.0,
    token_counter: Optional[TokenCounter] = None,
) -> dict[str, Any]:
    """Summarize runtime and token usage from an eval record or trajectory.

    Supported shapes include:
    - AMABench episode records returned by ``examples/run_amabench.py``.
    - Per-question rows with ``latency_seconds`` and nested ``trajectory``.
    - Raw source trajectories (list of action/observation turns).
    """
    question_entries = _extract_question_entries(record)
    source_trajectory = _extract_source_trajectory(record)

    memory_construction_seconds = _as_float(
        _mapping_get(record, "memory_construction_sec"),
        default=0.0,
    )
    retrieval_seconds = 0.0
    generation_seconds = 0.0
    question_seconds = 0.0

    prompt_tokens = 0
    completion_tokens = 0
    retrieved_context_tokens = 0
    explicit_usage_questions = 0
    estimated_prompt_response_questions = 0
    lower_bound_questions = 0
    missing_token_questions = 0

    if question_entries:
        for entry in question_entries:
            retrieval_seconds += _as_float(entry.get("retrieve_sec"), default=0.0)
            generation_seconds += _as_float(entry.get("llm_sec"), default=0.0)
            latency = _as_float(entry.get("latency_seconds"))
            if latency is not None:
                question_seconds += latency

            retrieved_context_tokens += estimate_text_tokens(
                entry.get("retrieved_context", ""),
                model=model,
                encoding_name=encoding_name,
                chars_per_token=chars_per_token,
                token_counter=token_counter,
            )

            usage = _sum_usage_dicts(_collect_usage_dicts(entry))
            if usage["prompt_tokens"] or usage["completion_tokens"] or usage["total_tokens"]:
                prompt_tokens += usage["prompt_tokens"]
                completion_tokens += usage["completion_tokens"]
                explicit_usage_questions += 1
                continue

            prompt_text = _first_text(entry, _PROMPT_KEYS)
            completion_text = _first_text(entry, _COMPLETION_KEYS)
            if prompt_text or completion_text:
                prompt_tokens += estimate_text_tokens(
                    prompt_text,
                    model=model,
                    encoding_name=encoding_name,
                    chars_per_token=chars_per_token,
                    token_counter=token_counter,
                )
                completion_tokens += estimate_text_tokens(
                    completion_text,
                    model=model,
                    encoding_name=encoding_name,
                    chars_per_token=chars_per_token,
                    token_counter=token_counter,
                )
                estimated_prompt_response_questions += 1
                continue

            lower_bound_prompt = "\n".join(
                str(part)
                for part in (
                    entry.get("retrieved_context", ""),
                    entry.get("question", ""),
                )
                if part not in (None, "")
            )
            lower_bound_completion = _first_text(entry, _LOWER_BOUND_COMPLETION_KEYS)
            if lower_bound_prompt or lower_bound_completion:
                prompt_tokens += estimate_text_tokens(
                    lower_bound_prompt,
                    model=model,
                    encoding_name=encoding_name,
                    chars_per_token=chars_per_token,
                    token_counter=token_counter,
                )
                completion_tokens += estimate_text_tokens(
                    lower_bound_completion,
                    model=model,
                    encoding_name=encoding_name,
                    chars_per_token=chars_per_token,
                    token_counter=token_counter,
                )
                lower_bound_questions += 1
            else:
                missing_token_questions += 1
    else:
        usage = _sum_usage_dicts(_collect_usage_dicts(record))
        prompt_tokens += usage["prompt_tokens"]
        completion_tokens += usage["completion_tokens"]

    if question_entries and question_seconds == 0.0:
        question_seconds = retrieval_seconds + generation_seconds

    if question_seconds == 0.0:
        question_seconds = _as_float(
            _mapping_get(record, "latency_seconds"),
            default=0.0,
        )
    if question_seconds == 0.0:
        question_seconds = _as_float(
            _mapping_get(record, "episode_duration_sec"),
            default=0.0,
        )
    if question_seconds == 0.0:
        question_seconds = _as_float(
            _mapping_get(record, "duration_sec"),
            default=0.0,
        )

    construction_input_tokens = _extract_construction_tokens(
        record,
        source_trajectory=source_trajectory,
        model=model,
        encoding_name=encoding_name,
        chars_per_token=chars_per_token,
        token_counter=token_counter,
    )

    total_llm_tokens = prompt_tokens + completion_tokens
    total_runtime_seconds = memory_construction_seconds + question_seconds

    explicit_usage_detected = bool(explicit_usage_questions)
    if not explicit_usage_detected and not question_entries:
        explicit_usage_detected = bool(_collect_usage_dicts(record))

    return {
        "num_questions": len(question_entries),
        "memory_construction_seconds": memory_construction_seconds,
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": generation_seconds,
        "question_seconds": question_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "construction_input_tokens": construction_input_tokens,
        "retrieved_context_tokens": retrieved_context_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_llm_tokens": total_llm_tokens,
        "estimated_total_tokens": construction_input_tokens + total_llm_tokens,
        "questions_with_explicit_usage": explicit_usage_questions,
        "questions_with_prompt_response_estimate": estimated_prompt_response_questions,
        "questions_with_lower_bound_estimate": lower_bound_questions,
        "questions_missing_token_signal": missing_token_questions,
        "has_explicit_token_usage": explicit_usage_detected,
        "used_dataset_total_tokens": bool(_as_int(_mapping_get(record, "total_tokens"))),
    }

def summarize_resource_usage_by_method(
    records: Iterable[Mapping[str, Any]],
    *,
    method_key: str = "method",
    model: Optional[str] = None,
    encoding_name: Optional[str] = None,
    chars_per_token: float = 4.0,
    token_counter: Optional[TokenCounter] = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate runtime / token summaries across multiple methods."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        method = str(record.get(method_key, "unknown"))
        grouped[method].append(
            summarize_resource_usage(
                record,
                model=model,
                encoding_name=encoding_name,
                chars_per_token=chars_per_token,
                token_counter=token_counter,
            )
        )

    result: dict[str, dict[str, Any]] = {}
    for method, summaries in grouped.items():
        num_records = len(summaries)
        num_questions = sum(int(summary["num_questions"]) for summary in summaries)
        total_runtime = sum(float(summary["total_runtime_seconds"]) for summary in summaries)
        total_tokens = sum(int(summary["estimated_total_tokens"]) for summary in summaries)
        result[method] = {
            "num_records": num_records,
            "num_questions": num_questions,
            "memory_construction_seconds": sum(
                float(summary["memory_construction_seconds"]) for summary in summaries
            ),
            "retrieval_seconds": sum(float(summary["retrieval_seconds"]) for summary in summaries),
            "generation_seconds": sum(float(summary["generation_seconds"]) for summary in summaries),
            "question_seconds": sum(float(summary["question_seconds"]) for summary in summaries),
            "total_runtime_seconds": total_runtime,
            "construction_input_tokens": sum(
                int(summary["construction_input_tokens"]) for summary in summaries
            ),
            "prompt_tokens": sum(int(summary["prompt_tokens"]) for summary in summaries),
            "completion_tokens": sum(int(summary["completion_tokens"]) for summary in summaries),
            "total_llm_tokens": sum(int(summary["total_llm_tokens"]) for summary in summaries),
            "estimated_total_tokens": total_tokens,
            "avg_runtime_seconds_per_record": (
                total_runtime / num_records if num_records else 0.0
            ),
            "avg_tokens_per_record": (
                total_tokens / num_records if num_records else 0.0
            ),
            "avg_tokens_per_question": (
                total_tokens / num_questions if num_questions else 0.0
            ),
            "questions_with_explicit_usage": sum(
                int(summary["questions_with_explicit_usage"]) for summary in summaries
            ),
            "questions_with_prompt_response_estimate": sum(
                int(summary["questions_with_prompt_response_estimate"]) for summary in summaries
            ),
            "questions_with_lower_bound_estimate": sum(
                int(summary["questions_with_lower_bound_estimate"]) for summary in summaries
            ),
            "questions_missing_token_signal": sum(
                int(summary["questions_missing_token_signal"]) for summary in summaries
            ),
        }
    return result

def _extract_construction_tokens(
    record: Any,
    *,
    source_trajectory: Any,
    model: Optional[str],
    encoding_name: Optional[str],
    chars_per_token: float,
    token_counter: Optional[TokenCounter],
) -> int:
    explicit_total = _as_int(_mapping_get(record, "total_tokens"))
    if explicit_total is not None and explicit_total > 0:
        return explicit_total

    trajectory_text = _mapping_get(record, "trajectory_text") or _mapping_get(record, "traj_text")
    if trajectory_text not in (None, ""):
        return estimate_text_tokens(
            trajectory_text,
            model=model,
            encoding_name=encoding_name,
            chars_per_token=chars_per_token,
            token_counter=token_counter,
        )

    return estimate_trajectory_tokens(
        source_trajectory,
        model=model,
        encoding_name=encoding_name,
        chars_per_token=chars_per_token,
        token_counter=token_counter,
    )

def _extract_question_entries(record: Any) -> list[Mapping[str, Any]]:
    if isinstance(record, Sequence) and not isinstance(record, (str, bytes, bytearray)):
        items = [item for item in record if isinstance(item, Mapping)]
        if items and any(_looks_like_question_entry(item) for item in items):
            return items
        return []

    if not isinstance(record, Mapping):
        return []

    if _looks_like_question_entry(record):
        return [record]

    for key in ("question_rows", "rows", "qa_trajectory", "qa_traj"):
        value = record.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]

    trajectory = record.get("trajectory")
    if isinstance(trajectory, list) and any(
        _looks_like_question_entry(item) for item in trajectory if isinstance(item, Mapping)
    ):
        return [item for item in trajectory if isinstance(item, Mapping)]

    return []

def _extract_source_trajectory(record: Any) -> Any:
    if isinstance(record, Sequence) and not isinstance(record, (str, bytes, bytearray)):
        items = [item for item in record if isinstance(item, Mapping)]
        if items and not any(_looks_like_question_entry(item) for item in items):
            return list(record)
        return []

    if not isinstance(record, Mapping):
        return []

    for key in ("source_trajectory", "episode_trajectory", "input_trajectory", "raw_trajectory"):
        value = record.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return value

    trajectory = record.get("trajectory")
    if isinstance(trajectory, list) and any(
        _looks_like_source_step(item) for item in trajectory if isinstance(item, Mapping)
    ):
        return trajectory

    return []

def _collect_usage_dicts(obj: Any) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    seen: set[int] = set()

    def _walk(value: Any) -> None:
        if not isinstance(value, (Mapping, list, tuple)):
            return
        marker = id(value)
        if marker in seen:
            return
        seen.add(marker)

        if isinstance(value, Mapping):
            if _looks_like_usage_dict(value):
                out.append(value)
            for key, child in value.items():
                if key in _USAGE_KEYS and isinstance(child, Mapping) and _looks_like_usage_dict(child):
                    out.append(child)
                    continue
                _walk(child)
            return

        for child in value:
            _walk(child)

    _walk(obj)
    return out

def _sum_usage_dicts(usages: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for usage in usages:
        prompt_tokens += _as_int(usage.get("prompt_tokens"), default=0)
        completion_tokens += _as_int(usage.get("completion_tokens"), default=0)
        total_tokens += _as_int(usage.get("total_tokens"), default=0)

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens > 0:
        prompt_tokens = total_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

def _looks_like_question_entry(item: Mapping[str, Any]) -> bool:
    return bool(_QUESTION_TRAJ_KEYS & set(item.keys()))

def _looks_like_source_step(item: Mapping[str, Any]) -> bool:
    return bool(_SOURCE_TRAJ_KEYS & set(item.keys()))

def _looks_like_usage_dict(item: Mapping[str, Any]) -> bool:
    return any(key in item for key in ("prompt_tokens", "completion_tokens", "total_tokens"))

def _first_text(item: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""

def _mapping_get(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    return None

def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default

def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except Exception:
        return default
