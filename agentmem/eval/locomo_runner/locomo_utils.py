"""LoCoMo-specific utilities: category maps, metric wrappers, row I/O."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Optional

from agentmem.eval.locomo_table import per_row_locomo_metrics, summarize_locomo_rows
from agentmem.eval.metrics import compute_metrics

_LOCOMO_NAMED_TASK_CATEGORY_MAP: dict[str, set[int] | None] = {
    "custom": None,
    "multihop": {1},
    "temporal": {2},
    "opendomain": {3},
    "singlehop": {4},
    "adversarial": {5},
    "all4": {1, 2, 3, 4},
    "all5": {1, 2, 3, 4, 5},
}

_LOCOMO_CATEGORY_LABELS: dict[int, str] = {
    1: "multihop",
    2: "temporal",
    3: "opendomain",
    4: "singlehop",
    5: "adversarial",
}

def _parse_categories(raw: str) -> set[int]:
    """Parse a comma-separated category string like ``'1,2,3'`` into a set of ints."""
    values = set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid category token '{token}'. Use integers 1-5.") from exc
    if not values:
        values = {1, 2, 3, 4, 5}
    return values

def _resolve_locomo_category_filter(task: str, fallback_raw: str) -> set[int]:
    """Resolve the category filter from a named task or fallback string."""
    normalized = (task or "custom").strip().lower()
    if normalized not in _LOCOMO_NAMED_TASK_CATEGORY_MAP:
        raise ValueError(
            f"Unknown --locomo-task '{task}'. Valid values: {sorted(_LOCOMO_NAMED_TASK_CATEGORY_MAP.keys())}"
        )
    mapped = _LOCOMO_NAMED_TASK_CATEGORY_MAP[normalized]
    if mapped is not None:
        return set(mapped)
    return _parse_categories(fallback_raw)

def _resolve_max_questions(raw: Optional[int]) -> Optional[int]:
    """Validate and return the max-questions limit, or None for unlimited."""
    if raw is None:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None

def _locomo_reference_answer(qa: Any) -> Optional[str]:
    """Extract the gold answer from a LoCoMo QA pair."""
    category: Optional[int] = None
    try:
        if qa.category is not None:
            category = int(qa.category)
    except Exception:
        category = None

    if category == 5:
        return "Not mentioned in the conversation"

    answer = getattr(qa, "final_answer", None)
    if answer is None:
        return None
    return str(answer)

def _mean_metrics(rows: list[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Compute mean of each metric across rows."""
    metric_names: set[str] = set()
    metric_values: dict[str, list[float]] = {}

    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            metric_names.add(name)
            if isinstance(value, (int, float)):
                metric_values.setdefault(name, []).append(float(value))

    return {
        name: (statistics.mean(metric_values[name]) if metric_values.get(name) else None)
        for name in sorted(metric_names)
    }

def _mean_latency_seconds(rows: list[Dict[str, Any]]) -> Optional[float]:
    """Compute mean latency across rows."""
    values = [
        float(row["latency_seconds"])
        for row in rows
        if isinstance(row.get("latency_seconds"), (int, float))
    ]
    return statistics.mean(values) if values else None

def _locomo_category_breakdown(rows: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group rows by category and compute per-category metrics."""
    grouped: dict[int, list[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("category") is None:
            continue
        try:
            category = int(row["category"])
        except Exception:
            continue
        grouped.setdefault(category, []).append(row)

    result: Dict[str, Dict[str, Any]] = {}
    for category in sorted(grouped.keys()):
        category_rows = grouped[category]
        task_name = _LOCOMO_CATEGORY_LABELS.get(category, f"category_{category}")
        result[task_name] = {
            "category_id": category,
            "num_questions": len(category_rows),
            "num_answered": len([row for row in category_rows if row.get("error") is None]),
            "num_failed": len([row for row in category_rows if row.get("error")]),
            "metrics": _mean_metrics(category_rows),
        }
    return result

def _compute_locomo_metrics(
    prediction: str,
    reference: Optional[str],
    *,
    metric_profile: str = "paper",
    question: Optional[str] = None,
    llm_judge: Optional[Any] = None,
) -> dict[str, float]:
    """Compute all metrics for a single LoCoMo prediction.

    Args:
        prediction: Model prediction string.
        reference: Gold answer string.
        metric_profile: ``"paper"`` for SimpleMem paper-compatible metrics.
        question: The question text (needed for LLM judge).
        llm_judge: Optional LLMJudge instance for model-based scoring.
    """
    if reference is None:
        return {}
    reference_text = str(reference)
    extras = per_row_locomo_metrics(
        prediction=prediction,
        reference=reference_text,
        metric_profile=metric_profile,
    )
    metrics = compute_metrics(
        prediction=prediction,
        reference=reference_text,
        extra_metrics=extras,
    )
    if "rouge_l_f1" in metrics and "rougeL_f" not in metrics:
        metrics["rougeL_f"] = metrics["rouge_l_f1"]

    if llm_judge is not None and question is not None:
        try:
            result = llm_judge.judge(question=question, gold=reference_text, prediction=prediction)
            metrics["llm_judge_correct"] = 1.0 if result.correct else 0.0
            metrics["llm_judge_score"] = result.score
            if result.error:
                metrics["llm_judge_error"] = result.error
        except Exception as exc:
            metrics["llm_judge_error"] = str(exc)

    return metrics

def _locomo_paper_table_summary(
    rows: list[Dict[str, Any]], include_categories: list[int]
) -> dict[str, Any]:
    """Generate a SimpleMem paper-style summary table."""
    filtered = tuple(sorted({c for c in include_categories if c in {1, 2, 3, 4, 5}}))
    if not filtered:
        return {}
    return summarize_locomo_rows(
        rows,
        include_categories=filtered,
        bleu_key="bleu1",
        metric_profile="paper",
    )

def _locomo_rows_path(args: argparse.Namespace, task: str) -> Optional[Path]:
    """Return the JSONL path for a task's rows without writing anything."""
    if not args.locomo_save_rows:
        return None
    out_dir = (args.locomo_rows_dir / args.run_tag).resolve()
    sample_limit = "all" if args.max_samples is None else str(args.max_samples)
    question_limit = _resolve_max_questions(args.max_questions)
    question_limit_tag = "all" if question_limit is None else str(question_limit)
    suffix = f"{args.locomo_task}_s{args.sample_start}_n{sample_limit}_q{question_limit_tag}"
    return out_dir / f"{task.replace('-', '_')}_{suffix}.jsonl"

def _write_locomo_rows(
    *,
    args: argparse.Namespace,
    task: str,
    rows: list[Dict[str, Any]],
) -> Optional[Path]:
    """Write per-question result rows to JSONL."""
    path = _locomo_rows_path(args, task)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
