"""Shared runner for LoCoMo and AMABench baseline tasks.

The _run_baseline_locomo function is the universal eval loop used by
all C1-C9 baselines, LongContext, HippoRAGv2, and memory system adapters.
"""

from __future__ import annotations

import json
import inspect
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from data import DatasetFactory

from agentmem.eval.locomo_runner.types import TaskResult
from agentmem.eval.locomo_runner.provider import _shared_provider_kwargs, _history_artifact_path
from agentmem.eval.locomo_runner.prompts import _normalize_locomo_prediction, _coerce_int_category
from agentmem.eval.locomo_runner.io_utils import _write_jsonl
from agentmem.eval.locomo_runner.locomo_utils import (
    _resolve_locomo_category_filter,
    _resolve_max_questions,
    _locomo_reference_answer,
    _locomo_rows_path,
    _compute_locomo_metrics,
    _locomo_category_breakdown,
    _locomo_paper_table_summary,
    _mean_latency_seconds,
    _mean_metrics,
)

def _call_optional_hook(adapter: Any, method_name: str, *, sample_id: Optional[str] = None) -> Any:
    method = getattr(adapter, method_name, None)
    if not callable(method):
        return None
    if sample_id is None:
        return method()
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method()
    if "sample_id" in signature.parameters:
        return method(sample_id=sample_id)
    return method()

def _sum_numeric_fields(target: dict[str, Any], source: Optional[dict[str, Any]]) -> None:
    if not source:
        return
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[key] = float(target.get(key, 0.0)) + float(value)
        elif key not in target:
            target[key] = value

def _sample_resource_row(
    *,
    sample_index: int,
    num_questions: int,
    num_answered: int,
    num_failed: int,
    build_resource: Optional[dict[str, Any]],
    question_rows: list[Dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not build_resource:
        build_resource = {}
    question_resource_rows = [
        row.get("resource_usage")
        for row in question_rows
        if isinstance(row.get("resource_usage"), dict)
    ]
    if not build_resource and not question_resource_rows:
        return None

    row: dict[str, Any] = {
        "sample_index": sample_index,
        "num_questions": num_questions,
        "num_answered": num_answered,
        "num_failed": num_failed,
    }
    _sum_numeric_fields(row, build_resource)

    question_seconds = 0.0
    prompt_tokens = 0.0
    completion_tokens = 0.0
    total_tokens = 0.0
    retrieved_context_tokens = 0.0
    retrieval_calls = 0.0
    llm_calls = 0.0

    for q_row in question_rows:
        latency = q_row.get("latency_seconds")
        if isinstance(latency, (int, float)):
            question_seconds += float(latency)
        usage = q_row.get("resource_usage")
        if not isinstance(usage, dict):
            continue
        prompt_tokens += float(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += float(usage.get("completion_tokens", 0) or 0)
        total_tokens += float(usage.get("total_tokens", 0) or 0)
        retrieved_context_tokens += float(usage.get("retrieved_context_tokens", 0) or 0)
        retrieval_calls += float(usage.get("retrieval_calls", 0) or 0)
        llm_calls += float(usage.get("llm_calls", 0) or 0)

    row["question_seconds"] = question_seconds
    row["total_runtime_seconds"] = float(row.get("build_wallclock_seconds", 0.0) or 0.0) + question_seconds
    row["prompt_tokens"] = prompt_tokens
    row["completion_tokens"] = completion_tokens
    row["total_llm_tokens"] = prompt_tokens + completion_tokens
    row["total_tokens"] = total_tokens
    row["construction_input_tokens"] = float(row.get("construction_input_tokens", 0.0) or 0.0)
    row["estimated_total_tokens"] = row["construction_input_tokens"] + row["total_llm_tokens"]
    row["retrieved_context_tokens"] = retrieved_context_tokens
    row["retrieval_calls"] = retrieval_calls
    row["llm_calls"] = llm_calls or float(row.get("llm_calls", 0.0) or 0.0)
    return row

def _aggregate_sample_resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_samples": len(rows)}
    ignored_numeric_keys = {"sample_index"}
    for row in rows:
        for key, value in row.items():
            if key in ignored_numeric_keys:
                continue
            if isinstance(value, (int, float)):
                summary[key] = float(summary.get(key, 0.0)) + float(value)
    return summary

def _is_sensitive_summary_key(key: str) -> bool:
    lowered = str(key).lower()
    return (
        "api_key" in lowered
        or lowered.endswith("_secret")
        or lowered == "secret"
        or lowered.endswith("_password")
        or lowered == "password"
        or lowered.endswith("_token")
        or lowered == "token"
    )

def _sanitize_summary_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if _is_sensitive_summary_key(str(key))
                else _sanitize_summary_value(nested)
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_summary_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_summary_value(item) for item in value)
    return value

def _sanitize_summary_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): (
            "<redacted>"
            if _is_sensitive_summary_key(str(key))
            else _sanitize_summary_value(val)
        )
        for key, val in values.items()
    }

def _run_baseline_locomo(
    task: str,
    adapter_cls: type,
    adapter_kwargs: dict[str, Any],
    args: Any,
    index_cache: Optional[Any] = None,
) -> TaskResult:
    """Shared runner for all LoCoMo tasks (C1-C9, LongContext, HippoRAGv2, etc.).

    Args:
        task: Task identifier string.
        adapter_cls: Adapter class to instantiate.
        adapter_kwargs: Kwargs passed to adapter constructor.
        args: Parsed CLI arguments.
        index_cache: Optional IndexDiskCache for sharing indexes across tasks.
    """
    start = time.perf_counter()

    try:
        dataset = DatasetFactory.create("locomo", datasets_dir=args.datasets_dir)
        dataset_path = dataset.ensure_data(
            path=args.dataset_path,
            variant=args.locomo_variant,
            auto_download=args.download,
        )
    except Exception as exc:
        return TaskResult(
            task=task, status="failed", duration_sec=0.0,
            summary={"reason": "dataset load failure"}, error=str(exc),
        )

    raw_samples = dataset.load_samples(dataset_path)
    if args.sample_start < 0 or args.sample_start >= len(raw_samples):
        return TaskResult(
            task=task, status="failed", duration_sec=0.0,
            summary={"reason": "invalid sample_start"},
            error=f"sample_start={args.sample_start} out of range for {len(raw_samples)} samples.",
        )

    stop = (
        len(raw_samples)
        if args.max_samples is None
        else min(len(raw_samples), args.sample_start + args.max_samples)
    )
    samples = raw_samples[args.sample_start:stop]
    if not samples:
        return TaskResult(
            task=task, status="skipped", duration_sec=0.0,
            summary={"reason": "no samples selected"},
        )

    category_filter = _resolve_locomo_category_filter(args.locomo_task, args.locomo_categories)
    max_questions = _resolve_max_questions(args.max_questions)

    try:
        adapter_init_kwargs: dict[str, Any] = {
            "provider_kwargs": _shared_provider_kwargs(args, task=task),
            **adapter_kwargs,
        }
        if index_cache is not None:
            adapter_init_kwargs["index_cache"] = index_cache
        adapter = adapter_cls(**adapter_init_kwargs)
    except Exception as exc:
        return TaskResult(
            task=task, status="failed", duration_sec=0.0,
            summary={"reason": f"{task} init failed"}, error=str(exc),
        )

    rows_path = _locomo_rows_path(args, task)
    question_rows: list[Dict[str, Any]] = []
    sample_resource_rows: list[dict[str, Any]] = []
    completed_keys: set[tuple] = set()
    if rows_path is not None:
        rows_dir = rows_path.parent
        task_prefix = f"{task.replace('-', '_')}_{args.locomo_task}_"
        if rows_dir.exists():
            for candidate in sorted(rows_dir.glob(f"{task_prefix}*.jsonl")):
                with candidate.open(encoding="utf-8") as _rf:
                    for _line in _rf:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _row = json.loads(_line)
                            if candidate == rows_path:
                                question_rows.append(_row)
                            if _row.get("question") is not None:
                                completed_keys.add((_row["sample_index"], _row["question"]))
                        except Exception:
                            pass
        if completed_keys:
            print(
                f"[resume] {task}: {len(completed_keys)} questions already answered "
                f"({len(question_rows)} in current file) — skipping those.",
                flush=True,
            )

    _rows_fh: Optional[Any] = None
    if rows_path is not None:
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        _rows_fh = rows_path.open("a", encoding="utf-8")

    artifacts: list[Path] = []
    history_artifact = _history_artifact_path(args)
    if history_artifact is not None:
        artifacts.append(history_artifact)

    llm_judge = None
    judge_model = getattr(args, "llm_judge_model", None)
    if judge_model:
        from agentmem.eval.llm_judge import LLMJudge
        llm_judge = LLMJudge(
            model=judge_model,
            mode=getattr(args, "llm_judge_mode", "locomo"),
            api_key=getattr(args, "llm_judge_api_key", None),
            base_url=getattr(args, "llm_judge_base_url", None),
        )

    try:
        for local_index, sample in enumerate(samples):
            sample_id = args.sample_start + local_index
            try:
                build_resource: Optional[dict[str, Any]] = None
                _call_optional_hook(adapter, "reset", sample_id=f"sample_{sample_id:04d}")
                ingest_started = time.perf_counter()
                for dialogue in dataset.iter_dialogues(sample):
                    prefix = f"[{dialogue.speaker}]"
                    if dialogue.timestamp:
                        prefix += f" [time={dialogue.timestamp}]"
                    adapter.observe(f"{prefix} {dialogue.content}", timestamp=dialogue.timestamp)
                _call_optional_hook(adapter, "finalize", sample_id=f"sample_{sample_id:04d}")
                ingest_seconds = time.perf_counter() - ingest_started

                build_resource = _call_optional_hook(adapter, "sample_resource_usage")
                sample_question_rows: list[Dict[str, Any]] = []
                q_count = 0
                for qa in dataset.iter_qa_pairs(sample):
                    if max_questions is not None and q_count >= max_questions:
                        break
                    if qa.category is not None and int(qa.category) not in category_filter:
                        continue

                    q_count += 1

                    if (sample_id, qa.question) in completed_keys:
                        continue

                    gold = _locomo_reference_answer(qa)
                    q_start = time.perf_counter()
                    question_result: Dict[str, Any] = {
                        "sample_index": sample_id,
                        "question": qa.question,
                        "category": qa.category,
                        "gold": gold,
                        "prediction": None,
                        "error": None,
                        "latency_seconds": None,
                        "metrics": None,
                    }
                    try:
                        pred = adapter.ask(qa.question, category=qa.category)
                        prediction = _normalize_locomo_prediction(
                            pred,
                            category=_coerce_int_category(qa.category),
                            question=qa.question,
                        )
                        question_result["prediction"] = prediction
                        question_result["latency_seconds"] = time.perf_counter() - q_start
                        if gold is not None:
                            question_result["metrics"] = _compute_locomo_metrics(
                                prediction=prediction,
                                reference=gold,
                                metric_profile=args.locomo_metric_profile,
                                question=qa.question,
                                llm_judge=llm_judge,
                            )
                        traj = adapter.last_trajectory() if hasattr(adapter, "last_trajectory") else None
                        if traj is not None:
                            question_result["trajectory"] = traj
                        resource_usage = _call_optional_hook(adapter, "last_resource_usage")
                        if resource_usage is not None:
                            question_result["resource_usage"] = resource_usage

                        canonical_eff = _call_optional_hook(adapter, "canonical_efficiency")
                        if canonical_eff is not None:
                            question_result["efficiency"] = canonical_eff
                        observe_outcome = getattr(adapter, "observe_outcome", None)
                        if callable(observe_outcome):
                            observe_outcome(
                                question=qa.question,
                                prediction=prediction,
                                gold=gold,
                                category=_coerce_int_category(qa.category),
                                metrics=question_result.get("metrics"),
                            )
                        question_rows.append(question_result)
                        sample_question_rows.append(question_result)
                    except Exception as exc:
                        question_result["error"] = str(exc)
                        question_result["latency_seconds"] = time.perf_counter() - q_start
                        question_rows.append(question_result)
                        sample_question_rows.append(question_result)

                    if _rows_fh is not None:
                        _rows_fh.write(json.dumps(question_result, ensure_ascii=False) + "\n")
                        _rows_fh.flush()

            except Exception as exc:
                _fail_row: Dict[str, Any] = {
                    "sample_index": sample_id,
                    "question": None, "category": None, "gold": None,
                    "prediction": None, "error": f"sample-level failure: {exc}",
                    "latency_seconds": None, "metrics": None,
                }
                question_rows.append(_fail_row)
                sample_question_rows.append(_fail_row)
                if _rows_fh is not None:
                    _rows_fh.write(json.dumps(_fail_row, ensure_ascii=False) + "\n")
                    _rows_fh.flush()
            finally:
                sample_resource_row = _sample_resource_row(
                    sample_index=sample_id,
                    num_questions=q_count,
                    num_answered=len([r for r in sample_question_rows if r.get("error") is None]),
                    num_failed=len([r for r in sample_question_rows if r.get("error")]),
                    build_resource=build_resource,
                    question_rows=sample_question_rows,
                )
                if sample_resource_row is not None:
                    sample_resource_rows.append(sample_resource_row)
    finally:
        if _rows_fh is not None:
            _rows_fh.close()

    adapter.shutdown()
    rows_file = rows_path
    if rows_file is not None:
        artifacts.append(rows_file)
    resources_file = None
    if rows_file is not None and sample_resource_rows:
        resources_file = rows_file.with_name(rows_file.stem + "_resources.jsonl")
        _write_jsonl(resources_file, sample_resource_rows)
        artifacts.append(resources_file)

    duration = time.perf_counter() - start
    summary = {
        "dataset_path": str(dataset_path),
        "num_samples": len(samples),
        "num_questions": len(question_rows),
        "num_answered": len([r for r in question_rows if r.get("error") is None]),
        "num_failed": len([r for r in question_rows if r.get("error")]),
        "run_tag": args.run_tag,
        "locomo_task": args.locomo_task,
        "categories_included": sorted(category_filter),
        "max_questions_per_sample": max_questions,
        "locomo_metric_profile": args.locomo_metric_profile,
        "category_breakdown": _locomo_category_breakdown(question_rows),
        "paper_table_bleu1": _locomo_paper_table_summary(
            question_rows, include_categories=sorted(category_filter),
        ),
        "avg_latency_seconds": _mean_latency_seconds(question_rows),
        "metrics": _mean_metrics(question_rows),
        "rows_file": (str(rows_file) if rows_file is not None else None),
        "resources_file": (str(resources_file) if resources_file is not None else None),
        "resource_usage": _aggregate_sample_resources(sample_resource_rows),
        **_sanitize_summary_mapping(adapter_kwargs),
    }
    status = "ok" if summary["num_failed"] == 0 else "partial"
    return TaskResult(
        task=task, status=status, duration_sec=duration,
        summary=summary, artifacts=artifacts,
    )
