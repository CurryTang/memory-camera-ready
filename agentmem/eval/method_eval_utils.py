"""Shared helpers for repo-owned benchmark adapters and runners."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from agentmem.eval.resource_metrics import summarize_resource_usage

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def ensure_jsonl_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    ensure_jsonl_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def sum_usage_dicts(usages: Iterable[Mapping[str, Any] | None]) -> dict[str, int]:
    total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for usage in usages:
        if not usage:
            continue
        total["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        total["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        explicit_total = int(usage.get("total_tokens", 0) or 0)
        if explicit_total:
            total["total_tokens"] += explicit_total
        else:
            total["total_tokens"] += (
                int(usage.get("prompt_tokens", 0) or 0)
                + int(usage.get("completion_tokens", 0) or 0)
            )
    return total

def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)

def aggregate_rows_resource_usage(
    rows: Iterable[Mapping[str, Any]],
    *,
    model: Optional[str] = None,
) -> dict[str, Any]:
    summaries = [summarize_resource_usage(row, model=model) for row in rows]
    if not summaries:
        return {
            "num_records": 0,
            "num_questions": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_llm_tokens": 0,
            "retrieved_context_tokens": 0,
            "memory_construction_seconds": 0.0,
            "question_seconds": 0.0,
            "total_runtime_seconds": 0.0,
        }

    return {
        "num_records": len(summaries),
        "num_questions": sum(int(s["num_questions"]) for s in summaries),
        "prompt_tokens": sum(int(s["prompt_tokens"]) for s in summaries),
        "completion_tokens": sum(int(s["completion_tokens"]) for s in summaries),
        "total_llm_tokens": sum(int(s["total_llm_tokens"]) for s in summaries),
        "retrieved_context_tokens": sum(int(s["retrieved_context_tokens"]) for s in summaries),
        "memory_construction_seconds": sum(
            float(s["memory_construction_seconds"]) for s in summaries
        ),
        "question_seconds": sum(float(s["question_seconds"]) for s in summaries),
        "total_runtime_seconds": sum(float(s["total_runtime_seconds"]) for s in summaries),
        "records_with_explicit_usage": sum(
            1 for s in summaries if bool(s.get("has_explicit_token_usage"))
        ),
    }

class Stopwatch:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start
