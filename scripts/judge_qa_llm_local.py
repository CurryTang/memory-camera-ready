#!/usr/bin/env python3
"""Judge QA answer files with the unified local LLM judge."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentmem.eval.llm_judge import LLMJudge

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if not row.get("llm_judge_error")]
    correct = [row for row in judged if row.get("llm_judge_correct") is True]
    return {
        "num_rows": len(rows),
        "num_judged": len(judged),
        "num_errors": len(rows) - len(judged),
        "llm_judge_accuracy": (len(correct) / len(judged)) if judged else 0.0,
        "llm_judge_correct": len(correct),
    }

def _summarize(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    judge_mode: str,
    model: str,
    base_url: str,
    input_path: Path,
) -> dict[str, Any]:
    by_question_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("question_type") is not None:
            by_question_type[str(row.get("question_type"))].append(row)
        if row.get("category") is not None:
            by_category[str(row.get("category"))].append(row)
    summary: dict[str, Any] = {
        "benchmark": benchmark,
        "judge_model": model,
        "judge_base_url": base_url,
        "judge_mode": judge_mode,
        "input_path": str(input_path),
        "overall": _aggregate(rows),
    }
    if by_question_type:
        summary["by_question_type"] = {
            key: _aggregate(items) for key, items in sorted(by_question_type.items())
        }
    if by_category:
        summary["by_category"] = {
            key: _aggregate(items) for key, items in sorted(by_category.items())
        }
    return summary

def judge_file(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem.replace("answers_", "judged_").replace("results_", "judged_")
        + input_path.suffix
    )
    summary_path = Path(args.summary) if args.summary else output_path.with_name("summary_judged.json")

    rows = _read_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    judge = LLMJudge(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        mode=args.mode,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        temperature=0.0,
    )
    result = judge.judge_batch(
        rows,
        question_key=args.question_key,
        gold_key=args.gold_key,
        prediction_key=args.prediction_key,
        progress=not args.quiet,
    )

    judged_rows: list[dict[str, Any]] = []
    for row, judged in zip(rows, result.results):
        out = dict(row)
        out["llm_judge_correct"] = bool(judged.correct)
        out["llm_judge_score"] = float(judged.score)
        out["llm_judge_reason"] = judged.reason
        out["llm_judge_raw"] = judged.raw_response
        out["llm_judge_latency_sec"] = round(float(judged.latency_sec or 0.0), 3)
        out["llm_judge_model"] = args.model
        out["llm_judge_mode"] = args.mode
        if judged.error:
            out["llm_judge_error"] = judged.error
        judged_rows.append(out)

    _write_jsonl(output_path, judged_rows)
    summary = _summarize(
        judged_rows,
        benchmark=args.benchmark,
        judge_mode=args.mode,
        model=args.model,
        base_url=args.base_url,
        input_path=input_path,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--benchmark", default="qa")
    parser.add_argument("--mode", choices=("hotpotqa", "locomo", "amabench", "strict"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--gold-key", default="gold")
    parser.add_argument("--prediction-key", default="prediction")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    judge_file(args)

if __name__ == "__main__":
    main()
