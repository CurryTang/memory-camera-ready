#!/usr/bin/env python3
"""Judge MemoryAgentBench answer files with the unified LLM judge.

The MemoryAgentBench runner records token-F1/EM for quick diagnostics, but
paper-grade reporting uses a unified LLM-as-judge score. This script rescoring
`answers_*.jsonl` files with `agentmem.eval.llm_judge.LLMJudge` in binary
yes/no mode and writes both judged rows and a summary.
"""

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

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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

def _task_description(row: dict[str, Any]) -> str:
    category = str(row.get("category") or "").upper()
    source = str(row.get("source") or "")
    instructions = {
        "AR": "Accurate Retrieval: answer using only the memorized context.",
        "TTL": "Test-Time Learning: infer the learned label or output from examples.",
        "LRU": "Long-Range Understanding: answer from the long memorized context and follow the requested output format.",
        "CR": "Conflict Resolution: use the newest non-conflicting relevant fact.",
    }
    parts = [instructions.get(category, "MemoryAgentBench question.")]
    if category:
        parts.append(f"Category: {category}")
    if source:
        parts.append(f"Source: {source}")
    return "\n".join(parts)

def _summarize(rows: list[dict[str, Any]], *, model: str, base_url: str, input_path: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("category") or "unknown")].append(row)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        judged = [row for row in items if not row.get("llm_judge_error")]
        correct = [row for row in judged if bool(row.get("llm_judge_correct"))]
        return {
            "num_rows": len(items),
            "num_judged": len(judged),
            "num_errors": len(items) - len(judged),
            "llm_judge_accuracy": (len(correct) / len(judged)) if judged else 0.0,
            "llm_judge_correct": len(correct),
        }

    return {
        "benchmark": "memoryagentbench",
        "judge_model": model,
        "judge_base_url": base_url,
        "judge_mode": "amabench yes/no factual correctness",
        "input_path": str(input_path),
        "overall": aggregate(rows),
        "by_category": {cat: aggregate(items) for cat, items in sorted(groups.items())},
    }

def judge_file(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem.replace("answers_", "judged_") + input_path.suffix
    )
    summary_path = Path(args.summary) if args.summary else output_path.with_name("summary_judged.json")

    rows = _load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    judge = LLMJudge(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        mode="amabench",
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        temperature=0.0,
    )

    for row in rows:
        row.setdefault("task_description", _task_description(row))

    result = judge.judge_batch(
        rows,
        question_key="question",
        gold_key="gold",
        prediction_key="prediction",
        task_description_key="task_description",
        episode_id_key="episode_id",
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
        if judged.error:
            out["llm_judge_error"] = judged.error
        judged_rows.append(out)

    _write_jsonl(output_path, judged_rows)
    summary = _summarize(judged_rows, model=args.model, base_url=args.base_url, input_path=input_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to answers_*.jsonl")
    parser.add_argument("--output", default="", help="Path for judged rows JSONL")
    parser.add_argument("--summary", default="", help="Path for judged summary JSON")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Optional quick-test limit")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    judge_file(args)

if __name__ == "__main__":
    main()
