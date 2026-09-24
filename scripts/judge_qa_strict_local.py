#!/usr/bin/env python3
"""Strictly rejudge QA prediction rows with a local OpenAI-compatible LLM."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are an impartial judge evaluating QA accuracy.

Determine whether the predicted answer is factually correct given the question and reference answer.
Focus on factual equivalence, not exact wording. Do not give credit merely because the prediction
mentions the same topic. Extra context is acceptable only if it does not contradict the reference.
Return only JSON."""

USER_TEMPLATE = """Question:
{question}

Reference answer:
{gold}
{gold_context_block}
Predicted answer:
{prediction}

Judge strictly:
- CORRECT only if the prediction contains the key fact(s) required by the reference answer.
- WRONG if the prediction is vague, merely topical, incomplete on a required fact, contradicts the reference, or says the answer is unknown.
- Dates, names, quantities, and yes/no polarity must match.

Return exactly:
{{"correct": true, "reason": "<brief reason>"}}
or
{{"correct": false, "reason": "<brief reason>"}}"""

def _format_gold_context(row: dict[str, Any]) -> str:
    """Build the optional gold-paragraph block for the judge prompt.

    Looks for the supporting paragraphs in row["gold_paragraphs"] (preferred,
    pre-extracted text) or reconstructs them from row["context"] +
    row["supporting_facts"] (HotpotQA layout). Returns "" if nothing is
    available so judging works for benchmarks without supporting facts.
    """
    paragraphs = row.get("gold_paragraphs")
    if not paragraphs:
        sf = row.get("supporting_facts") or []
        gold_titles = {item[0] for item in sf if isinstance(item, (list, tuple)) and item}
        ctx = row.get("context") or []
        out: list[str] = []
        for entry in ctx:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[0] in gold_titles:
                title, sentences = entry[0], entry[1]
                body = " ".join(str(s) for s in sentences) if isinstance(sentences, list) else str(sentences)
                out.append(f"{title}\n{body}")
        paragraphs = out
    if not paragraphs:
        return ""
    rendered = "\n\n".join(str(p).strip() for p in paragraphs if str(p).strip())
    if not rendered:
        return ""
    return f"\nGold supporting paragraphs (background context for the judge — DO NOT reward predictions that merely repeat these without containing the reference answer):\n{rendered}\n"

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def chat_completion(
    *,
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"].get("content") or ""
    return str(content), payload.get("usage") or {}

def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {cleaned[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Judge response JSON is not an object")
    return parsed

def normalize_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "correct", "yes"}

def summarize(rows: list[dict[str, Any]], *, input_path: Path, output_path: Path, model: str) -> dict[str, Any]:
    judgeable = [row for row in rows if not row.get("strict_judge_error")]
    correct = sum(1 for row in judgeable if row.get("strict_judge_correct"))
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in judgeable:
        category = str(row.get("category", row.get("qa_type", "unknown")))
        by_category[category]["total"] += 1
        if row.get("strict_judge_correct"):
            by_category[category]["correct"] += 1
    categories = {
        category: {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
        }
        for category, counts in sorted(by_category.items())
    }
    generous_correct = [
        row for row in rows
        if row.get("llm_judge_correct") is True and not row.get("strict_judge_correct")
    ]
    strict_rescues = [
        row for row in rows
        if row.get("llm_judge_correct") is False and row.get("strict_judge_correct") is True
    ]
    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "judge_model": model,
        "total": len(rows),
        "valid_count": len(judgeable),
        "correct": correct,
        "accuracy": correct / len(judgeable) if judgeable else 0.0,
        "parse_errors": len(rows) - len(judgeable),
        "categories": categories,
        "generous_to_strict_wrong": len(generous_correct),
        "strict_rescues_from_generous_wrong": len(strict_rescues),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--gold-key", default="gold")
    parser.add_argument("--prediction-key", default="prediction")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_rows = read_jsonl(args.input)
    judged_rows = read_jsonl(args.output) if args.resume and args.output.exists() else []
    start_at = len(judged_rows)
    started = time.perf_counter()

    for index, row in enumerate(source_rows[start_at:], start=start_at + 1):
        out = dict(row)
        prompt = USER_TEMPLATE.format(
            question=str(row.get(args.question_key, "")),
            gold=str(row.get(args.gold_key, "")),
            prediction=str(row.get(args.prediction_key, "")),
            gold_context_block=_format_gold_context(row),
        )
        try:
            content, usage = chat_completion(
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            parsed = extract_json_object(content)
            out["strict_judge_correct"] = normalize_bool(parsed.get("correct"))
            out["strict_judge_score"] = 1.0 if out["strict_judge_correct"] else 0.0
            out["strict_judge_reason"] = str(parsed.get("reason", "") or "")[:1000]
            out["strict_judge_raw_response"] = content
            out["strict_judge_usage"] = usage
        except Exception as exc:
            out["strict_judge_correct"] = False
            out["strict_judge_score"] = 0.0
            out["strict_judge_reason"] = f"judge_error: {exc}"
            out["strict_judge_error"] = str(exc)
        judged_rows.append(out)

        if index % args.flush_every == 0 or index == len(source_rows):
            write_jsonl(args.output, judged_rows)
            done = len(judged_rows)
            correct = sum(1 for item in judged_rows if item.get("strict_judge_correct"))
            print(
                f"{done}/{len(source_rows)} rows judged, "
                f"acc={correct / done * 100:.1f}%, elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    write_jsonl(args.output, judged_rows)
    summary = summarize(judged_rows, input_path=args.input, output_path=args.output, model=args.model)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
