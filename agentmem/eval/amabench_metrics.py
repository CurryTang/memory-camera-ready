"""
AMABench evaluation metrics.

Ported from datasets/amabench/code/utils/evaluation_metrics.py.

Primary metric: LLM-as-Judge (binary yes/no).
Secondary: EM, token-level F1, numeric accuracy, contains, set overlap.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()

def compute_exact_match(predicted: str, golden: str) -> float:
    return 1.0 if normalize_text(predicted) == normalize_text(golden) else 0.0

def compute_f1_score(predicted: str, golden: str) -> float:
    pred_tokens = tokenize(predicted)
    gold_tokens = tokenize(golden)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_numeric_accuracy(predicted: str, golden: str, tolerance: float = 1e-6) -> float:
    pred_numbers = re.findall(r"-?\d+\.?\d*", predicted)
    gold_numbers = re.findall(r"-?\d+\.?\d*", golden)
    if not pred_numbers or not gold_numbers:
        return compute_exact_match(predicted, golden)
    try:
        return 1.0 if abs(float(pred_numbers[0]) - float(gold_numbers[0])) <= tolerance else 0.0
    except ValueError:
        return compute_exact_match(predicted, golden)

def compute_contains_score(predicted: str, golden: str) -> float:
    p, g = normalize_text(predicted), normalize_text(golden)
    return 1.0 if (g in p or p in g) else 0.0

def compute_set_overlap(predicted: str, golden: str) -> float:
    pred_tokens = set(tokenize(predicted))
    gold_tokens = set(tokenize(golden))
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    return len(pred_tokens & gold_tokens) / len(pred_tokens | gold_tokens)

def compute_multi_choice_accuracy(predicted: str, golden: str) -> float:
    pred_match = re.search(r"\d+", predicted)
    gold_match = re.search(r"\d+", golden)
    if pred_match and gold_match:
        return 1.0 if pred_match.group() == gold_match.group() else 0.0
    return compute_exact_match(predicted, golden)

_JUDGE_PROMPT = """\
You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

{context}

Question: {question}

Reference Answer: {golden_answer}

Predicted Answer: {predicted_answer}

Is the predicted answer correct? Respond with ONLY "yes" or "no". Do not include any thinking process, explanation, or additional text.

Answer:"""

def compute_llm_as_judge(
    question: str,
    golden_answer: str,
    predicted_answer: str,
    judge_fn: Any,
    *,
    task_type: str = "",
    task_description: str = "",
    episode_id: str = "",
) -> float:
    """
    Binary LLM-as-judge evaluation.

    Args:
        judge_fn: Callable(prompt: str) -> str. Any function that takes a
                  prompt string and returns the judge model's response text.
                  This decouples the metric from a specific LLM client.
    """
    context_parts: list[str] = []
    if episode_id:
        context_parts.append(f"Episode ID: {episode_id}")
    if task_description:
        context_parts.append(f"Task Context: {task_description}")

    prompt = _JUDGE_PROMPT.format(
        context="\n".join(context_parts),
        question=question,
        golden_answer=golden_answer,
        predicted_answer=predicted_answer,
    )

    response = judge_fn(prompt)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = cleaned.strip("`\"' \n\t")
    if cleaned.lower().startswith("answer:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    token_match = re.match(r"^(yes|no)\b", cleaned, flags=re.IGNORECASE)
    if token_match:
        return 1.0 if token_match.group(1).lower() == "yes" else 0.0
    return 0.0

def compute_all_metrics(predicted: str, golden: str) -> dict[str, float]:
    return {
        "exact_match": compute_exact_match(predicted, golden),
        "f1_score": compute_f1_score(predicted, golden),
        "numeric_accuracy": compute_numeric_accuracy(predicted, golden),
        "multi_choice_accuracy": compute_multi_choice_accuracy(predicted, golden),
        "contains_score": compute_contains_score(predicted, golden),
        "set_overlap": compute_set_overlap(predicted, golden),
    }

def per_question_metrics(
    predicted: str,
    golden: str,
    *,
    question: Optional[str] = None,
    judge_fn: Optional[Any] = None,
    task_type: str = "",
    task_description: str = "",
    episode_id: str = "",
) -> dict[str, float]:
    """Compute AMABench metrics for a single question."""
    metrics = compute_all_metrics(predicted, golden)
    if judge_fn is not None and question is not None:
        metrics["llm_judge"] = compute_llm_as_judge(
            question,
            golden,
            predicted,
            judge_fn,
            task_type=task_type,
            task_description=task_description,
            episode_id=episode_id,
        )
    return metrics

def summarize_amabench_rows(
    rows: list[dict[str, Any]],
    *,
    primary_metric: str = "llm_judge",
    group_key: str = "domain",
) -> dict[str, Any]:
    """
    Aggregate per-question rows into AMABench-style summary.

    Each row is expected to have at least: ``{group_key}``, ``metrics`` (dict),
    and optionally ``error``.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("error"):
            continue
        g = str(row.get(group_key, "unknown"))
        groups.setdefault(g, []).append(row)

    by_group: dict[str, dict[str, Any]] = {}
    primary_scores: list[float] = []

    for g, bucket in sorted(groups.items()):
        metric_keys: set[str] = set()
        for r in bucket:
            m = r.get("metrics")
            if isinstance(m, dict):
                metric_keys.update(m.keys())

        group_metrics: dict[str, float] = {}
        for mk in sorted(metric_keys):
            vals = [
                float(r["metrics"][mk])
                for r in bucket
                if isinstance(r.get("metrics"), dict) and mk in r["metrics"]
            ]
            if vals:
                group_metrics[mk] = sum(vals) / len(vals)

        by_group[g] = {
            "num_questions": len(bucket),
            "metrics": group_metrics,
        }
        if primary_metric in group_metrics:
            primary_scores.append(group_metrics[primary_metric])

    avg_primary = sum(primary_scores) / len(primary_scores) if primary_scores else None

    return {
        "groups": by_group,
        "group_key": group_key,
        "primary_metric": primary_metric,
        "average": {primary_metric: avg_primary},
    }
