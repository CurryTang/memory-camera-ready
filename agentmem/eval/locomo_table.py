"""
Utilities for LoCoMo-style table reporting (F1/BLEU by category + average).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from agentmem.eval.metrics import token_f1, tokenize

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.tokenize import word_tokenize
except Exception:                                          
    SmoothingFunction = None
    sentence_bleu = None
    word_tokenize = None

LOCOMO_CATEGORY_LABELS: dict[int, str] = {
    1: "MultiHop",
    2: "Temporal",
    3: "OpenDomain",
    4: "SingleHop",
    5: "Adversarial",
}

def _bleu_with_weights(
    prediction: str,
    reference: str,
    *,
    weights: tuple[float, float, float, float],
    smooth: bool = True,
    tokenizer_fn: Optional[Callable[[str], list[str]]] = None,
) -> float:
    """
    Lightweight sentence BLEU-4 (0-1 scale), implemented without external deps.
    """
    tokenizer = tokenizer_fn or tokenize
    pred_tokens = tokenizer(prediction)
    ref_tokens = tokenizer(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    def _ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
        counts: dict[tuple[str, ...], int] = {}
        if len(tokens) < n:
            return counts
        for idx in range(0, len(tokens) - n + 1):
            key = tuple(tokens[idx : idx + n])
            counts[key] = counts.get(key, 0) + 1
        return counts

    precisions: list[float] = []
    for n in (1, 2, 3, 4):
        pred_counts = _ngram_counts(pred_tokens, n)
        ref_counts = _ngram_counts(ref_tokens, n)
        if not pred_counts:

            precisions.append(1.0)
            continue

        clipped = 0
        total = sum(pred_counts.values())
        for key, value in pred_counts.items():
            clipped += min(value, ref_counts.get(key, 0))

        if smooth:
            precisions.append((clipped + 0.1) / (total + 0.1))
        else:
            precisions.append(clipped / total if total > 0 else 0.0)

    if any(p <= 0 for p in precisions):
        return 0.0

    c = len(pred_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1.0 - (r / max(c, 1)))
    active_weights = list(weights)
    score_log = 0.0
    for weight, precision in zip(active_weights, precisions):
        if weight <= 0:
            continue
        score_log += weight * math.log(precision)
    return bp * math.exp(score_log)

def _paper_tokenize(text: str) -> list[str]:
    raw = "" if text is None else str(text).lower()

    raw = raw.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw.split() if raw else []

def _paper_bleu_tokenize(text: str) -> list[str]:
    raw = "" if text is None else str(text).lower()
    if not raw:
        return []
    if word_tokenize is not None:
        try:
            return word_tokenize(raw)
        except Exception:
            pass
    return _paper_tokenize(raw)

def _paper_sentence_bleu(
    prediction: str,
    reference: str,
    *,
    weights: tuple[float, float, float, float],
) -> float:
    if sentence_bleu is None or SmoothingFunction is None:
        return _bleu_with_weights(
            prediction,
            reference,
            weights=weights,
            smooth=True,
            tokenizer_fn=_paper_bleu_tokenize,
        )

    pred_tokens = _paper_bleu_tokenize(prediction)
    ref_tokens = _paper_bleu_tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    try:
        smooth = SmoothingFunction().method1
        return float(sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=smooth))
    except Exception:
        return _bleu_with_weights(
            prediction,
            reference,
            weights=weights,
            smooth=True,
            tokenizer_fn=_paper_bleu_tokenize,
        )

def paper_token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(_paper_tokenize(prediction))
    ref_tokens = set(_paper_tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = pred_tokens & ref_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def bleu1(
    prediction: str,
    reference: str,
    *,
    smooth: bool = True,
    tokenizer_fn: Optional[Callable[[str], list[str]]] = None,
) -> float:
    return _bleu_with_weights(
        prediction,
        reference,
        weights=(1.0, 0.0, 0.0, 0.0),
        smooth=smooth,
        tokenizer_fn=tokenizer_fn,
    )

def bleu2(
    prediction: str,
    reference: str,
    *,
    smooth: bool = True,
    tokenizer_fn: Optional[Callable[[str], list[str]]] = None,
) -> float:
    return _bleu_with_weights(
        prediction,
        reference,
        weights=(0.5, 0.5, 0.0, 0.0),
        smooth=smooth,
        tokenizer_fn=tokenizer_fn,
    )

def bleu3(
    prediction: str,
    reference: str,
    *,
    smooth: bool = True,
    tokenizer_fn: Optional[Callable[[str], list[str]]] = None,
) -> float:
    return _bleu_with_weights(
        prediction,
        reference,
        weights=(1 / 3, 1 / 3, 1 / 3, 0.0),
        smooth=smooth,
        tokenizer_fn=tokenizer_fn,
    )

def bleu4(
    prediction: str,
    reference: str,
    *,
    smooth: bool = True,
    tokenizer_fn: Optional[Callable[[str], list[str]]] = None,
) -> float:
    return _bleu_with_weights(
        prediction,
        reference,
        weights=(0.25, 0.25, 0.25, 0.25),
        smooth=smooth,
        tokenizer_fn=tokenizer_fn,
    )

def per_row_locomo_metrics(
    prediction: str,
    reference: Optional[str],
    *,
    metric_profile: str = "paper",
) -> dict[str, float]:
    if reference is None:
        return {}
    prediction_text = "" if prediction is None else str(prediction)
    reference_text = str(reference)
    normalized_profile = (metric_profile or "paper").strip().lower()
    if normalized_profile == "paper":
        f1_value = paper_token_f1(prediction_text, reference_text)
        bleu1_value = _paper_sentence_bleu(
            prediction_text,
            reference_text,
            weights=(1.0, 0.0, 0.0, 0.0),
        )
        bleu2_value = _paper_sentence_bleu(
            prediction_text,
            reference_text,
            weights=(0.5, 0.5, 0.0, 0.0),
        )
        bleu3_value = _paper_sentence_bleu(
            prediction_text,
            reference_text,
            weights=(1 / 3, 1 / 3, 1 / 3, 0.0),
        )
        bleu4_value = _paper_sentence_bleu(
            prediction_text,
            reference_text,
            weights=(0.25, 0.25, 0.25, 0.25),
        )
    else:
        f1_value = token_f1(prediction_text, reference_text)
        bleu1_value = bleu1(prediction_text, reference_text, tokenizer_fn=tokenize)
        bleu2_value = bleu2(prediction_text, reference_text, tokenizer_fn=tokenize)
        bleu3_value = bleu3(prediction_text, reference_text, tokenizer_fn=tokenize)
        bleu4_value = bleu4(prediction_text, reference_text, tokenizer_fn=tokenize)
    return {

        "f1": f1_value,

        "token_f1": token_f1(prediction_text, reference_text),
        "bleu1": bleu1_value,
        "bleu2": bleu2_value,
        "bleu3": bleu3_value,
        "bleu4": bleu4_value,
    }

def summarize_locomo_rows(
    rows: Iterable[dict[str, Any]],
    *,
    include_categories: tuple[int, ...] = (1, 2, 3, 4),
    bleu_key: str = "bleu1",
    metric_profile: str = "paper",
) -> dict[str, Any]:
    """
    Aggregate rows into LoCoMo paper-style summary.

    Expected row keys: category, prediction, gold, error.
    """
    rows_list = list(rows)
    groups: dict[int, list[dict[str, Any]]] = {c: [] for c in include_categories}
    for row in rows_list:
        if row.get("error"):
            continue
        try:
            category = int(row.get("category"))
        except Exception:
            continue
        if category not in groups:
            continue
        groups[category].append(row)

    by_category: dict[str, dict[str, Any]] = {}
    category_scores_f1: list[float] = []
    category_scores_bleu: list[float] = []

    for category in include_categories:
        bucket = groups[category]
        f1_scores: list[float] = []
        bleu_scores: list[float] = []
        for row in bucket:
            pred = "" if row.get("prediction") is None else str(row.get("prediction"))
            gold = row.get("gold")
            if gold is None:
                continue
            gold_text = str(gold)
            metrics_blob = row.get("metrics")
            if isinstance(metrics_blob, dict) and isinstance(metrics_blob.get("f1"), (int, float)):
                f1_scores.append(float(metrics_blob["f1"]))
            elif (metric_profile or "paper").strip().lower() == "paper":
                f1_scores.append(paper_token_f1(pred, gold_text))
            else:
                f1_scores.append(token_f1(pred, gold_text))
            normalized_profile = (metric_profile or "paper").strip().lower()
            if normalized_profile == "paper":
                if bleu_key == "bleu1":
                    bleu_scores.append(
                        _paper_sentence_bleu(pred, gold_text, weights=(1.0, 0.0, 0.0, 0.0))
                    )
                elif bleu_key == "bleu2":
                    bleu_scores.append(
                        _paper_sentence_bleu(pred, gold_text, weights=(0.5, 0.5, 0.0, 0.0))
                    )
                elif bleu_key == "bleu3":
                    bleu_scores.append(
                        _paper_sentence_bleu(pred, gold_text, weights=(1 / 3, 1 / 3, 1 / 3, 0.0))
                    )
                else:
                    bleu_scores.append(
                        _paper_sentence_bleu(pred, gold_text, weights=(0.25, 0.25, 0.25, 0.25))
                    )
            else:
                if bleu_key == "bleu1":
                    bleu_scores.append(bleu1(pred, gold_text, tokenizer_fn=tokenize))
                elif bleu_key == "bleu2":
                    bleu_scores.append(bleu2(pred, gold_text, tokenizer_fn=tokenize))
                elif bleu_key == "bleu3":
                    bleu_scores.append(bleu3(pred, gold_text, tokenizer_fn=tokenize))
                else:
                    bleu_scores.append(bleu4(pred, gold_text, tokenizer_fn=tokenize))

        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
        mean_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else None
        if mean_f1 is not None:
            category_scores_f1.append(mean_f1)
        if mean_bleu is not None:
            category_scores_bleu.append(mean_bleu)

        by_category[str(category)] = {
            "label": LOCOMO_CATEGORY_LABELS.get(category, f"Category{category}"),
            "num_questions": len(bucket),
            "f1": mean_f1,
            bleu_key: mean_bleu,
            "f1_pct": None if mean_f1 is None else mean_f1 * 100.0,
            f"{bleu_key}_pct": None if mean_bleu is None else mean_bleu * 100.0,
        }

    avg_f1 = sum(category_scores_f1) / len(category_scores_f1) if category_scores_f1 else None
    avg_bleu = (
        sum(category_scores_bleu) / len(category_scores_bleu)
        if category_scores_bleu
        else None
    )

    return {
        "categories": by_category,
        "average": {
            "f1": avg_f1,
            bleu_key: avg_bleu,
            "f1_pct": None if avg_f1 is None else avg_f1 * 100.0,
            f"{bleu_key}_pct": None if avg_bleu is None else avg_bleu * 100.0,
        },
        "bleu_key": bleu_key,
    }

def load_rows_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
