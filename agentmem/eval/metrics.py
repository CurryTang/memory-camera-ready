"""
Evaluation metrics.

Extracted from examples/locomo_evaluation.py so they can be reused
across any benchmark, not just LoCoMo.

Also provides retrieval metrics (Recall@k, Precision@k, nDCG@k) via
the compute_retrieval_metrics() function.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Dict, Optional, Sequence

def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: Any) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []

def exact_match(prediction: Any, reference: Any) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0

def token_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = sum(
        1 for token in pred_tokens if ref_counts.get(token, 0) > 0
        and ref_counts.__setitem__(token, ref_counts[token] - 1) is None                                    
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            current = dp[j]
            dp[j] = prev + 1 if token_a == token_b else max(dp[j], dp[j - 1])
            prev = current
    return dp[-1]

def rouge_l_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_metrics(
    prediction: str,
    reference: Optional[Any],
    extra_metrics: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    """
    Compute standard metrics for a single prediction/reference pair.

    Args:
        prediction: Model output string.
        reference: Gold answer string. If None, string-match metrics are skipped.
        extra_metrics: Additional pre-computed metrics to merge in.

    Returns:
        Dict of metric_name → float score.
    """
    metrics: dict[str, float] = {}
    if reference is not None:
        prediction_text = "" if prediction is None else str(prediction)
        reference_text = str(reference)
        metrics["exact_match"] = exact_match(prediction_text, reference_text)
        metrics["token_f1"] = token_f1(prediction_text, reference_text)
        metrics["rouge_l_f1"] = rouge_l_f1(prediction_text, reference_text)
    if extra_metrics:
        metrics.update(extra_metrics)
    return metrics

def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Recall@k = |retrieved[:k] ∩ relevant| / |relevant|.

    Returns 0.0 when relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)

def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Precision@k = |retrieved[:k] ∩ relevant| / k.

    Returns 0.0 when k == 0.
    """
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / k

def _dcg(gains: Sequence[float]) -> float:
    """Discounted Cumulative Gain for an ordered list of relevance gains."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """nDCG@k using binary relevance (1 if relevant, 0 otherwise).

    Returns 0.0 when relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    if k == 0:
        return 1.0
    relevant = set(relevant_ids)
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved_ids[:k]]
    dcg = _dcg(gains)
    ideal_gains = [1.0] * min(k, len(relevant))
    idcg = _dcg(ideal_gains)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def compute_retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Compute Recall@k, Precision@k, and nDCG@k for multiple k values.

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids: Ground-truth relevant IDs (unordered).
        k_values: k values to evaluate at.

    Returns:
        Dict mapping metric names like "recall@5", "precision@5", "ndcg@5".
    """
    result: Dict[str, float] = {}
    for k in k_values:
        result[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        result[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    return result
