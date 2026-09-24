"""Domain-specific judges for MemoryArena.

These run inside the harness right after the agent produces a prediction.
The harness uses ``judge_signal`` to feed back into ``MultiSessionMemory.update``;
gold is NEVER passed into the memory's update.
"""

from __future__ import annotations

import json
import re
from typing import Any

_ASIN_RE = re.compile(r"\bB[A-Z0-9]{9}\b")
_EXACT_ANSWER_RE = re.compile(
    r"(?:\*\*)?\s*Exact\s+Answer\s*(?:\*\*)?\s*:\s*(.+?)(?:\n\s*(?:\*\*)?\s*Confidence\s*(?:\*\*)?\s*:|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
_CONFIDENCE_SPLIT_RE = re.compile(r"(?:\*\*)?\s*Confidence\s*(?:\*\*)?\s*:", flags=re.IGNORECASE)

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def _content_tokens(text: str) -> set[str]:
    labels = {"answer", "final", "exact", "confidence", "month", "country", "poem", "book"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token not in labels
    }

def asin_judge(pred: str, gold: Any) -> dict[str, Any]:
    """bundled_shopping: extract ASIN and exact-match against gold."""
    gold_asin: str | None
    if isinstance(gold, dict):
        gold_asin = gold.get("target_asin") or gold.get("asin")
    else:
        gold_asin = str(gold) if gold else None
    pred_asin = None
    m = _ASIN_RE.search(pred or "")
    if m:
        pred_asin = m.group(0)
    return {
        "correct": (pred_asin is not None and pred_asin == gold_asin),
        "pred_asin": pred_asin,
        "gold_asin": gold_asin,
    }

def substring_judge(pred: str, gold: Any) -> dict[str, Any]:
    """Lenient string-containment fallback used for non-shopping configs."""
    gold_text = json.dumps(gold, ensure_ascii=False) if not isinstance(gold, str) else gold
    return {
        "correct": _normalize(gold_text) in _normalize(pred or ""),
        "gold_excerpt": gold_text[:300],
    }

def extract_exact_answer(text: Any) -> str | None:
    """Extract MemoryArena's trailing ``Exact Answer: ...`` field when present."""
    if text is None:
        return None
    raw = str(text).strip()
    match = _EXACT_ANSWER_RE.search(raw)
    if not match:

        prefix = _CONFIDENCE_SPLIT_RE.split(raw, maxsplit=1)[0].strip()
        prefix = re.sub(r"^\s*(?:\*\*)?\s*(?:Answer|Final Answer)\s*(?:\*\*)?\s*:\s*", "", prefix, flags=re.I)
        prefix = prefix.strip(" *\n\t")
        if prefix and len(prefix.split()) <= 12 and "\n" not in prefix:
            return re.sub(r"\s+", " ", prefix)
        return None
    answer = match.group(1).strip()
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip(" *")

def progressive_search_judge(pred: str, gold: Any) -> dict[str, Any]:
    """Judge Progressive Web Search with the final exact-answer field.

    The HF snapshot stores each subtask answer as a rationale plus an
    ``Exact Answer:`` line. Agents are prompted to emit the same concise final
    line; if they omit the marker, we conservatively require the gold exact
    answer to appear in the prediction.
    """
    gold_exact = extract_exact_answer(gold)
    pred_exact = extract_exact_answer(pred)
    if gold_exact:
        if pred_exact:
            pred_norm = _normalize(pred_exact)
            gold_norm = _normalize(gold_exact)
            gold_tokens = _content_tokens(gold_exact)
            pred_tokens = _content_tokens(pred_exact)
            correct = (
                pred_norm == gold_norm
                or gold_norm in pred_norm
                or (bool(gold_tokens) and gold_tokens.issubset(pred_tokens))
            )
        else:
            pred_tokens = _content_tokens(pred or "")
            gold_tokens = _content_tokens(gold_exact)
            correct = _normalize(gold_exact) in _normalize(pred or "") or (
                bool(gold_tokens) and gold_tokens.issubset(pred_tokens)
            )
        return {
            "correct": correct,
            "gold_exact": gold_exact,
            "pred_exact": pred_exact,
        }
    verdict = substring_judge(pred, gold)
    verdict["gold_exact"] = None
    verdict["pred_exact"] = pred_exact
    return verdict

def llm_strict_judge(
    pred: str,
    gold: Any,
    question: str,
    *,
    judge_fn,
) -> dict[str, Any]:
    """Defer to ``judge_fn(question, pred, gold) -> {"correct": bool, ...}``.

    ``judge_fn`` is injected by the harness so we don't hard-import any
    OpenAI/SGLang client here. Use ``scripts/judge_qa_strict_local.py``
    machinery for the actual prompt.
    """
    return judge_fn(question=question, prediction=pred, gold=gold)

_DOMAIN_TO_JUDGE = {
    "bundled_shopping": "asin",
    "progressive_search": "progressive_search",
    "group_travel_planner": "leaf_recall",
}

def judge_for(domain: str) -> str:
    return _DOMAIN_TO_JUDGE.get(domain, "substring")

def leaf_recall_judge(pred: str, gold: Any) -> dict[str, Any]:
    """Soft process score for structured plans.

    This approximates MemoryArena's group-travel sPS by counting how many
    non-empty leaf fields from the gold structured plan appear in the predicted
    plan. It gives partial credit when full-plan exact success is unrealistic.
    """
    leaves = [_normalize_leaf(x) for x in _flatten_leaves(gold)]
    leaves = [x for x in leaves if x and x != "-"]
    if not leaves:
        return {"correct": False, "score": 0.0, "matched": 0, "total": 0}
    pred_norm = _normalize_leaf(pred)
    matched = sum(1 for leaf in leaves if leaf in pred_norm)
    score = matched / len(leaves)
    return {
        "correct": score >= 0.999,
        "score": score,
        "matched": matched,
        "total": len(leaves),
    }

def _flatten_leaves(value: Any) -> list[str]:
    if isinstance(value, dict):
        leaves: list[str] = []
        for item in value.values():
            leaves.extend(_flatten_leaves(item))
        return leaves
    if isinstance(value, list):
        leaves = []
        for item in value:
            leaves.extend(_flatten_leaves(item))
        return leaves
    if value is None:
        return []
    return [str(value)]

def _normalize_leaf(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()
