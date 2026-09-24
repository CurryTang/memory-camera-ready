"""Unified QA prompts and answer normalization for LoCoMo evaluation.

All adapters should use these functions to ensure consistent prompting
and answer post-processing across methods.
"""

from __future__ import annotations

import re
from typing import Any, Optional

def _parse_prefixed_dialogue(raw: str) -> tuple[str, Optional[str], str]:
    """Parse ``[Speaker] [time=TS] content`` format into components."""
    text = "" if raw is None else str(raw)
    match = re.match(r"^\[([^\]]+)\](?:\s+\[time=([^\]]+)\])?\s*(.*)$", text.strip())
    if not match:
        return "Unknown", None, text.strip()
    speaker = (match.group(1) or "Unknown").strip() or "Unknown"
    timestamp = (match.group(2) or "").strip() or None
    content = (match.group(3) or "").strip()
    return speaker, timestamp, content

def _build_locomo_answer_prompt(question: str, context: str, category: Optional[int]) -> str:
    """Build the QA prompt for LoCoMo evaluation.

    This is the single source of truth for the QA prompt used by ALL methods
    (baselines C1-C9, SimpleMem, MemT, LongContext, HippoRAGv2, etc.).
    """
    if category == 5:
        prompt = (
            "Return exactly: Not mentioned in the conversation.\n\n"
            f"Memory snippets:\n{context}\n\nQuestion: {question}\n\n"
        )
    elif category == 3:
        prompt = (
            "Use only the provided memory snippets to answer the question. "
            "This is an open-domain/counterfactual question: infer the most likely answer from evidence. "
            "Do not answer with Not mentioned in the conversation. "
            "For binary questions, start with Yes/No; when uncertain but supported, use Likely yes/Likely no.\n\n"
            f"Memory snippets:\n{context}\n\nQuestion: {question}\n\n"
        )
    else:
        prompt = (
            "Use only the provided memory snippets to answer the question. "
            "Give a short grounded answer from the snippets. "
            "If the snippets support a likely inference, provide that inference briefly. "
            "Only when there is no relevant evidence, reply exactly: Not mentioned in the conversation.\n\n"
            f"Memory snippets:\n{context}\n\nQuestion: {question}\n\n"
        )
    if category == 2:
        prompt += "Answer with temporal order/date facts when possible.\n"
    prompt += "Return only the answer, no explanation."
    return prompt

def _coerce_int_category(value: Any) -> Optional[int]:
    """Coerce a category value to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None

def _question_is_binary(question: str) -> bool:
    """Check if a question expects a yes/no answer."""
    q = str(question or "").strip().lower()
    return q.startswith(
        (
            "would ",
            "will ",
            "is ",
            "are ",
            "do ",
            "does ",
            "did ",
            "can ",
            "could ",
            "should ",
            "has ",
            "have ",
            "had ",
            "was ",
            "were ",
        )
    )

def _normalize_locomo_prediction(
    raw: Any,
    category: Optional[int] = None,
    question: Optional[str] = None,
) -> str:
    """Normalize a model's raw prediction for LoCoMo evaluation.

    Strips markdown fences, <think> tags, extracts last line,
    normalizes abstention phrases, and handles binary question heuristics.
    """
    text = "" if raw is None else str(raw)
    if not text:
        if category == 5:
            return "Not mentioned in the conversation"
        return ""

    cleaned = text
    cleaned = re.sub(r"```(?:json)?", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", " ", cleaned)
    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"(?is)^\s*<think>.*$", " ", cleaned)
    cleaned = cleaned.strip()

    if not cleaned and "not mentioned in the conversation" in text.lower():
        return "Not mentioned in the conversation"
    if not cleaned:
        return "Not mentioned in the conversation" if category == 5 else ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) > 1:
        cleaned = lines[-1]
    else:
        cleaned = lines[0]

    cleaned = re.sub(r"^\s*answer\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip("\"'").strip()
    if category == 5:
        return "Not mentioned in the conversation"

    abstain = re.sub(r"[.!]+$", "", cleaned.lower()).strip()
    if abstain in {
        "not mentioned",
        "not explicitly mentioned",
        "not mentioned in the conversation",
        "unknown",
        "i don't know",
        "cannot determine",
        "can't determine",
    }:
        return "Not mentioned in the conversation"

    if category == 3 and _question_is_binary(str(question or "")):
        lowered = cleaned.lower()
        negative_cues = [
            "likely no",
            "no,",
            " no ",
            "would not",
            "wouldn't",
            "unlikely",
            "not likely",
            "not pursue",
        ]
        positive_cues = [
            "likely yes",
            "yes,",
            " yes ",
            "would still",
            "would continue",
            "likely",
            "probably",
        ]
        has_negative = any(cue in lowered for cue in negative_cues)
        has_positive = any(cue in lowered for cue in positive_cues)
        if has_negative and not has_positive:
            return "Likely no"
        if has_positive and not has_negative:
            return "Likely yes"

    return cleaned
