"""Dataset loader for HUST-AI-HYZ MemoryAgentBench.

The upstream benchmark stores one long haystack/context per row and list-valued
``questions`` / ``answers`` fields. This module normalizes those rows into the
schema consumed by ``examples/run_benchmark.py``:

``{haystack_id, category, context_text, qas: [{question, gold, ...}]}``
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

CATEGORY_TO_SPLIT: dict[str, str] = {
    "AR": "Accurate_Retrieval",
    "TTL": "Test_Time_Learning",
    "LRU": "Long_Range_Understanding",
    "CR": "Conflict_Resolution",
}

DEFAULT_CATEGORIES: tuple[str, ...] = ("TTL", "CR", "LRU")

def load_memoryagentbench(
    *,
    categories: Iterable[str] | None = None,
    cache_dir: str | Path | None = None,
    test_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load and normalize MemoryAgentBench haystacks.

    Args:
        categories: Category abbreviations to include: AR, TTL, LRU, CR.
        cache_dir: Hugging Face cache directory. Defaults to ``$HF_HOME`` or
            ``~/.cache/agentmem/mab``.
        test_file: Optional local JSON/JSONL file. Rows may already be
            normalized or may use the upstream HF row schema.
    """
    wanted = _normalize_categories(categories)
    if test_file:
        rows = _load_local_rows(Path(test_file))
        haystacks = [
            _normalize_any_row(row, fallback_category=_infer_category(row), row_index=i)
            for i, row in enumerate(rows)
        ]
        return [row for row in haystacks if row["category"] in wanted]

    root = Path(cache_dir or os.environ.get("HF_HOME") or Path.home() / ".cache" / "agentmem" / "mab")
    root.mkdir(parents=True, exist_ok=True)

    haystacks: list[dict[str, Any]] = []
    try:
        load_dataset = _load_hf_load_dataset()
    except ImportError:
        load_dataset = None

    if load_dataset is not None:
        for category in wanted:
            split = CATEGORY_TO_SPLIT[category]
            ds = load_dataset(
                "ai-hyz/MemoryAgentBench",
                split=split,
                revision="main",
                cache_dir=str(root),
            )
            for row_index, row in enumerate(ds):
                haystacks.append(_normalize_hf_row(dict(row), category=category, row_index=row_index))
        return haystacks

    haystacks.extend(_load_hf_rows_via_parquet(categories=wanted, cache_dir=root))
    return haystacks

def _load_hf_load_dataset() -> Any:
    """Import Hugging Face ``datasets.load_dataset`` despite local data dirs.

    This repo has a top-level ``datasets/`` directory for local benchmark
    artifacts. On some hosts that directory becomes a namespace package and
    shadows the Hugging Face package. Temporarily removing repo paths from
    ``sys.path`` lets Python resolve the installed dependency.
    """

    try:
        from datasets import load_dataset

        return load_dataset
    except (ImportError, AttributeError):
        pass

    repo_root = Path(__file__).resolve().parents[3]
    cwd = Path.cwd().resolve()
    original = list(sys.path)
    filtered: list[str] = []
    for item in original:
        if not item:
            path = cwd
        else:
            try:
                path = Path(item).resolve()
            except Exception:
                filtered.append(item)
                continue
        if path == repo_root or path == repo_root / "datasets":
            continue
        filtered.append(item)

    prior = sys.modules.pop("datasets", None)
    try:
        sys.path[:] = filtered
        from datasets import load_dataset

        return load_dataset
    except Exception:
        if prior is not None:
            sys.modules["datasets"] = prior
        raise
    finally:
        sys.path[:] = original

def _load_hf_rows_via_parquet(*, categories: Iterable[str], cache_dir: Path) -> list[dict[str, Any]]:
    """Fallback loader for hosts with ``huggingface_hub`` + ``polars`` only."""

    try:
        from huggingface_hub import hf_hub_download
        import polars as pl
    except Exception as exc:
        raise ImportError(
            "Install either `datasets` or both `huggingface_hub` and `polars` "
            "to load ai-hyz/MemoryAgentBench."
        ) from exc

    haystacks: list[dict[str, Any]] = []
    for category in categories:
        split = CATEGORY_TO_SPLIT[_canonical_category(category)]
        filename = f"data/{split}-00000-of-00001.parquet"
        path = hf_hub_download(
            repo_id="ai-hyz/MemoryAgentBench",
            repo_type="dataset",
            filename=filename,
            revision="main",
            cache_dir=str(cache_dir),
        )
        frame = pl.read_parquet(path)
        for row_index, row in enumerate(frame.to_dicts()):
            haystacks.append(_normalize_hf_row(_jsonable_mapping(row), category=category, row_index=row_index))
    return haystacks

def _jsonable_mapping(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "to_list"):
            value = value.to_list()
        elif hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        out[str(key)] = value
    return out

def _normalize_categories(categories: Iterable[str] | None) -> list[str]:
    raw = list(categories or DEFAULT_CATEGORIES)
    normalized: list[str] = []
    for item in raw:
        key = str(item).strip().upper()
        if not key:
            continue
        if key not in CATEGORY_TO_SPLIT:
            raise ValueError(f"Unknown MemoryAgentBench category {item!r}; choose from {sorted(CATEGORY_TO_SPLIT)}")
        if key not in normalized:
            normalized.append(key)
    return normalized or list(DEFAULT_CATEGORIES)

def _load_local_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _normalize_any_row(row: dict[str, Any], *, fallback_category: str, row_index: int) -> dict[str, Any]:
    if "context_text" in row and "qas" in row:
        category = _canonical_category(row.get("category") or fallback_category)
        return {
            "haystack_id": str(row.get("haystack_id") or row.get("episode_id") or f"{category.lower()}_{row_index}"),
            "category": category,
            "source": str(row.get("source", "")),
            "context_text": str(row.get("context_text") or ""),
            "qas": [_normalize_local_qa(qa, i) for i, qa in enumerate(row.get("qas") or [])],
            "context_tokens": row.get("context_tokens"),
            "context_chars": row.get("context_chars"),
            "selection": row.get("selection", {}),
            "metadata": row.get("metadata", {}),
        }
    return _normalize_hf_row(row, category=fallback_category, row_index=row_index)

def _normalize_hf_row(row: dict[str, Any], *, category: str, row_index: int) -> dict[str, Any]:
    category = _canonical_category(category)
    metadata = _coerce_mapping(row.get("metadata"))
    source = str(row.get("source") or metadata.get("source") or "")
    questions = _ensure_list(row.get("questions"))
    answers = _ensure_list(row.get("answers"))
    qa_pair_ids = _ensure_list(row.get("qa_pair_ids") or metadata.get("qa_pair_ids"))
    question_ids = _ensure_list(row.get("question_ids") or metadata.get("question_ids"))
    question_types = _ensure_list(row.get("question_types") or metadata.get("question_types"))
    question_dates = _ensure_list(row.get("question_dates") or metadata.get("question_dates"))
    previous_events = _ensure_list(row.get("previous_events") or metadata.get("previous_events"))

    qas: list[dict[str, Any]] = []
    if len(questions) > 1 and len(answers) > 1:
        pairs = zip(questions, answers)
    else:
        pairs = [(questions[0] if questions else "", answers)]
    for qa_index, (question, gold) in enumerate(pairs):
        qas.append(
            {
                "question": str(question or ""),
                "gold": _normalize_gold(gold),
                "qa_pair_id": _pick(qa_pair_ids, qa_index),
                "question_id": _pick(question_ids, qa_index),
                "question_type": _pick(question_types, qa_index),
                "question_date": _pick(question_dates, qa_index),
                "previous_event": _pick(previous_events, qa_index),
            }
        )

    haystack_id = (
        row.get("haystack_id")
        or row.get("id")
        or row.get("uuid")
        or metadata.get("haystack_id")
        or metadata.get("uuid")
        or f"{category.lower()}_{source or 'sample'}_{row_index}"
    )
    return {
        "haystack_id": str(haystack_id),
        "category": category,
        "source": source,
        "context_text": str(row.get("context") or row.get("context_text") or ""),
        "qas": qas,
        "metadata": metadata,
    }

def _normalize_local_qa(qa: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "question": str(qa.get("question", "")),
        "gold": _normalize_gold(qa.get("gold", qa.get("answer", qa.get("answers", "")))),
        "qa_pair_id": qa.get("qa_pair_id", qa.get("id", str(index))),
        **{k: v for k, v in qa.items() if k not in {"question", "gold", "answer", "answers", "qa_pair_id", "id"}},
    }

def _infer_category(row: dict[str, Any]) -> str:
    raw = row.get("category") or row.get("dataset") or row.get("split") or ""
    try:
        return _canonical_category(raw)
    except ValueError:
        metadata = _coerce_mapping(row.get("metadata"))
        return _canonical_category(metadata.get("category") or metadata.get("dataset") or "TTL")

def _canonical_category(value: Any) -> str:
    raw = str(value or "").strip()
    upper = raw.upper()
    if upper in CATEGORY_TO_SPLIT:
        return upper
    for category, split in CATEGORY_TO_SPLIT.items():
        if raw == split:
            return category
    lowered = raw.lower()
    aliases = {
        "accurate_retrieval": "AR",
        "test_time_learning": "TTL",
        "long_range_understanding": "LRU",
        "conflict_resolution": "CR",
    }
    if lowered in aliases:
        return aliases[lowered]
    raise ValueError(f"Unknown MemoryAgentBench category {value!r}")

def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}

def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return [value]

def _normalize_gold(value: Any) -> Any:
    values = _ensure_list(value)
    if len(values) == 1:
        return values[0]
    return values

def _pick(values: list[Any], index: int) -> Any:
    if not values:
        return ""
    if index < len(values):
        return values[index]
    return values[0]
