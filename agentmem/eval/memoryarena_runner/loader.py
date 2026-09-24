"""MemoryArena dataset loader + manifest-based subsample selection."""

from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

CONFIGS = (
    "bundled_shopping",
    "progressive_search",
    "group_travel_planner",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_DIR = _REPO_ROOT / "agentmem" / "eval" / "memoryarena_runner" / "manifests"
_LOCAL_DATA_ROOT = _REPO_ROOT / "datasets" / "memoryarena"

def _import_hf_load_dataset():
    """Import HF ``datasets.load_dataset`` despite this repo's datasets/ folder.

    The project has a local ``datasets`` directory. In a checkout run from the
    repository root, that directory can shadow Hugging Face's package as a
    namespace package. Temporarily removing repo entries from ``sys.path`` keeps
    the harness usable in both local and packaged environments.
    """
    old_path = list(sys.path)
    old_datasets = sys.modules.get("datasets")
    repo_str = str(_REPO_ROOT)
    try:
        if old_datasets is not None and getattr(old_datasets, "__file__", None) is None:
            sys.modules.pop("datasets", None)
        sys.path = [
            item
            for item in sys.path
            if item not in ("", repo_str) and Path(item or ".").resolve() != _REPO_ROOT
        ]
        load_dataset = importlib.import_module("datasets").load_dataset

        return load_dataset
    finally:
        sys.path = old_path
        if old_datasets is not None and "datasets" not in sys.modules:
            sys.modules["datasets"] = old_datasets

def load_manifest(config: str) -> list[int] | None:
    """Return the list of HF row indices to evaluate for ``config``.

    Returns None when no manifest is checked in (caller should fall back to
    the full split or a runtime sample).
    """
    if config not in CONFIGS:
        raise ValueError(f"Unknown MemoryArena config: {config!r}; expected one of {CONFIGS}")
    p = _MANIFEST_DIR / f"{config}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

def _read_local_jsonl(config: str) -> list[dict[str, Any]]:
    path = _LOCAL_DATA_ROOT / config / "data.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"MemoryArena local data not found at {path}. "
            "Install Hugging Face datasets or materialize datasets/memoryarena/<config>/data.jsonl."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _load_rows(config: str, split: str, source: str) -> list[dict[str, Any]]:
    if source not in {"auto", "hf", "local"}:
        raise ValueError("source must be one of: auto, hf, local")
    if source in {"auto", "local"}:
        local_path = _LOCAL_DATA_ROOT / config / "data.jsonl"
        if local_path.exists():
            return _read_local_jsonl(config)
        if source == "local":
            return _read_local_jsonl(config)
    try:
        load_dataset = _import_hf_load_dataset()
        ds = load_dataset("ZexueHe/memoryarena", config, split=split)
        return [dict(row) for row in ds]
    except Exception:
        if source == "hf":
            raise
        return _read_local_jsonl(config)

def load_tasks(
    config: str,
    *,
    split: str = "test",
    limit: int | None = None,
    use_manifest: bool = True,
    source: str = "auto",
) -> Iterator[dict[str, Any]]:
    """Yield MemoryArena tasks for ``config``.

    When ``use_manifest`` is True and a manifest exists, only those row
    indices are yielded (in manifest order). Otherwise the full split is
    yielded; ``limit`` truncates.
    """
    if config not in CONFIGS:
        raise ValueError(f"Unknown MemoryArena config: {config!r}; expected one of {CONFIGS}")
    rows = _load_rows(config, split, source)
    indices: list[int]
    if use_manifest:
        manifest = load_manifest(config)
        if manifest is not None:
            indices = manifest
        else:
            indices = list(range(len(rows)))
    else:
        indices = list(range(len(rows)))
    if limit is not None:
        indices = indices[:limit]
    for i in indices:
        row = dict(rows[i])
        row["_hf_index"] = i
        yield row

def get_background(task: dict[str, Any], session_id: int) -> str:
    """Render the background string for session ``session_id`` of ``task``.

    MemoryArena rows have heterogeneous shapes per config; this function
    normalizes the common cases. See HF dataset card for field details.
    """
    if task.get("backgrounds") is not None:
        b = task["backgrounds"]
        if isinstance(b, list):
            return b[session_id] if session_id < len(b) else ""
        return str(b)
    if task.get("base_person") is not None:
        return json.dumps(task["base_person"], ensure_ascii=False, indent=2)
    return ""

def num_subtasks(task: dict[str, Any]) -> int:
    qs = task.get("questions") or []
    return len(qs)

def iter_sessions(task: dict[str, Any]) -> Iterable[tuple[int, str, Any | None, str]]:
    """Yield normalized ``(session_id, question, gold, background)`` tuples."""
    questions = task.get("questions") or []
    answers = task.get("answers") or []
    for i, question in enumerate(questions):
        gold = answers[i] if i < len(answers) else None
        yield i, str(question), gold, get_background(task, i)
