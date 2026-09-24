"""Generic file I/O and subprocess helpers for evaluation tasks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

def _repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[3]

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _count_lines(path: Path) -> Optional[int]:
    """Count non-empty lines in a file, or None if the file doesn't exist."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f if _.strip())

def _collect_artifacts(
    base: Path,
    patterns: list[str],
) -> list[Path]:
    """Collect existing files matching patterns relative to base."""
    result: list[Path] = []
    for pattern in patterns:
        candidate = base / pattern
        if candidate.exists():
            result.append(candidate)
    return result

def _run_subprocess(
    command: list[str],
    cwd: Path,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return the result."""
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        check=False,
        text=True,
    )

def _dump_memrl_store_records(store: Any, path: Path) -> None:
    """Dump EpisodicMemoryStore records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in store.iter_chronological():
            row = {
                "id": rec.id,
                "content": rec.content,
                "metadata": dict(rec.metadata or {}),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
