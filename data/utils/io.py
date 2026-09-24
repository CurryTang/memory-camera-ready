"""
I/O helpers for dataset loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def load_json(path: str | Path) -> Any:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"JSON file not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)
