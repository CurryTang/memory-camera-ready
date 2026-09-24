"""Base class for AMAbench memory methods."""

from __future__ import annotations

import json
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

class BaseMethod(ABC):
    """All methods implement memory_construction and memory_retrieve."""

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        with open(path, "r") as f:
            if path.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f) or {}
            return json.load(f)

    @abstractmethod
    def memory_construction(self, traj_text: str, task: str = "") -> Any:
        ...

    @abstractmethod
    def memory_retrieve(self, memory: Any, question: str) -> str:
        ...
