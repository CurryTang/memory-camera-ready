from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

@dataclass
class PlugMemConfig:
    """Configuration for the native PlugMem bridge."""

    source_root: Path | str
    env_overrides: Mapping[str, str] = field(default_factory=dict)
    required_dirs: tuple[str, ...] = (
        "memory_retrieving",
        "memory_structuring",
        "memory_reasoning",
    )

    def __post_init__(self) -> None:
        self.source_root = Path(self.source_root).expanduser().resolve()
        self.env_overrides = {str(key): str(value) for key, value in dict(self.env_overrides).items()}
        self.required_dirs = tuple(str(item) for item in self.required_dirs)
