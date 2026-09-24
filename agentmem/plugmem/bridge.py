from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

from agentmem.plugmem.config import PlugMemConfig

class PlugMemBridge:
    """Resolve and validate a PlugMem source checkout."""

    def __init__(self, config: PlugMemConfig) -> None:
        self.config = config
        self.source_root = self.resolve_source_root(
            config.source_root,
            required_dirs=config.required_dirs,
        )
        self._validate_required_dirs()

    @staticmethod
    def resolve_source_root(
        raw_root: Path | str,
        required_dirs: tuple[str, ...] = (
            "memory_retrieving",
            "memory_structuring",
            "memory_reasoning",
        ),
    ) -> Path:
        root = Path(raw_root).expanduser().resolve()
        candidates = [
            root,
            root / "code",
            root / "src",
            root / "code" / "src",
        ]

        seen: set[Path] = set()
        ordered: list[Path] = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)

        partial_candidate: Path | None = None
        for candidate in ordered:
            if not candidate.is_dir():
                continue
            if PlugMemBridge._has_required_dirs(candidate, required_dirs):
                return candidate
            if partial_candidate is None:
                partial_candidate = candidate

        if partial_candidate is not None:
            missing = [
                rel for rel in required_dirs if not (partial_candidate / rel).is_dir()
            ]
            raise FileNotFoundError(
                "Missing required PlugMem directories under "
                f"{partial_candidate}: {', '.join(missing)}. "
                "If you use the vendored checkout, run "
                "`git submodule update --init --recursive vendor/plugmem`."
            )

        raise FileNotFoundError(
            f"Unable to resolve a PlugMem source root from {root}. "
            "If you use the vendored checkout, run "
            "`git submodule update --init --recursive vendor/plugmem`."
        )

    @staticmethod
    def _has_required_dirs(candidate: Path, required_dirs: tuple[str, ...]) -> bool:
        return all((candidate / rel).is_dir() for rel in required_dirs)

    def _validate_required_dirs(self) -> None:
        missing = [rel for rel in self.config.required_dirs if not (self.source_root / rel).is_dir()]
        if missing:
            raise FileNotFoundError(
                "Missing required PlugMem directories under "
                f"{self.source_root}: {', '.join(missing)}. "
                "If you use the vendored checkout, run "
                "`git submodule update --init --recursive vendor/plugmem`."
            )

    @contextmanager
    def scoped_env(self, overrides: Mapping[str, str] | None = None) -> Iterator[None]:
        merged = dict(self.config.env_overrides)
        if overrides:
            merged.update({str(key): str(value) for key, value in overrides.items()})

        previous: dict[str, str | None] = {}
        try:
            for key, value in merged.items():
                previous[key] = os.environ.get(key)
                os.environ[key] = value
            yield
        finally:
            for key, old_value in previous.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
