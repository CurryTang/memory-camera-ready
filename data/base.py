"""
Base interfaces for evaluation datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from data.utils.download import download_file

@dataclass(frozen=True)
class DownloadConfig:
    """
    Download metadata for a dataset variant.
    """

    url: str
    filename: str
    description: str = ""
    timeout_seconds: int = 60

@dataclass(frozen=True)
class DialogueRecord:
    """
    Canonical dialogue record for memory ingestion.
    """

    speaker: str
    content: str
    timestamp: Optional[str] = None

@dataclass(frozen=True)
class QAPair:
    """
    Canonical QA pair for evaluation.
    """

    question: str
    answer: Optional[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def final_answer(self) -> Optional[str]:
        if self.category == 5:
            return self.adversarial_answer
        return self.answer

class BaseDataset(ABC):
    """
    Base dataset interface and shared filesystem/download behavior.
    """

    NAME = "base"
    CATEGORY = "uncategorized"
    MEMORY_AXES: tuple[str, ...] = tuple()
    INTERFACE_STYLE = "dialogue_qa"
    DEFAULT_VARIANT = "default"
    DOWNLOAD_CONFIGS: dict[str, DownloadConfig] = {}

    def __init__(self, datasets_dir: str | Path = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.dataset_dir = self.datasets_dir / self.NAME
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def available_variants(self) -> list[str]:
        return sorted(self.DOWNLOAD_CONFIGS.keys())

    def describe(self) -> dict[str, Any]:
        """
        Return lightweight dataset metadata for discovery/UI.
        """
        return {
            "name": self.NAME,
            "category": self.CATEGORY,
            "memory_axes": list(self.MEMORY_AXES),
            "interface_style": self.INTERFACE_STYLE,
            "variants": self.available_variants(),
        }

    def get_download_config(self, variant: Optional[str] = None) -> DownloadConfig:
        if not self.DOWNLOAD_CONFIGS:
            raise NotImplementedError(
                f"Dataset '{self.NAME}' does not declare auto-download variants. "
                "Pass an explicit dataset path to ensure_data(path=...)."
            )
        target_variant = variant or self.DEFAULT_VARIANT
        if target_variant not in self.DOWNLOAD_CONFIGS:
            raise ValueError(
                f"Unknown variant '{target_variant}' for dataset '{self.NAME}'. "
                f"Available: {self.available_variants()}"
            )
        return self.DOWNLOAD_CONFIGS[target_variant]

    def get_data_path(self, variant: Optional[str] = None) -> Path:
        config = self.get_download_config(variant=variant)
        return self.dataset_dir / config.filename

    def download(self, variant: Optional[str] = None, force: bool = False) -> Path:
        config = self.get_download_config(variant=variant)
        destination = self.dataset_dir / config.filename
        download_file(
            url=config.url,
            destination=destination,
            timeout_seconds=config.timeout_seconds,
            overwrite=force,
        )
        return destination

    def ensure_data(
        self,
        path: Optional[str | Path] = None,
        variant: Optional[str] = None,
        auto_download: bool = False,
    ) -> Path:
        if path is not None:
            dataset_path = Path(path)
            if dataset_path.exists():
                return dataset_path
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}.")

        if self.DOWNLOAD_CONFIGS:
            dataset_path = self.get_data_path(variant=variant)
            if dataset_path.exists():
                return dataset_path
            if auto_download:
                return self.download(variant=variant, force=False)
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}. "
                "Use auto_download=True or call download() first."
            )

        raise FileNotFoundError(
            f"Dataset '{self.NAME}' requires an explicit path because no "
            "auto-download variant is configured."
        )

    @abstractmethod
    def load_samples(self, path: str | Path) -> list[Any]:
        """
        Load raw dataset samples from disk.
        """

    @abstractmethod
    def iter_dialogues(self, sample: Any) -> Iterable[DialogueRecord]:
        """
        Yield dialogue records from one sample.
        """

    @abstractmethod
    def iter_qa_pairs(self, sample: Any) -> Iterable[QAPair]:
        """
        Yield QA pairs from one sample.
        """
