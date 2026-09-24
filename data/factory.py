"""
Factory and registry for dataset adapters.
"""

from __future__ import annotations

from typing import Type

from data.base import BaseDataset

class DatasetFactory:
    """
    Registry-backed factory for dataset classes.
    """

    _registry: dict[str, Type[BaseDataset]] = {}

    @classmethod
    def register(cls, name: str, dataset_cls: Type[BaseDataset]) -> None:
        normalized_name = name.strip().lower()
        cls._registry[normalized_name] = dataset_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDataset:
        normalized_name = name.strip().lower()
        if normalized_name not in cls._registry:
            raise ValueError(
                f"Unknown dataset '{name}'. Available datasets: {cls.available()}"
            )
        dataset_cls = cls._registry[normalized_name]
        return dataset_cls(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def available_by_category(cls) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for name, dataset_cls in cls._registry.items():
            category = str(getattr(dataset_cls, "CATEGORY", "uncategorized"))
            grouped.setdefault(category, []).append(name)
        return {key: sorted(values) for key, values in sorted(grouped.items())}

    @classmethod
    def canonical_by_category(cls) -> dict[str, list[str]]:
        grouped: dict[str, set[str]] = {}
        for _, dataset_cls in cls._registry.items():
            category = str(getattr(dataset_cls, "CATEGORY", "uncategorized"))
            canonical_name = str(getattr(dataset_cls, "NAME", dataset_cls.__name__.lower()))
            grouped.setdefault(category, set()).add(canonical_name)
        return {key: sorted(values) for key, values in sorted(grouped.items())}

    @classmethod
    def describe(cls, name: str) -> dict[str, object]:
        normalized_name = name.strip().lower()
        if normalized_name not in cls._registry:
            raise ValueError(
                f"Unknown dataset '{name}'. Available datasets: {cls.available()}"
            )
        dataset_cls = cls._registry[normalized_name]
        return {
            "name": normalized_name,
            "class_name": dataset_cls.__name__,
            "category": getattr(dataset_cls, "CATEGORY", "uncategorized"),
            "memory_axes": list(getattr(dataset_cls, "MEMORY_AXES", tuple())),
            "interface_style": getattr(dataset_cls, "INTERFACE_STYLE", "dialogue_qa"),
            "variants": sorted(getattr(dataset_cls, "DOWNLOAD_CONFIGS", {}).keys()),
        }

def register_dataset(name: str):
    """
    Class decorator for dataset registration.
    """

    def decorator(dataset_cls: Type[BaseDataset]) -> Type[BaseDataset]:
        DatasetFactory.register(name, dataset_cls)
        return dataset_cls

    return decorator
