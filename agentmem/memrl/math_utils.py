"""Shared numerical utilities for RL modules.

Provides canonical implementations of safe standard deviation and z-score
normalization used across memrl_runtime, memt_runtime, trainer, and objectives.
"""

from __future__ import annotations

from math import sqrt
from typing import Optional

def safe_std(values: list[float], *, population: bool = False) -> float:
    """Compute standard deviation with fallback for degenerate cases.

    Args:
        values: List of numeric values.
        population: If True, compute population std (ddof=0).
                    If False, compute sample std (ddof=1). Default: sample.

    Returns:
        Standard deviation, or 1.0 if len(values) <= 1 or std < 1e-9.
    """
    n = len(values)
    if n <= 1:
        return 1.0
    mean = sum(values) / n
    ddof = 0 if population else 1
    var = sum((v - mean) ** 2 for v in values) / (n - ddof)
    std = sqrt(max(var, 0.0))
    return std if std > 1e-9 else 1.0

def zscore(
    value: float,
    mean: float,
    std: float,
    *,
    clamp: Optional[float] = None,
) -> float:
    """Compute z-score with safe division and optional clamping.

    Args:
        value: Value to normalize.
        mean: Mean of the distribution.
        std: Standard deviation (safe_std recommended).
        clamp: If not None, clamp result to [-clamp, +clamp].

    Returns:
        Z-score, optionally clamped.
    """
    z = (value - mean) / (std if std > 1e-9 else 1.0)
    if clamp is not None:
        return max(min(z, clamp), -clamp)
    return z

__all__ = ["safe_std", "zscore"]
