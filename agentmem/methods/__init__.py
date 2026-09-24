"""Unified method interfaces for the 3-factor evaluation (quality, tokens, efficiency).

Every memory/retrieval/agentic method is expressed as a subclass of
:class:`BaseMethod` (QA over a fixed trajectory) or
:class:`BaseAgenticMethod` (ReAct-style, method decides actions online).
The :class:`MeteredMethod` mixin adds hardware-independent counters so all
methods emit the same schema regardless of their internal design.
"""

from agentmem.methods.base import (
    BaseAgenticMethod,
    BaseMethod,
    EfficiencyCounters,
    MeteredMethod,
    MethodKind,
)
from agentmem.methods.registry import (
    AVAILABLE_METHODS,
    EXPERIMENTAL_METHODS,
    PAPER_METHODS,
    build_method,
)

__all__ = [
    "BaseMethod",
    "BaseAgenticMethod",
    "EfficiencyCounters",
    "MeteredMethod",
    "MethodKind",
    "AVAILABLE_METHODS",
    "PAPER_METHODS",
    "EXPERIMENTAL_METHODS",
    "build_method",
]
