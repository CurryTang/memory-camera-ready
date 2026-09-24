"""
agentmem.memt — Mem-T integration layer.

Wraps the vendored Mem-T codebase (vendor/memt/) with a thin adapter for use
within agentmem.

Original components (MemoryBuilder, MemoryRetriever, MemoryFormation,
MemoryUpdate, VectorDB) are imported directly from the original code.
Only the wiring (MemTRuntimeEngine) and type bridges live here.
"""

from agentmem.memt.runtime import MemTConfig, MemTMemoryBank, MemTRuntimeEngine
from agentmem.memt.types import (
    MemTConstructionResult,
    MemTMemTrajectory,
    MemTRetrievalResult,
    MemTToolCall,
    MemTTraceStep,
)

Mem_TConfig = MemTConfig
Mem_TRuntimeEngine = MemTRuntimeEngine
MemTCollections = MemTMemoryBank.COLLECTIONS

__all__ = [
    "MemTConfig",
    "MemTRuntimeEngine",
    "MemTMemoryBank",
    "MemTCollections",
    "Mem_TConfig",
    "Mem_TRuntimeEngine",
    "MemTToolCall",
    "MemTTraceStep",
    "MemTRetrievalResult",
    "MemTMemTrajectory",
    "MemTConstructionResult",
]
