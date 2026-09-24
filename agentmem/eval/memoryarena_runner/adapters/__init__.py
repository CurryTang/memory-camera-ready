"""Multi-session adapters wrapping each existing method.

Reference adapter: longcontext (concatenate all session traces).
Add new adapters per method by subclassing and providing build/answer wiring.
"""

from agentmem.eval.memoryarena_runner.adapters.buffer import BufferMemory
from agentmem.eval.memoryarena_runner.adapters.longcontext import LongContextAdapter
from agentmem.eval.memoryarena_runner.adapters.structured import ProgressiveSearchStructuredMemory
from agentmem.eval.memoryarena_runner.adapters.unified import (
    AMAAgentAdapter,
    AutoHarnessAdapter,
    DCILiteAdapter,
    DCILiteSummarizeAdapter,
    AutoMemAdapter,
    HippoRAGAdapter,
    HippoRAGv2Adapter,
    LightMemAdapter,
    MemRLAdapter,
    MemTAdapter,
    PlugMemAdapter,
    SimpleMemAdapter,
    UnifiedMethodSessionAdapter,
)

__all__ = [
    "AMAAgentAdapter",
    "AutoHarnessAdapter",
    "BufferMemory",
    "DCILiteAdapter",
    "DCILiteSummarizeAdapter",
    "AutoMemAdapter",
    "HippoRAGAdapter",
    "HippoRAGv2Adapter",
    "LightMemAdapter",
    "LongContextAdapter",
    "MemRLAdapter",
    "MemTAdapter",
    "PlugMemAdapter",
    "ProgressiveSearchStructuredMemory",
    "SimpleMemAdapter",
    "UnifiedMethodSessionAdapter",
]
