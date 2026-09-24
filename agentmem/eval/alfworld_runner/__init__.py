"""ALFWorld prompt-agent evaluation package.

This package is used in environments with very different dependency surfaces.
Keep imports fail-open so light consumers (for example memory backend utilities)
do not require the full prompt-agent stack at import time.
"""

from __future__ import annotations

__all__: list[str] = []

try:
    from agentmem.eval.alfworld_runner.agents import (
        BaseAlfWorldAgent,
        ClassicReActAlfWorldAgent,
        MemoryAugmentedReActAlfWorldAgent,
        MemoryAugmentedReflexionAlfWorldAgent,
        ReActAlfWorldAgent,
        ReflexionAlfWorldAgent,
        VanillaAlfWorldAgent,
        create_agent,
    )

    __all__ += [
        "BaseAlfWorldAgent",
        "VanillaAlfWorldAgent",
        "ReActAlfWorldAgent",
        "ClassicReActAlfWorldAgent",
        "MemoryAugmentedReActAlfWorldAgent",
        "ReflexionAlfWorldAgent",
        "MemoryAugmentedReflexionAlfWorldAgent",
        "create_agent",
    ]
except Exception:
    pass

try:
    from agentmem.eval.alfworld_runner.memory import (
        AMAAgentMemoryBackend,
        BaseAlfWorldMemoryBackend,
        HippoRAGv2MemoryBackend,
        MemRLAlfWorldMemoryBackend,
        MemTMemoryBackend,
        PlugMemMemoryBackend,
        SimpleMemMemoryBackend,
        create_memory_backend,
    )

    __all__ += [
        "BaseAlfWorldMemoryBackend",
        "HippoRAGv2MemoryBackend",
        "PlugMemMemoryBackend",
        "SimpleMemMemoryBackend",
        "AMAAgentMemoryBackend",
        "MemTMemoryBackend",
        "MemRLAlfWorldMemoryBackend",
        "create_memory_backend",
    ]
except Exception:
    pass

try:
    from agentmem.eval.alfworld_runner.runner import (
        AlfWorldEvalReport,
        AlfWorldEvalRunner,
        AlfWorldTask,
        TaskResult,
        extract_task_desc,
        load_alfworld_tasks,
    )

    __all__ += [
        "AlfWorldTask",
        "TaskResult",
        "AlfWorldEvalRunner",
        "AlfWorldEvalReport",
        "extract_task_desc",
        "load_alfworld_tasks",
    ]
except Exception:
    pass
