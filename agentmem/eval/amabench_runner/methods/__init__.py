"""AMAbench memory methods — loaded lazily to avoid heavy startup imports.

Methods in :data:`_UNIFIED_METHODS` are additionally available through the
hardware-independent counter layer (:mod:`agentmem.methods`). When
``get_method(name, metered=True)`` is used, those methods are wrapped so that
every ``memory_construction`` / ``memory_retrieve`` call populates an
:class:`~agentmem.methods.EfficiencyCounters` exposed via ``.counters``.
"""

from importlib import import_module

from agentmem.eval.amabench_runner.methods.base import BaseMethod

METHOD_REGISTRY = {
    "bm25": ("agentmem.eval.amabench_runner.methods.bm25", "BM25Method"),
    "embedding": ("agentmem.eval.amabench_runner.methods.embedding_mem", "EmbeddingMethod"),
    "longcontext": ("agentmem.eval.amabench_runner.methods.longcontext", "LongContextMethod"),
    "hipporag": ("agentmem.eval.amabench_runner.methods.hipporag", "HippoRAGMethod"),
    "simplemem": ("agentmem.eval.amabench_runner.methods.simplemem", "SimpleMemMethod"),
    "lightmem": ("agentmem.eval.amabench_runner.methods.lightmem", "LightMemMethod"),
    "ama-agent": ("agentmem.eval.amabench_runner.methods.ama_agent", "AMAAgentMethod"),
    "ama_agent": ("agentmem.eval.amabench_runner.methods.ama_agent", "AMAAgentMethod"),
    "memt": ("agentmem.eval.amabench_runner.methods.memt", "MemTMethod"),
    "mem-t": ("agentmem.eval.amabench_runner.methods.memt", "MemTMethod"),
    "plugmem": ("agentmem.eval.amabench_runner.methods.plugmem", "PlugMemMethod"),
    "memrl": ("agentmem.eval.amabench_runner.methods.memrl", "MemRLMethod"),
    "dci_lite": ("agentmem.eval.amabench_runner.methods.dci_lite", "DCILiteMethod"),
    "dci_lite_sum": ("agentmem.eval.amabench_runner.methods.dci_lite", "DCILiteSumMethod"),
    "automem": ("agentmem.eval.amabench_runner.methods.dci_lite", "AutoMemMethod"),
    "automem_u1": ("agentmem.eval.amabench_runner.methods.dci_lite", "AutoMemU1Method"),
    "automem_u2": ("agentmem.eval.amabench_runner.methods.dci_lite", "AutoMemU2Method"),
    "automem_u3": ("agentmem.eval.amabench_runner.methods.dci_lite", "AutoMemU3Method"),
    "automem_graph": ("agentmem.eval.amabench_runner.methods.dci_lite", "AutoMemGraphMethod"),
}

_UNIFIED_METHODS = frozenset(
    {"simplemem", "hipporag", "plugmem", "memt", "mem-t", "longcontext", "ama_agent", "ama-agent"}
)

def get_method(name: str, *, metered: bool = False, **kwargs) -> BaseMethod:
    """Instantiate a method adapter.

    When ``metered=True`` and ``name`` is in :data:`_UNIFIED_METHODS`, the
        adapter is wrapped by :func:`agentmem.methods.build_method` so that
        ``memory_construction`` / ``memory_retrieve`` calls go through the
        counter-instrumented boundary.
    """
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {list(METHOD_REGISTRY.keys())}")

    if metered and name in _UNIFIED_METHODS:
        from agentmem.methods import build_method

        unified_name = (
            "memt" if name in {"memt", "mem-t"}
            else "ama_agent" if name in {"ama_agent", "ama-agent"}
            else name
        )
        return build_method(unified_name, **kwargs)

    import inspect

    module_name, class_name = METHOD_REGISTRY[name]
    module = import_module(module_name)
    cls = getattr(module, class_name)
    params = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return cls(**filtered)
