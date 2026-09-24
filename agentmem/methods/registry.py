"""Unified method registry for paper and ablation runs.

This is the single allow-list used by ``examples/run_benchmark.py``.  Methods
fall into two groups:

* paper-grade baselines that appear in the main tables;
* experimental AutoMem/DCI variants used for ablations and diagnostics.

Adapters either implement the unified ``build`` / ``answer`` interface directly
or wrap a legacy AMABench adapter.  Wrapped adapters return retrieved context
only; the benchmark runner owns the final answer prompt so answer generation is
consistent across methods and cannot see gold labels.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from agentmem.methods.base import (
    BaseMethod,
    EfficiencyCounters,
    MeteredMethod,
    MethodKind,
)

_AMABENCH_BACKED = {
    "simplemem": ("agentmem.eval.amabench_runner.methods.simplemem", "SimpleMemMethod", MethodKind.SUMMARY),
    "hipporag":  ("agentmem.eval.amabench_runner.methods.hipporag",  "HippoRAGMethod",  MethodKind.STRUCTURED),
    "plugmem":   ("agentmem.eval.amabench_runner.methods.plugmem",   "PlugMemMethod",   MethodKind.STRUCTURED),
    "memt":      ("agentmem.eval.amabench_runner.methods.memt",      "MemTMethod",      MethodKind.STRUCTURED),
    "ama_agent": ("agentmem.eval.amabench_runner.methods.ama_agent", "AMAAgentMethod",  MethodKind.AGENTIC),
}

_DIRECT = {
    "lightmem":    ("agentmem.methods.lightmem",     "LightMemMethod",         MethodKind.SUMMARY),
    "longcontext": ("agentmem.methods.longcontext",  "LongContextMethod",      MethodKind.LONG_CONTEXT),
    "mem0":        ("agentmem.methods.mem0",         "Mem0Method",             MethodKind.STRUCTURED),
    "memoryos":    ("agentmem.methods.memoryos",     "MemoryOSMethod",         MethodKind.STRUCTURED),
    "amem":        ("agentmem.methods.amem",         "AMemMethod",             MethodKind.STRUCTURED),
    "memrl":       ("agentmem.methods.memrl",        "MemRLMethod",            MethodKind.STRUCTURED),
    "autoharness": ("agentmem.methods.autoharness",  "AutoHarnessMethod",      MethodKind.STRUCTURED),
    "dci_lite":    ("agentmem.methods.dci_lite",     "DCILiteMethod",          MethodKind.AGENTIC),
    "dci_lite_sum": ("agentmem.methods.dci_lite",    "DCILiteSummarizeMethod", MethodKind.AGENTIC),
    "dci_memory":    ("agentmem.methods.automem",    "AutoMemMethod",          MethodKind.AGENTIC),
    "automem":       ("agentmem.methods.automem",   "AutoMemMethod",          MethodKind.AGENTIC),
    "automem_u1":    ("agentmem.methods.automem",   "AutoMemU1Method",        MethodKind.AGENTIC),
    "automem_u2":    ("agentmem.methods.automem",   "AutoMemU2Method",        MethodKind.AGENTIC),
    "automem_u3":    ("agentmem.methods.automem",   "AutoMemU3Method",        MethodKind.AGENTIC),
    "automem_sum":   ("agentmem.methods.automem",   "AutoMemSumMethod",       MethodKind.AGENTIC),
    "automem_nograph": ("agentmem.methods.automem", "AutoMemNoGraphMethod",   MethodKind.AGENTIC),
    "automem_graph": ("agentmem.methods.automem",   "AutoMemGraphMethod",     MethodKind.AGENTIC),
    "automem_cost":      ("agentmem.methods.automem_cost", "AutoMemCostMethod",      MethodKind.AGENTIC),
    "automem_cost_spc":  ("agentmem.methods.automem_cost", "AutoMemCostSPCOnly",     MethodKind.AGENTIC),
    "automem_cost_spc_gs": ("agentmem.methods.automem_cost", "AutoMemCostSPCGS",     MethodKind.AGENTIC),
    "automem_kvcache":   ("agentmem.methods.automem_cost", "AutoMemCostKVFriendly",  MethodKind.AGENTIC),
    "automem_kv":        ("agentmem.methods.automem_kv",   "AutoMemKVMethod",        MethodKind.AGENTIC),
    "automem_graph_kv":  ("agentmem.methods.automem_kv",   "AutoMemGraphKVMethod",   MethodKind.AGENTIC),
    "automem_single":            ("agentmem.methods.automem_kv", "AutoMemSinglePassMethod",      MethodKind.AGENTIC),
    "automem_oneretry":          ("agentmem.methods.automem_kv", "AutoMemOneRetryMethod",        MethodKind.AGENTIC),
    "automem_graph_single":      ("agentmem.methods.automem_kv", "AutoMemGraphSinglePassMethod", MethodKind.AGENTIC),
    "automem_graph_oneretry":    ("agentmem.methods.automem_kv", "AutoMemGraphOneRetryMethod",   MethodKind.AGENTIC),
    "automem_graph_template":    ("agentmem.methods.automem_kv", "AutoMemGraphTemplateMethod",   MethodKind.AGENTIC),
}

PAPER_METHODS: tuple[str, ...] = (
    "longcontext",
    "simplemem",
    "lightmem",
    "hipporag",
    "plugmem",
    "ama_agent",
    "memt",
    "memrl",
    "mem0",
    "memoryos",
    "amem",
    "dci_lite",
    "dci_lite_sum",
)
"""Methods intended for paper-grade table cells in the current library."""

EXPERIMENTAL_METHODS: tuple[str, ...] = tuple(
    name for name in sorted({*_DIRECT, *_AMABENCH_BACKED}) if name not in PAPER_METHODS
)
"""Ablation/debug methods. Do not include in paper tables unless explicitly requested."""

class _AmaBenchDelegatedMethod(MeteredMethod, BaseMethod):
    """Wraps a legacy amabench adapter and instruments the boundary calls.

    The legacy adapter's API is ``memory_construction(traj_text, task)`` and
    ``memory_retrieve(memory, question) -> context_text``. We keep that
    exact contract internally and only add counters around the two calls.
    Accurate LLM-token counts for the build phase rely on the underlying
    adapter exposing usage dicts in its trajectory dump; where it does not,
    the runner's legacy ``resource_metrics`` still provides a fallback.
    """

    def __init__(
        self,
        *,
        inner: Any,
        kind: MethodKind,
        name: str,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        super().__init__()
        self._inner = inner
        self.kind = kind
        self.name = name
        self._token_counter = token_counter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def build(self, traj_text: str, *, task: str = "") -> Any:
        with self._counters.time_block("build_wallclock_seconds"):
            memory = self._inner.memory_construction(traj_text, task=task)
        return memory

    def memory_construction(self, traj_text: str, task: str = "") -> Any:
        return self.build(traj_text, task=task)

    def memory_retrieve(self, memory: Any, question: str) -> str:
        return self.answer(memory, question)

    def answer(self, memory: Any, question: str) -> str:
        with self._counters.time_block("wallclock_seconds"):
            context = self._inner.memory_retrieve(memory, question)
        ctx_tokens = self._count_tokens(context)

        self._counters.record_retrieval(
            candidates_scored=0,
            evidence_injected=1 if context else 0,
            context_tokens=ctx_tokens,
        )
        return str(context)

    def persistent_store_bytes(self, memory: Any) -> int:
        size = 0
        save_dir = getattr(self._inner, "save_dir", None)
        if save_dir:
            from pathlib import Path

            p = Path(save_dir)
            if p.exists():
                size = sum(
                    f.stat().st_size for f in p.rglob("*") if f.is_file()
                )
        return size

    def _count_tokens(self, text: Any) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0
        return len(str(text or "").split())

def build_method(name: str, /, **kwargs: Any) -> BaseMethod:
    """Instantiate a unified method adapter.

    Usage::

        m = build_method("simplemem", config_path="...")
        mem = m.build(traj_text)
        ctx = m.answer(mem, question)
        counters = m.counters.to_dict()

    ``name`` is case-insensitive. Passing construction kwargs forwards them
    to the underlying (legacy or native) adapter's ``__init__``.
    """
    key = name.lower().replace("-", "").replace("_", "")
    key = {
        "memt": "memt",
        "memT": "memt",
        "mem0": "mem0",
        "memzero": "mem0",
        "memoryos": "memoryos",
        "memory-os": "memoryos",
        "amem": "amem",
        "a-mem": "amem",
        "agenticmemory": "amem",
        "amaagent": "ama_agent",
        "dcilite": "dci_lite",
        "dcilitesum": "dci_lite_sum",
        "dcimemory": "dci_memory",
        "automem": "automem",
        "automemu1": "automem_u1",
        "automemu2": "automem_u2",
        "automemu3": "automem_u3",
        "automemsum": "automem_sum",
        "automemnograph": "automem_nograph",
        "automemgraph": "automem_graph",
        "automemcost": "automem_cost",
        "automemcostspc": "automem_cost_spc",
        "automemcostspcgs": "automem_cost_spc_gs",
        "automemkvcache": "automem_kvcache",
        "automemkv": "automem_kv",
        "automemgraphkv": "automem_graph_kv",
        "automemsingle": "automem_single",
        "automemoneretry": "automem_oneretry",
        "automemgraphsingle": "automem_graph_single",
        "automemgraphoneretry": "automem_graph_oneretry",
        "automemgraphtemplate": "automem_graph_template",
    }.get(key, key)

    if key in _DIRECT:
        module_name, class_name, _ = _DIRECT[key]
        cls = getattr(import_module(module_name), class_name)
        return cls(**kwargs)

    if key in _AMABENCH_BACKED:
        module_name, class_name, kind = _AMABENCH_BACKED[key]
        inner_cls = getattr(import_module(module_name), class_name)
        import inspect

        params = inspect.signature(inner_cls.__init__).parameters
        forwarded = {k: v for k, v in kwargs.items() if k in params}
        inner = inner_cls(**forwarded)
        tc = kwargs.get("token_counter")
        return _AmaBenchDelegatedMethod(
            inner=inner, kind=kind, name=key, token_counter=tc
        )

    raise ValueError(
        f"Unknown method {name!r}. Available: {sorted(_DIRECT) + sorted(_AMABENCH_BACKED)}"
    )

AVAILABLE_METHODS: tuple[str, ...] = tuple(sorted({*_DIRECT, *_AMABENCH_BACKED}))
