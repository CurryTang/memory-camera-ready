"""DCI-Agent-Lite adapters for AMABench.

DCI Lite is lexical/direct-corpus by design: it works over the raw trajectory
text and uses the LLM only to plan bounded corpus searches. The AMABench runner
still owns final answer generation, so this adapter returns retrieved context
from the unified DCI implementation rather than a direct final answer.
"""

from __future__ import annotations

from typing import Any, Optional

from agentmem.eval.amabench_runner.methods.base import BaseMethod
from agentmem.methods.dci_lite import (
    DCILiteMethod as UnifiedDCILiteMethod,
    DCILiteSummarizeMethod,
    DCIMemory,
)
from agentmem.methods.automem import (
    AutoMemMethod as UnifiedAutoMemMethod,
    AutoMemU1Method as UnifiedAutoMemU1Method,
    AutoMemU2Method as UnifiedAutoMemU2Method,
    AutoMemU3Method as UnifiedAutoMemU3Method,
)

class DCILiteMethod(BaseMethod):
    """AMABench wrapper around the unified DCI Lite direct-corpus method."""

    method_cls = UnifiedDCILiteMethod

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_max_tokens: int = 4096,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        top_k: int = 10,
        max_records: int = 4096,
        max_record_chars: int = 20000,
        max_context_chars: int = 24000,
        compact_threshold_chars: int = 240000,
        recent_context_chars: int = 20000,
        controller_max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        request_timeout: float = 90.0,
        save_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        **_kw: Any,
    ) -> None:
        if config_path:
            cfg = self._load_config(config_path)
            llm_model = cfg.get("llm_model", cfg.get("model", llm_model))
            llm_base_url = cfg.get("llm_base_url", cfg.get("base_url", llm_base_url))
            llm_api_key = cfg.get("llm_api_key", cfg.get("api_key", llm_api_key))
            llm_max_tokens = int(cfg.get("llm_max_tokens", llm_max_tokens))
            top_k = int(cfg.get("top_k", top_k))
            max_records = int(cfg.get("max_records", max_records))
            max_record_chars = int(cfg.get("max_record_chars", max_record_chars))
            max_context_chars = int(cfg.get("max_context_chars", max_context_chars))
            compact_threshold_chars = int(
                cfg.get("compact_threshold_chars", compact_threshold_chars)
            )
            recent_context_chars = int(cfg.get("recent_context_chars", recent_context_chars))
            if cfg.get("controller_max_tokens") is not None:
                controller_max_tokens = int(cfg["controller_max_tokens"])
            temperature = float(cfg.get("temperature", temperature))
            request_timeout = float(cfg.get("request_timeout", request_timeout))
            save_dir = cfg.get("save_dir", save_dir)

        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.embedding_api_key = embedding_api_key
        self._last_trajectory: dict[str, Any] = {}

        self._method = self.method_cls(
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            top_k=top_k,
            max_records=max_records,
            max_record_chars=max_record_chars,
            max_context_chars=max_context_chars,
            compact_threshold_chars=compact_threshold_chars,
            recent_context_chars=recent_context_chars,
            controller_max_tokens=controller_max_tokens or llm_max_tokens,
            temperature=temperature,
            request_timeout=request_timeout,
            save_dir=save_dir,
        )

    @property
    def counters(self):
        return self._method.counters

    def reset_counters(self) -> None:
        self._method.reset_counters()

    def memory_construction(self, traj_text: str, task: str = "") -> DCIMemory:
        memory = self._method.memory_construction(traj_text, task=task)
        self._last_trajectory = {
            "dci_method": self._method.name,
            "dci_context_level": memory.context_level,
            "dci_record_count": len(memory.records),
            "dci_summary_chars": len(memory.summary or ""),
        }
        return memory

    def memory_retrieve(self, memory: DCIMemory, question: str) -> str:

        context = self._method.answer(memory, question)
        self._last_trajectory = {
            "dci_method": self._method.name,
            "dci_context_level": memory.context_level,
            "dci_record_count": len(memory.records),
            "dci_summary_chars": len(memory.summary or ""),
            "dci_retrieved_chars": len(context),
        }
        return context

    def last_trajectory(self) -> dict[str, Any]:
        return self._last_trajectory

class DCILiteSumMethod(DCILiteMethod):
    """AMABench wrapper for DCI Lite with level4 summarization enabled."""

    method_cls = DCILiteSummarizeMethod

class AutoMemMethod(DCILiteMethod):
    """AMABench wrapper for AutoMem (all three §5 upgrades enabled)."""

    method_cls = UnifiedAutoMemMethod

class AutoMemU1Method(DCILiteMethod):
    """AMABench wrapper: only U1 (memory primitives) enabled."""

    method_cls = UnifiedAutoMemU1Method

class AutoMemU2Method(DCILiteMethod):
    """AMABench wrapper: only U2 (adaptive dump) enabled."""

    method_cls = UnifiedAutoMemU2Method

class AutoMemU3Method(DCILiteMethod):
    """AMABench wrapper: only U3 (per-corpus cache) enabled."""

    method_cls = UnifiedAutoMemU3Method

class AutoMemGraphMethod(DCILiteMethod):
    """AMABench wrapper: AutoMem + U4 graph_query (Memgraph)."""

    from agentmem.methods.automem import AutoMemGraphMethod as _UG
    method_cls = _UG

