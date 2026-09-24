"""AMA-Agent method for AMAbench.

The paper description and the released AMA-Bench config are not fully aligned.
By default we use paper-faithful settings here:
  - retrieval_mode="embed"
  - top_k=5
  - chunk_size=8192
  - causal=true
  - enable_tools=true
  - temperature=0
  - max_tokens=16384  # paper config; 4096 truncates causal-graph + tool outputs

To reproduce the released config semantics explicitly, pass
``--method-config datasets/amabench/code/configs/ama_agent.yaml``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

from agentmem.eval.amabench_runner.methods.base import BaseMethod
from agentmem.retrieval.ama_agent import construct_state_memory, memory_retrieve

class AMAAgentMethod(BaseMethod):
    """AMA-Agent with causality graph + tool-augmented retrieval.

    Per paper: uses embedding-based node retrieval as primary path,
    with tool-augmented search as fallback when evidence is insufficient.
    """

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        retrieval_mode: str = "embed",
        top_k: int = 5,
        neighbor_radius: int = 0,
        chunk_size: int = 8192,
        causal: bool = True,
        enable_tools: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        embedding_engine: Optional[Any] = None,
        retrieval_prefilter_k: Optional[int] = None,
        causal_graph_max_edges: int = 24,
        tool_retrieval_scope: str = "full",
        tool_retrieval_prefilter_k: Optional[int] = None,
        tool_retrieval_max_results: int = 5,
        context_guardrail: Optional[str] = None,
        prepend_objective: bool = False,
        config_path: Optional[str] = None,
        **_kw: Any,
    ) -> None:
        if config_path:
            cfg = self._load_config(config_path)
            llm_model = llm_model or cfg.get("llm_model", cfg.get("model"))
            llm_base_url = llm_base_url or cfg.get("llm_base_url", cfg.get("base_url"))
            llm_api_key = llm_api_key or cfg.get("llm_api_key", cfg.get("api_key"))
            retrieval_mode = cfg.get("retrieval_mode", retrieval_mode)
            top_k = int(cfg.get("top_k", top_k))
            neighbor_radius = int(cfg.get("neighbor_radius", neighbor_radius))
            chunk_size = int(cfg.get("chunk_size", chunk_size))
            causal = bool(cfg.get("causal", causal))
            enable_tools = bool(cfg.get("enable_tools", enable_tools))
            temperature = float(cfg.get("temperature", temperature))
            max_tokens = int(cfg.get("max_tokens", max_tokens))
            if cfg.get("retrieval_prefilter_k") is not None:
                retrieval_prefilter_k = int(cfg["retrieval_prefilter_k"])
            causal_graph_max_edges = int(
                cfg.get("causal_graph_max_edges", causal_graph_max_edges)
            )
            tool_retrieval_scope = str(
                cfg.get("tool_retrieval_scope", tool_retrieval_scope)
            )
            if cfg.get("tool_retrieval_prefilter_k") is not None:
                tool_retrieval_prefilter_k = int(cfg["tool_retrieval_prefilter_k"])
            tool_retrieval_max_results = int(
                cfg.get("tool_retrieval_max_results", tool_retrieval_max_results)
            )
            context_guardrail = cfg.get("context_guardrail", context_guardrail)
            prepend_objective = bool(cfg.get("prepend_objective", prepend_objective))
            embed_cfg = cfg.get("embedding_engine") or {}
            embedding_model = embedding_model or embed_cfg.get("model_name")
            embedding_base_url = embedding_base_url or embed_cfg.get("base_url")
            embedding_api_key = embedding_api_key or embed_cfg.get("api_key")

        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.neighbor_radius = neighbor_radius
        self.chunk_size = chunk_size
        self.causal = causal
        self.enable_tools = enable_tools
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieval_prefilter_k = retrieval_prefilter_k
        self.causal_graph_max_edges = causal_graph_max_edges
        self.tool_retrieval_scope = tool_retrieval_scope
        self.tool_retrieval_prefilter_k = tool_retrieval_prefilter_k
        self.tool_retrieval_max_results = tool_retrieval_max_results
        self.context_guardrail = context_guardrail
        self.prepend_objective = prepend_objective
        self._last_trajectory: Optional[dict[str, Any]] = None

        from agentmem.eval.amabench_runner.model_client import ModelClient

        self._model_client = ModelClient(
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            max_tokens=max_tokens,
        )

        if embedding_engine is not None:
            self._embed_engine = embedding_engine
        elif embedding_model and embedding_base_url:
            self._embed_engine = self._make_embed_fn(
                embedding_model, embedding_base_url,
                embedding_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            )
        else:
            self._embed_engine = None

            if self.retrieval_mode == "embed":
                print(
                    "[ama-agent WARNING] no embedding endpoint configured; "
                    "falling back to retrieval_mode='qwen' (LLM scoring). "
                    "Paper-faithful runs require embedding_model + "
                    "embedding_base_url to be set.",
                    flush=True,
                )
                self.retrieval_mode = "qwen"

    @staticmethod
    def _make_embed_fn(model: str, base_url: str, api_key: str):
        """Create an embedding function supporting both single text and batch inputs."""
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
        debug_timing = os.getenv("AMABENCH_DEBUG_TIMING") == "1"
        max_chars = max(512, int(os.getenv("AMA_EMBED_MAX_CHARS", "12000")))

        def prepare_text(value: Any) -> str:
            text = "" if value is None else str(value)
            if len(text) > max_chars:
                return text[:max_chars]
            return text

        def embed(text):
            """Accept str (single) or list[str] (batch) input."""
            batch_size = len(text) if isinstance(text, list) else 1
            if debug_timing:
                print(
                    f"[amabench-debug] embed start model={model} batch_size={batch_size}",
                    flush=True,
            )
            t0 = time.perf_counter()
            if isinstance(text, list):

                resp = client.embeddings.create(
                    input=[prepare_text(item) for item in text],
                    model=model,
                )

                result = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
            else:
                resp = client.embeddings.create(input=[prepare_text(text)], model=model)
                result = resp.data[0].embedding
            if debug_timing:
                print(
                    f"[amabench-debug] embed done model={model} batch_size={batch_size} "
                    f"latency_sec={time.perf_counter() - t0:.2f}",
                    flush=True,
                )
            return result

        return embed

    def _call_llm(self, prompt: str) -> tuple[None, str]:
        response = self._model_client.query(prompt, temperature=self.temperature)
        return None, response

    def memory_construction(self, traj_text: str, task: str = "") -> dict[str, Any]:
        """Stage 1: construct state memory + causality graph + embeddings."""
        return construct_state_memory(
            trajectory_text=traj_text,
            task=task,
            call_llm_func=self._call_llm,
            chunk_size=self.chunk_size,
            embed_engine=self._embed_engine,
            causal=self.causal,
        )

    def memory_retrieve(self, memory: dict[str, Any], question: str) -> str:
        """Stage 2: embedding retrieval → sufficiency check → tool search."""
        context, debug = memory_retrieve(
            memory=memory,
            question=question,
            call_llm_func=self._call_llm,
            top_k=self.top_k,
            neighbor_radius=self.neighbor_radius,
            retrieval_mode=self.retrieval_mode,
            embed_engine=self._embed_engine,
            enable_tools=self.enable_tools,
            model_client=self._model_client,
            retrieval_prefilter_k=self.retrieval_prefilter_k,
            causal_graph_max_edges=self.causal_graph_max_edges,
            tool_retrieval_scope=self.tool_retrieval_scope,
            tool_retrieval_prefilter_k=self.tool_retrieval_prefilter_k,
            tool_retrieval_max_results=self.tool_retrieval_max_results,
            context_guardrail=self.context_guardrail,
            prepend_objective=self.prepend_objective,
            return_debug=True,
        )
        self._last_trajectory = debug
        return context

    def last_trajectory(self) -> Optional[dict[str, Any]]:
        return self._last_trajectory
