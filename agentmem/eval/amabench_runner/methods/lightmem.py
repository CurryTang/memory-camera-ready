"""LightMem method for AMABench and ALFWorld-style procedural trajectories."""

from __future__ import annotations

import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agentmem.eval.amabench_runner.methods.base import BaseMethod
from agentmem.methods.lightmem import LightMemMethod as UnifiedLightMemMethod
from agentmem.methods.lightmem_support import (
    PROCEDURAL_LIGHTMEM_PROMPTS,
    ensure_embedding_dims,
    format_lightmem_retrieval,
    make_lightmem_config,
    procedural_turns_from_text,
    resolve_rq1_lightmem_endpoints,
    sanitize_collection_name,
    timestamped_messages_from_turns,
)

@dataclass
class _EpisodeLightMem:
    method: UnifiedLightMemMethod
    memory: Any

class LightMemMethod(BaseMethod):
    """Benchmark-aware LightMem adapter for procedural trajectories.

    The core design choices are:
    - align input shaping with the action/observation structure used by
      AMABench and ALFWorld
    - use LightMem's event extraction mode so compressed memory keeps both
      factual state and causal/procedural relations
    - run the offline consolidation pass after ingestion, which is the part of
      LightMem that differentiates it from SimpleMem-style online extraction
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        retrieve_limit: int = 8,
        config_path: Optional[str] = None,
        storage_root: Optional[str] = None,
        collection_prefix: str = "lightmem_proc",
        pre_compress: bool = False,
        topic_segment: bool = False,
        precomp_topic_shared: bool = False,
        messages_use: str = "hybrid",
        metadata_generate: bool = False,
        text_summary: bool = False,
        extract_threshold: float = 0.1,
        extraction_mode: str = "event",
        construct_update_queue: bool = True,
        offline_update: bool = False,
        update_queue_top_k: int = 8,
        update_queue_keep_top_n: int = 4,
        update_score_threshold: float = 0.8,
        llm_max_tokens: int = 4096,
        max_turns: Optional[int] = None,
        embedding_dims: Optional[int] = None,
        **_kw,
    ) -> None:
        cfg = self._load_config(config_path) if config_path else {}
        endpoints = resolve_rq1_lightmem_endpoints(
            llm_model=llm_model or cfg.get("llm_model"),
            llm_base_url=llm_base_url or cfg.get("llm_base_url"),
            llm_api_key=llm_api_key or cfg.get("llm_api_key"),
            embedding_model=embedding_model or cfg.get("embedding_model"),
            embedding_base_url=embedding_base_url or cfg.get("embedding_base_url"),
            embedding_api_key=embedding_api_key or cfg.get("embedding_api_key"),
        )
        self.llm_model = endpoints["llm_model"]
        self.llm_base_url = endpoints["llm_base_url"]
        self.llm_api_key = endpoints["llm_api_key"]
        self.embedding_model = endpoints["embedding_model"]
        self.embedding_base_url = endpoints["embedding_base_url"]
        self.embedding_api_key = endpoints["embedding_api_key"]
        self.retrieve_limit = int(cfg.get("retrieve_limit", retrieve_limit))
        self.collection_prefix = str(cfg.get("collection_prefix", collection_prefix))
        self.storage_root = Path(
            storage_root
            or cfg.get("storage_root")
            or tempfile.mkdtemp(prefix="lightmem_proc_")
        )
        self.pre_compress = bool(cfg.get("pre_compress", pre_compress))
        self.topic_segment = bool(cfg.get("topic_segment", topic_segment))
        self.precomp_topic_shared = bool(cfg.get("precomp_topic_shared", precomp_topic_shared))
        self.messages_use = str(cfg.get("messages_use", messages_use))
        self.metadata_generate = bool(cfg.get("metadata_generate", metadata_generate))
        self.text_summary = bool(cfg.get("text_summary", text_summary))
        self.extract_threshold = float(cfg.get("extract_threshold", extract_threshold))
        self.extraction_mode = str(cfg.get("extraction_mode", extraction_mode))
        self.construct_update_queue = bool(cfg.get("construct_update_queue", construct_update_queue))
        self.offline_update = bool(cfg.get("offline_update", offline_update))
        self.update_queue_top_k = int(cfg.get("update_queue_top_k", update_queue_top_k))
        self.update_queue_keep_top_n = int(cfg.get("update_queue_keep_top_n", update_queue_keep_top_n))
        self.update_score_threshold = float(cfg.get("update_score_threshold", update_score_threshold))
        self.llm_max_tokens = int(cfg.get("llm_max_tokens", llm_max_tokens))
        env_max_turns = os.environ.get("LIGHTMEM_MAX_TURNS")
        self.max_turns = cfg.get("max_turns", max_turns)
        if env_max_turns:
            self.max_turns = int(env_max_turns)
        self.topic_segmenter_config = cfg.get("topic_segmenter")
        self.pre_compressor_config = cfg.get("pre_compressor")
        self.embedding_dims = cfg.get("embedding_dims", embedding_dims)
        self._episode_counter = 0
        self._last_trace: dict[str, Any] = {}
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def memory_construction(self, traj_text: str, task: str = "") -> _EpisodeLightMem:
        self._episode_counter += 1
        turns = procedural_turns_from_text(traj_text, task=task)
        if self.max_turns:
            turns = turns[-int(self.max_turns):]
        messages = timestamped_messages_from_turns(turns)
        collection_name = sanitize_collection_name(f"{self.collection_prefix}_{self._episode_counter}")
        dims = ensure_embedding_dims(
            model=self.embedding_model,
            base_url=self.embedding_base_url,
            api_key=self.embedding_api_key or "EMPTY",
            configured_dims=int(self.embedding_dims) if self.embedding_dims else None,
        )
        config = make_lightmem_config(
            collection_name=collection_name,
            root_dir=self.storage_root,
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            embedding_model=self.embedding_model,
            embedding_base_url=self.embedding_base_url,
            embedding_api_key=self.embedding_api_key or "EMPTY",
            embedding_dims=dims,
            pre_compress=self.pre_compress,
            topic_segment=self.topic_segment,
            precomp_topic_shared=self.precomp_topic_shared,
            messages_use=self.messages_use,
            metadata_generate=self.metadata_generate,
            text_summary=self.text_summary,
            extract_threshold=self.extract_threshold,
            extraction_mode=self.extraction_mode,
            llm_max_tokens=self.llm_max_tokens,
            topic_segmenter=self.topic_segmenter_config,
            pre_compressor=self.pre_compressor_config,
        )
        method = UnifiedLightMemMethod(
            config=config,
            retrieve_limit=self.retrieve_limit,
            metadata_generate_prompt=PROCEDURAL_LIGHTMEM_PROMPTS,
            direct_store_when_unsegmented=True,
        )
        memory = method.build_from_messages(messages, task=task)
        if self.construct_update_queue:
            memory.construct_update_queue_all_entries(
                top_k=self.update_queue_top_k,
                keep_top_n=self.update_queue_keep_top_n,
            )
        if self.offline_update:
            memory.offline_update_all_entries(score_threshold=self.update_score_threshold)
        self._last_trace = {
            "lightmem_collection": collection_name,
            "lightmem_num_messages": len(messages),
        }
        return _EpisodeLightMem(method=method, memory=memory)

    def memory_retrieve(self, memory: _EpisodeLightMem, question: str) -> str:
        results = memory.memory.retrieve(question, limit=self.retrieve_limit)
        context = format_lightmem_retrieval(
            results,
            header=(
                "LightMem memory: compressed summaries and topic-linked snippets from the "
                "trajectory. Prefer exact object names, state updates, and action outcomes."
            ),
        )
        self._last_trace = {
            **self._last_trace,
            "lightmem_question": question,
            "lightmem_retrieved_context": context,
        }
        return context

    def last_trajectory(self) -> dict[str, Any]:
        return dict(self._last_trace)
