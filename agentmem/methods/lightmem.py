"""LightMem adapter (zjunlp/LightMem) for the 3-factor evaluation.

Paper: https://arxiv.org/abs/2510.18866 — summary-at-write-time memory with
optional pre-compression, topic segmentation, and hybrid retrieval. Falls in
the :class:`MethodKind.SUMMARY` bucket alongside SimpleMem.

Upstream source is tracked as a git submodule at ``vendor/lightmem`` and
installed editably via pixi. This wrapper delegates the heavy lifting to
``lightmem.memory.lightmem.LightMemory`` while populating the unified
:class:`EfficiencyCounters` so LightMem results are directly comparable to the
rest of the method fleet.
"""

from __future__ import annotations

import json
import copy
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agentmem.methods.base import (
    BaseMethod,
    EfficiencyCounters,
    MeteredMethod,
    MethodKind,
)
from agentmem.methods.lightmem_support import (
    document_messages_from_text,
    ensure_lightmem_openai_config_compat,
    ensure_embedding_dims,
    make_lightmem_config,
    resolve_rq1_lightmem_endpoints,
    sanitize_collection_name,
    timestamped_messages_from_turns,
)
from agentmem.methods.lightmem_compat import (
    load_lightmemory_cls,
    load_memory_entry_cls,
)

def _load_lightmemory_cls() -> Any:
    """Import LightMemory lazily so the module is import-safe without the dep."""
    try:
        LightMemory = load_lightmemory_cls()
    except ImportError as exc:                                                   
        raise ImportError(
            "LightMem could not be resolved to a compatible LightMemory API. "
            "Install zjunlp/LightMem from PyPI/source, or provide a vendored "
            "package exposing lightmem.memory.lightmem.LightMemory."
        ) from exc
    ensure_lightmem_openai_config_compat()
    return LightMemory

class LightMemMethod(MeteredMethod, BaseMethod):
    """Thin adapter around upstream LightMemory.

    Notes on instrumentation (the reason we don't just call LightMemory directly):

    - ``add_memory`` returns a dict that includes ``api_call_nums``,
      ``add_input_prompt``, ``add_output_prompt``. We use those to populate
      build-phase LLM counters without re-running the pipeline.
    - ``retrieve`` returns a list of evidence strings. Counted as one
      retrieval call, with ``evidence_units_injected`` = len(results) and
      ``retrieved_context_tokens`` estimated by the caller-provided
      token_counter.
    - Final answer generation is left to the benchmark's shared QA prompt,
      so query-phase LLM token accounting happens in the runner, not here.
    """

    kind = MethodKind.SUMMARY
    name = "lightmem"

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_dims: Optional[int] = None,
        storage_root: Optional[str] = None,
        collection_prefix: str = "lightmem",
        pre_compress: bool = False,
        topic_segment: bool = False,
        precomp_topic_shared: bool = False,
        messages_use: str = "hybrid",
        metadata_generate: bool = False,
        text_summary: bool = False,
        extract_threshold: float = 0.1,
        extraction_mode: str = "flat",
        llm_max_tokens: int = 4096,
        document_chunk_chars: int = 2800,
        document_overlap_chars: int = 250,
        document_chunk_tokens: int = 32000,
        document_overlap_tokens: int = 256,
        direct_store_when_unsegmented: bool = True,
        retrieve_limit: int = 10,
        token_counter: Optional[Any] = None,
        metadata_generate_prompt: Optional[Any] = None,
    ) -> None:
        super().__init__()
        cfg = self._load_config(config_path) if config_path else {}
        if config is None and config_path:
            config = cfg if self._looks_like_runtime_config(cfg) else None
        if config is None:
            endpoints = resolve_rq1_lightmem_endpoints(
                llm_model=llm_model or cfg.get("llm_model"),
                llm_base_url=llm_base_url or cfg.get("llm_base_url"),
                llm_api_key=llm_api_key or cfg.get("llm_api_key"),
                embedding_model=embedding_model or cfg.get("embedding_model"),
                embedding_base_url=embedding_base_url or cfg.get("embedding_base_url"),
                embedding_api_key=embedding_api_key or cfg.get("embedding_api_key"),
            )
            dims = ensure_embedding_dims(
                model=endpoints["embedding_model"],
                base_url=endpoints["embedding_base_url"],
                api_key=endpoints["embedding_api_key"],
                configured_dims=int(cfg.get("embedding_dims") or embedding_dims or 0) or None,
            )
            config = make_lightmem_config(
                collection_name=sanitize_collection_name(
                    str(cfg.get("collection_prefix", collection_prefix))
                ),
                root_dir=storage_root or cfg.get("storage_root"),
                llm_model=endpoints["llm_model"],
                llm_base_url=endpoints["llm_base_url"],
                llm_api_key=endpoints["llm_api_key"],
                embedding_model=endpoints["embedding_model"],
                embedding_base_url=endpoints["embedding_base_url"],
                embedding_api_key=endpoints["embedding_api_key"],
                embedding_dims=dims,
                pre_compress=bool(cfg.get("pre_compress", pre_compress)),
                topic_segment=bool(cfg.get("topic_segment", topic_segment)),
                precomp_topic_shared=bool(cfg.get("precomp_topic_shared", precomp_topic_shared)),
                messages_use=str(cfg.get("messages_use", messages_use)),
                metadata_generate=bool(cfg.get("metadata_generate", metadata_generate)),
                text_summary=bool(cfg.get("text_summary", text_summary)),
                extract_threshold=float(cfg.get("extract_threshold", extract_threshold)),
                extraction_mode=str(cfg.get("extraction_mode", extraction_mode)),
                llm_max_tokens=int(cfg.get("llm_max_tokens", llm_max_tokens)),
                topic_segmenter=cfg.get("topic_segmenter"),
                pre_compressor=cfg.get("pre_compressor"),
            )
        self._raw_config: Mapping[str, Any] = dict(config)
        self._retrieve_limit = int(retrieve_limit)
        self._token_counter = token_counter
        self._metadata_prompt = metadata_generate_prompt
        self._document_chunk_chars = int(cfg.get("document_chunk_chars", document_chunk_chars))
        self._document_overlap_chars = int(cfg.get("document_overlap_chars", document_overlap_chars))
        self._document_chunk_tokens = int(
            cfg.get("document_chunk_tokens")
            or os.environ.get("LIGHTMEM_MAB_DOCUMENT_CHUNK_TOKENS")
            or document_chunk_tokens
        )
        self._document_overlap_tokens = int(
            cfg.get("document_overlap_tokens")
            or os.environ.get("LIGHTMEM_MAB_DOCUMENT_OVERLAP_TOKENS")
            or document_overlap_tokens
        )
        os.environ.setdefault("LIGHTMEM_EMBED_MAX_TOKENS", str(self._document_chunk_tokens))
        self._direct_store_when_unsegmented = bool(
            cfg.get("direct_store_when_unsegmented", direct_store_when_unsegmented)
        )
        self._episode_counter = 0
        self._memory: Optional[Any] = None                              

    def build(self, traj_text: str, *, task: str = "") -> Any:
        LightMemory = _load_lightmemory_cls()
        memory = LightMemory.from_config(self._next_runtime_config())
        if len(str(traj_text or "")) > self._document_chunk_chars * 2:
            messages = document_messages_from_text(
                traj_text,
                domain=task,
                chunk_chars=self._document_chunk_chars,
                overlap_chars=self._document_overlap_chars,
            )
        else:
            messages = self._traj_text_to_messages(traj_text, task=task)
        messages = self._embedding_safe_messages(messages, task=task)

        with self._counters.time_block("build_wallclock_seconds"):
            result = memory.add_memory(
                messages,
                METADATA_GENERATE_PROMPT=self._metadata_prompt,
                force_segment=True,
                force_extract=True,
            )

        self._store_unsegmented_messages(memory, result)
        self._account_add_memory(result)
        return memory

    def memory_construction(self, traj_text: str, task: str = "") -> Any:
        return self.build(traj_text, task=task)

    def memory_retrieve(self, memory: Any, question: str) -> str:
        return self.answer(memory, question)

    def answer(self, memory: Any, question: str) -> str:

        results = memory.retrieve(question, limit=self._retrieve_limit)
        context = self._format_evidence(results)
        ctx_tokens = self._count_tokens(context)
        self._counters.record_retrieval(
            candidates_scored=0,                                            
            evidence_injected=len(results) if isinstance(results, Sequence) else 0,
            context_tokens=ctx_tokens,
        )
        return context

    def persistent_store_bytes(self, memory: Any) -> int:

        try:
            path = getattr(getattr(memory, "config", None), "storage_path", None)
            if path and Path(path).exists():
                return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        except Exception:
            pass
        return 0

    @staticmethod
    def _traj_text_to_messages(traj_text: str, *, task: str) -> list[dict[str, str]]:
        """Minimal adapter from a flat trajectory transcript to LightMem messages.

        LightMem expects a list of ``{"role": ..., "content": ...}`` dicts.
        For AMABench / LoCoMo transcripts we pass the whole trajectory as a
        single user message tagged with the task for context. Benchmark-
        specific runners can override this by calling
        :meth:`build_from_messages` instead of :meth:`build`.
        """
        header = f"[task] {task}\n" if task else ""
        text = header + str(traj_text)
        return timestamped_messages_from_turns([("user", text)])

    def build_from_messages(
        self, messages: Sequence[Mapping[str, Any]], *, task: str = ""
    ) -> Any:
        """Benchmark-specific entry point: caller-controlled message list."""
        LightMemory = _load_lightmemory_cls()
        memory = LightMemory.from_config(self._next_runtime_config())
        safe_messages = self._embedding_safe_messages(messages, task=task)
        with self._counters.time_block("build_wallclock_seconds"):
            result = memory.add_memory(
                safe_messages,
                METADATA_GENERATE_PROMPT=self._metadata_prompt,
                force_segment=True,
                force_extract=True,
            )
        self._store_unsegmented_messages(memory, result)
        self._account_add_memory(result)
        return memory

    def build_from_document_text(
        self,
        text: str,
        *,
        task: str = "",
        domain: str = "",
        sub_domain: str = "",
    ) -> Any:
        messages = document_messages_from_text(
            text,
            domain=domain or task,
            sub_domain=sub_domain,
            chunk_chars=self._document_chunk_chars,
            overlap_chars=self._document_overlap_chars,
        )
        return self.build_from_messages(messages, task=task)

    def _embedding_safe_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        task: str = "",
    ) -> list[dict[str, Any]]:
        """Split oversized message contents before LightMem embeds raw entries."""
        token_cap = max(512, int(self._document_chunk_tokens or 32000))
        overlap = max(0, min(int(self._document_overlap_tokens or 0), token_cap // 2))
        safe: list[dict[str, Any]] = []
        for msg in messages:
            content = str((msg or {}).get("content") or "")
            chunks = self._split_text_by_tokens(content, token_cap=token_cap, overlap=overlap)
            if len(chunks) <= 1:
                safe.append(dict(msg))
                continue
            for idx, chunk in enumerate(chunks, start=1):
                cloned = dict(msg)
                prefix = f"[{task}] " if task and idx == 1 else ""
                cloned["content"] = f"{prefix}[chunk {idx}/{len(chunks)}]\n{chunk}"
                safe.append(cloned)
        return safe

    @staticmethod
    def _split_text_by_tokens(text: str, *, token_cap: int, overlap: int) -> list[str]:
        text = str(text or "")
        if not text:
            return [text]
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            if len(tokens) <= token_cap:
                return [text]
            step = max(1, token_cap - overlap)
            chunks: list[str] = []
            for start in range(0, len(tokens), step):
                chunk = enc.decode(tokens[start : start + token_cap]).strip()
                if chunk:
                    chunks.append(chunk)
                if start + token_cap >= len(tokens):
                    break
            return chunks or [text[: token_cap * 4]]
        except Exception:
            char_cap = max(2048, token_cap * 4)
            if len(text) <= char_cap:
                return [text]
            overlap_chars = min(max(0, overlap * 4), char_cap // 2)
            step = max(1, char_cap - overlap_chars)
            chunks = []
            for start in range(0, len(text), step):
                chunk = text[start : start + char_cap].strip()
                if chunk:
                    chunks.append(chunk)
                if start + char_cap >= len(text):
                    break
            return chunks or [text[:char_cap]]

    def _account_add_memory(self, add_result: Any) -> None:
        """Pull LightMem's ``add_memory`` return dict into build-phase counters."""
        if not isinstance(add_result, Mapping):
            return
        input_prompts = add_result.get("add_input_prompt") or []
        output_prompts = add_result.get("add_output_prompt") or []

        for i, prompt in enumerate(input_prompts):
            out = output_prompts[i] if i < len(output_prompts) else ""
            self._counters.record_llm_call(
                prompt_tokens=self._count_tokens(prompt),
                completion_tokens=self._count_tokens(out),
                phase="build",
            )

        api_calls = int(add_result.get("api_call_nums", 0) or 0)
        if api_calls > self._counters.llm_calls:

            deficit = api_calls - self._counters.llm_calls
            for _ in range(deficit):
                self._counters.record_llm_call(
                    prompt_tokens=0, completion_tokens=0, phase="build"
                )

    @staticmethod
    def _looks_like_runtime_config(config: Mapping[str, Any]) -> bool:
        return "memory_manager" in config or "text_embedder" in config

    def _store_unsegmented_messages(self, memory: Any, add_result: Any) -> None:
        """Persist raw messages when upstream segmentation is disabled.

        Upstream LightMem returns normalized messages without storing them when
        ``topic_segment`` is false. RQ1 uses that mode for long procedural
        traces to avoid a heavy local LLMLingua segmenter, so the repo-owned
        adapter stores those messages as timestamped LightMem entries.
        """
        if not self._direct_store_when_unsegmented:
            return
        if not isinstance(add_result, Mapping):
            return
        emitted = add_result.get("emitted_messages") or []
        if not emitted:
            return
        try:
            MemoryEntry = load_memory_entry_cls()
        except ImportError:
            return
        entries = []
        for index, msg in enumerate(emitted):
            content = str((msg or {}).get("content") or "").strip()
            if not content:
                continue
            role = str((msg or {}).get("role") or "user")
            entries.append(
                MemoryEntry(
                    time_stamp=str((msg or {}).get("time_stamp") or ""),
                    float_time_stamp=float(index),
                    weekday=str((msg or {}).get("weekday") or ""),
                    memory=f"{role}: {content}",
                    original_memory=content,
                    compressed_memory=content,
                    speaker_id=role,
                    speaker_name=role,
                    topic_id=0,
                    topic_summary="unsegmented",
                )
            )
        if entries:
            memory.offline_update(entries)

    def _next_runtime_config(self) -> dict[str, Any]:
        self._episode_counter += 1
        config = copy.deepcopy(dict(self._raw_config))
        if self._episode_counter <= 1:
            return config
        suffix = f"_{self._episode_counter}"
        for key in ("embedding_retriever", "summary_retriever"):
            retriever_cfg = ((config.get(key) or {}).get("configs") or {})
            collection = retriever_cfg.get("collection_name")
            if collection:
                retriever_cfg["collection_name"] = sanitize_collection_name(
                    f"{collection}{suffix}"
                )
            path = retriever_cfg.get("path")
            if path:
                path_obj = Path(path)
                retriever_cfg["path"] = str(path_obj.with_name(f"{path_obj.name}{suffix}"))
        return config

    @staticmethod
    def _format_evidence(results: Any) -> str:
        if isinstance(results, str):
            return results
        if isinstance(results, Sequence):
            return "\n".join(str(r) for r in results)
        return str(results)

    def _count_tokens(self, text: Any) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0

        return len(str(text or "").split())
