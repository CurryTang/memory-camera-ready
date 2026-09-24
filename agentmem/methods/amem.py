"""A-Mem adapter for the unified build/answer method interface.

Upstream: https://github.com/agiresearch/a-mem
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from agentmem.methods.base import BaseMethod, MeteredMethod, MethodKind
from agentmem.methods.upstream_memory_utils import count_tokens, format_mapping_items, split_trajectory_text

def _amem_source_root(source_root: str | Path | None = None) -> Path:
    if source_root is not None:
        return Path(source_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "vendor" / "a-mem"

def _load_agentic_memory_system(source_root: str | Path | None = None) -> Any:
    root = _amem_source_root(source_root)
    if not root.exists():
        raise ImportError(
            "A-Mem source was not found. Initialize the submodule with "
            "`git submodule update --init vendor/a-mem`, or pass "
            "`amem_source_root=/path/to/a-mem`."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from agentic_memory.memory_system import AgenticMemorySystem
    except Exception as exc:                                                   
        raise ImportError(
            "A-Mem could not be imported. Install its dependencies "
            "(chromadb, sentence-transformers, litellm, rank-bm25, nltk) "
            f"and ensure the source root is importable: {root}."
        ) from exc
    return AgenticMemorySystem

class AMemMethod(MeteredMethod, BaseMethod):
    kind = MethodKind.STRUCTURED
    name = "amem"

    def __init__(
        self,
        *,
        amem_source_root: str | Path | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        retrieve_k: int = 5,
        chunk_chars: int = 2400,
        chunk_overlap_chars: int = 200,
        max_chunks: int | None = None,
        raw_ingest: bool | None = None,
        token_counter: Any | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.source_root = _amem_source_root(amem_source_root)
        self.embedding_model = embedding_model
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url or os.environ.get("OPENAI_BASE_URL")
        self.llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.retrieve_k = int(retrieve_k)
        self.chunk_chars = int(chunk_chars)
        self.chunk_overlap_chars = int(chunk_overlap_chars)
        self.max_chunks = int(max_chunks) if max_chunks else None
        self.raw_ingest = (
            os.environ.get("AMEM_RAW_INGEST", "").lower() in {"1", "true", "yes"}
            if raw_ingest is None
            else bool(raw_ingest)
        )
        self._token_counter = token_counter

    def build(self, traj_text: str, *, task: str = "") -> Any:
        AgenticMemorySystem = _load_agentic_memory_system(self.source_root)
        old_base_url = os.environ.get("OPENAI_BASE_URL")
        if self.llm_base_url:
            os.environ["OPENAI_BASE_URL"] = self.llm_base_url
        try:
            memory = AgenticMemorySystem(
                model_name=self.embedding_model,
                llm_backend=self.llm_backend,
                llm_model=self.llm_model,
                api_key=self.llm_api_key,
            )
        finally:
            if self.llm_base_url:
                if old_base_url is None:
                    os.environ.pop("OPENAI_BASE_URL", None)
                else:
                    os.environ["OPENAI_BASE_URL"] = old_base_url
        chunks = split_trajectory_text(
            traj_text,
            chunk_chars=self.chunk_chars,
            overlap_chars=self.chunk_overlap_chars,
            max_chunks=self.max_chunks,
        )
        with self._counters.time_block("build_wallclock_seconds"):
            for index, chunk in enumerate(chunks):
                content = f"Task: {task}\n\n{chunk}" if task and index == 0 else chunk
                if self.raw_ingest:
                    self._add_raw_note(memory, content, index=index)
                else:
                    memory.add_note(content=content, category=task or "Trajectory")
        self._counters.family_specific["amem_chunks"] = len(chunks)
        return memory

    def memory_construction(self, traj_text: str, task: str = "") -> Any:
        return self.build(traj_text, task=task)

    def answer(self, memory: Any, question: str) -> str:
        with self._counters.time_block("wallclock_seconds"):
            results = memory.search_agentic(question, k=self.retrieve_k)
        context = format_mapping_items(results, title="A-Mem Retrieved Memories")
        self._counters.record_retrieval(
            candidates_scored=0,
            evidence_injected=len(results),
            context_tokens=count_tokens(context, self._token_counter),
        )
        return context

    def memory_retrieve(self, memory: Any, question: str) -> str:
        return self.answer(memory, question)

    @staticmethod
    def _add_raw_note(memory: Any, content: str, *, index: int) -> str:
        from agentic_memory.memory_system import MemoryNote

        note = MemoryNote(
            content=content,
            category="Trajectory",
            tags=["trajectory", "raw"],
            keywords=[],
            context=f"Raw trajectory chunk {index}",
        )
        memory.memories[note.id] = note
        memory.retriever.add_document(
            note.content,
            {
                "id": note.id,
                "content": note.content,
                "keywords": note.keywords,
                "links": note.links,
                "retrieval_count": note.retrieval_count,
                "timestamp": note.timestamp,
                "last_accessed": note.last_accessed,
                "context": note.context,
                "evolution_history": note.evolution_history,
                "category": note.category,
                "tags": note.tags,
            },
            note.id,
        )
        return note.id
