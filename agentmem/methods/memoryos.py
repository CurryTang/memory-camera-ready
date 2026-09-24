"""MemoryOS adapter for the unified build/answer method interface.

Upstream: https://github.com/BAI-LAB/MemoryOS

The upstream public response API generates an answer itself.  For comparable
experiments, this adapter uses MemoryOS' native long-term storage and retriever
to return evidence context; the shared benchmark prompt remains responsible for
final answer generation.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from agentmem.methods.base import BaseMethod, MeteredMethod, MethodKind
from agentmem.methods.upstream_memory_utils import count_tokens, format_mapping_items, split_trajectory_text

def _memoryos_source_root(source_root: str | Path | None = None) -> Path:
    if source_root is not None:
        return Path(source_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "vendor" / "MemoryOS" / "memoryos-pypi"

def _load_memoryos_cls(source_root: str | Path | None = None) -> Any:
    root = _memoryos_source_root(source_root)
    if not root.exists():
        raise ImportError(
            "MemoryOS source was not found. Initialize the submodule with "
            "`git submodule update --init vendor/MemoryOS`, or pass "
            "`memoryos_source_root=/path/to/MemoryOS/memoryos-pypi`."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from memoryos import Memoryos
    except Exception as exc:                                                   
        raise ImportError(
            "MemoryOS could not be imported. Install its dependencies "
            "(openai, sentence-transformers, faiss-cpu) and ensure the "
            f"source root is importable: {root}."
        ) from exc
    return Memoryos

class MemoryOSMethod(MeteredMethod, BaseMethod):
    kind = MethodKind.STRUCTURED
    name = "memoryos"

    def __init__(
        self,
        *,
        memoryos_source_root: str | Path | None = None,
        save_dir: str | Path | None = None,
        user_id: str | None = None,
        assistant_id: str = "agentmem_eval",
        llm_model: str = "gpt-4o-mini",
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 7,
        chunk_chars: int = 2400,
        chunk_overlap_chars: int = 200,
        max_chunks: int | None = None,
        token_counter: Any | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.source_root = _memoryos_source_root(memoryos_source_root)
        self.save_dir = Path(save_dir) if save_dir else Path(tempfile.mkdtemp(prefix="memoryos_method_"))
        self.user_id = user_id or f"agentmem_{uuid.uuid4().hex[:12]}"
        self.assistant_id = assistant_id
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url or os.environ.get("OPENAI_BASE_URL")
        self.llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.embedding_model = embedding_model
        self.retrieve_k = int(retrieve_k)
        self.chunk_chars = int(chunk_chars)
        self.chunk_overlap_chars = int(chunk_overlap_chars)
        self.max_chunks = int(max_chunks) if max_chunks else None
        self._token_counter = token_counter

    def build(self, traj_text: str, *, task: str = "") -> Any:
        Memoryos = _load_memoryos_cls(self.source_root)
        chunks = split_trajectory_text(
            traj_text,
            chunk_chars=self.chunk_chars,
            overlap_chars=self.chunk_overlap_chars,
            max_chunks=self.max_chunks,
        )
        memory = Memoryos(
            user_id=self.user_id,
            assistant_id=self.assistant_id,
            openai_api_key=self.llm_api_key,
            openai_base_url=self.llm_base_url,
            data_storage_path=str(self.save_dir),
            short_term_capacity=max(len(chunks) + 2, 16),
            retrieval_queue_capacity=self.retrieve_k,
            llm_model=self.llm_model,
            embedding_model_name=self.embedding_model,
        )
        with self._counters.time_block("build_wallclock_seconds"):
            for index, chunk in enumerate(chunks):
                content = f"Task: {task}\n\n{chunk}" if task and index == 0 else chunk
                memory.user_long_term_memory.add_user_knowledge(content)
        self._counters.family_specific["memoryos_chunks"] = len(chunks)
        self._counters.persistent_store_bytes = self.persistent_store_bytes(memory)
        return memory

    def memory_construction(self, traj_text: str, task: str = "") -> Any:
        return self.build(traj_text, task=task)

    def answer(self, memory: Any, question: str) -> str:
        with self._counters.time_block("wallclock_seconds"):
            retrieved = memory.retriever.retrieve_context(
                user_query=question,
                user_id=memory.user_id,
                top_k_knowledge=self.retrieve_k,
            )
        sections: list[str] = []
        sections.append(format_mapping_items(retrieved.get("retrieved_user_knowledge", []), title="MemoryOS User Knowledge"))
        sections.append(format_mapping_items(retrieved.get("retrieved_assistant_knowledge", []), title="MemoryOS Assistant Knowledge"))
        sections.append(format_mapping_items(retrieved.get("retrieved_pages", []), title="MemoryOS Mid-Term Pages"))
        context = "\n\n".join(section for section in sections if section).strip()
        self._counters.record_retrieval(
            candidates_scored=0,
            evidence_injected=sum(len(retrieved.get(key, []) or []) for key in ("retrieved_user_knowledge", "retrieved_assistant_knowledge", "retrieved_pages")),
            context_tokens=count_tokens(context, self._token_counter),
        )
        return context

    def memory_retrieve(self, memory: Any, question: str) -> str:
        return self.answer(memory, question)

    def persistent_store_bytes(self, memory: Any) -> int:
        root = Path(getattr(memory, "data_storage_path", self.save_dir))
        if not root.exists():
            return 0
        return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
