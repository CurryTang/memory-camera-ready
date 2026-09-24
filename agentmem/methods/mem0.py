"""Mem0 adapter for the unified build/answer method interface.

The adapter prefers an installed ``mem0ai`` package and falls back to the
vendored copy under ``vendor/MemoryAgentBench``.  It returns retrieved evidence
only; benchmark runners own final answer generation for method comparability.
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from agentmem.methods.base import BaseMethod, MeteredMethod, MethodKind
from agentmem.methods.upstream_memory_utils import count_tokens, split_trajectory_text

def _known_embedding_dim(model: str) -> int:
    """Return the embedding width for a known model name."""
    lowered = str(model or "").lower()
    if "qwen3-embedding-4b" in lowered:
        return 2560
    if "qwen3-embedding-0.6b" in lowered:
        return 1024
    if "qwen3-embedding-8b" in lowered:
        return 4096
    if "text-embedding-3-large" in lowered:
        return 3072
    if "text-embedding-3-small" in lowered:
        return 1536
    if "titan-embed-text-v2" in lowered or "cohere.embed" in lowered:
        return 1024
    return 1536                             

def _mem0_source_root(source_root: str | Path | None = None) -> Path:
    if source_root is not None:
        return Path(source_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "vendor" / "MemoryAgentBench"

def _load_mem0_memory(source_root: str | Path | None = None) -> Any:
    try:
        from mem0 import Memory

        return Memory
    except Exception:
        pass

    root = _mem0_source_root(source_root)
    if not (root / "mem0").exists():
        raise ImportError(
            "Mem0 source was not found. Install mem0ai or initialize "
            "`vendor/MemoryAgentBench`."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    original_version = importlib.metadata.version

    def version_with_mem0ai_fallback(name: str) -> str:
        if name == "mem0ai":
            return "0+vendored"
        return original_version(name)

    importlib.metadata.version = version_with_mem0ai_fallback
    try:
        from mem0 import Memory
    except Exception as exc:                                             
        raise ImportError(
            "Mem0 could not be imported. Install mem0ai and its runtime "
            "dependencies, or use the vendored MemoryAgentBench source."
        ) from exc
    finally:
        importlib.metadata.version = original_version
    return Memory

class Mem0Method(MeteredMethod, BaseMethod):
    kind = MethodKind.STRUCTURED
    name = "mem0"

    def __init__(
        self,
        *,
        mem0_source_root: str | Path | None = None,
        save_dir: str | Path | None = None,
        user_id: str | None = None,
        llm_model: str = "gpt-4o-mini",
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_dims: int | None = None,
        vector_store: str = "faiss",
        retrieve_k: int = 10,
        chunk_chars: int = 2400,
        chunk_overlap_chars: int = 200,
        max_chunks: int | None = None,
        raw_ingest: bool | None = None,
        token_counter: Any | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.source_root = _mem0_source_root(mem0_source_root)
        self.save_dir = Path(save_dir) if save_dir else Path(tempfile.mkdtemp(prefix="mem0_method_"))
        self.user_id = user_id or f"agentmem_{uuid.uuid4().hex[:12]}"
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url or os.environ.get("OPENAI_BASE_URL")
        self.llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.embedding_model = embedding_model
        self.embedding_base_url = (
            embedding_base_url
            or os.environ.get("EMBEDDING_BASE_URL")
            or self.llm_base_url
        )
        self.embedding_api_key = (
            embedding_api_key
            or os.environ.get("EMBEDDING_API_KEY")
            or self.llm_api_key
        )
        self.embedding_dims = int(
            embedding_dims if embedding_dims is not None
            else _known_embedding_dim(self.embedding_model)
        )
        self.vector_store = vector_store
        self.retrieve_k = int(retrieve_k)
        self.chunk_chars = int(chunk_chars)
        self.chunk_overlap_chars = int(chunk_overlap_chars)
        self.max_chunks = int(max_chunks) if max_chunks else None
        self.raw_ingest = (
            os.environ.get("MEM0_RAW_INGEST", "").lower() in {"1", "true", "yes"}
            if raw_ingest is None
            else bool(raw_ingest)
        )
        self._token_counter = token_counter

    def build(self, traj_text: str, *, task: str = "") -> Any:
        Memory = _load_mem0_memory(self.source_root)
        memory = Memory.from_config(self._config())
        chunks = split_trajectory_text(
            traj_text,
            chunk_chars=self.chunk_chars,
            overlap_chars=self.chunk_overlap_chars,
            max_chunks=self.max_chunks,
        )
        with self._counters.time_block("build_wallclock_seconds"):
            for index, chunk in enumerate(chunks):
                content = f"Task: {task}\n\n{chunk}" if task and index == 0 else chunk
                memory.add(
                    [{"role": "user", "content": content}],
                    user_id=self.user_id,
                    infer=not self.raw_ingest,
                )
        self._counters.family_specific["mem0_chunks"] = len(chunks)
        self._counters.persistent_store_bytes = self.persistent_store_bytes(memory)
        return memory

    def memory_construction(self, traj_text: str, task: str = "") -> Any:
        return self.build(traj_text, task=task)

    def answer(self, memory: Any, question: str) -> str:
        with self._counters.time_block("wallclock_seconds"):
            retrieved = memory.search(
                query=question,
                user_id=self.user_id,
                limit=self.retrieve_k,
            )
        results = retrieved.get("results", retrieved) if isinstance(retrieved, dict) else retrieved
        context = self._format_results(results)
        self._counters.record_retrieval(
            candidates_scored=0,
            evidence_injected=len(results or []),
            context_tokens=count_tokens(context, self._token_counter),
        )
        return context

    def memory_retrieve(self, memory: Any, question: str) -> str:
        return self.answer(memory, question)

    def persistent_store_bytes(self, memory: Any) -> int:
        if not self.save_dir.exists():
            return 0
        return sum(path.stat().st_size for path in self.save_dir.rglob("*") if path.is_file())

    def _config(self) -> dict[str, Any]:
        collection = f"mem0_{uuid.uuid4().hex[:10]}"
        if self.vector_store == "chroma":
            vector_store = {
                "provider": "chroma",
                "config": {
                    "collection_name": collection,
                    "path": str(self.save_dir / "chroma"),
                },
            }
        else:
            vector_store = {
                "provider": "faiss",
                "config": {
                    "collection_name": collection,
                    "path": str(self.save_dir / "faiss"),
                    "embedding_model_dims": self.embedding_dims,
                    "distance_strategy": "cosine",
                    "normalize_L2": True,
                },
            }
        return {
            "version": "v1.1",
            "history_db_path": str(self.save_dir / "history.db"),
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.llm_model,
                    "api_key": self.llm_api_key,
                    "openai_base_url": self.llm_base_url,
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.embedding_model,
                    "api_key": self.embedding_api_key,
                    "openai_base_url": self.embedding_base_url,
                    "embedding_dims": self.embedding_dims,
                },
            },
            "vector_store": vector_store,
        }

    @staticmethod
    def _format_results(results: Any) -> str:
        if not results:
            return ""
        lines = ["# Mem0 Retrieved Memories"]
        count = 0
        for index, item in enumerate(results, start=1):
            if isinstance(item, dict):
                content = str(item.get("memory") or item.get("content") or item.get("text") or "").strip()
                score = item.get("score")
            else:
                content = str(item).strip()
                score = None
            if not content:
                continue
            count += 1
            suffix = f" (score={score})" if score is not None else ""
            lines.append(f"[{index}]{suffix}\n{content}")
        return "\n\n".join(lines) if count else ""
