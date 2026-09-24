"""MemRL adapter (vendor/memrl) for the 3-factor evaluation.

Paper: MemRL — procedural memory via Build/Retrieve/Update strategies backed
by the MemOS key-value store. The agent stores task trajectories (raw or
LLM-summarised) and retrieves them by embedding similarity at decision time.

Upstream source is tracked as a git submodule at ``vendor/memrl`` and
installed editably via pixi. This wrapper delegates the heavy lifting to
``memrl.service.MemoryService`` while populating the unified
:class:`EfficiencyCounters` so MemRL results are directly comparable to the
rest of the method fleet.
"""

from __future__ import annotations

import json
import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

from agentmem.methods.base import (
    BaseMethod,
    EfficiencyCounters,
    MeteredMethod,
    MethodKind,
)

def _load_memory_service_cls() -> Any:
    """Import MemoryService lazily so the module is import-safe without the dep."""
    root = Path(__file__).resolve().parents[2]
    vendor = root / "vendor" / "memrl"
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    if importlib.util.find_spec("memos") is None:
        raise ImportError(
            f"MemRL upstream dependency 'memos' is not importable. Root: {vendor}."
        )
    try:
        from memrl.service import MemoryService
    except ImportError as exc:
        raise ImportError(
            "MemRL is not importable. Ensure the submodule at vendor/memrl "
            f"is installed via pixi (editable install configured in pixi.toml). Root: {vendor}. "
            f"Underlying import error: {exc}"
        ) from exc
    return MemoryService

class MemRLMethod(MeteredMethod, BaseMethod):
    """Thin adapter around upstream MemRL MemoryService.

    Instrumentation notes:

    - ``build_memory`` ingests a trajectory string into the MemOS store.
      We wrap the call with wallclock timing and record it as a build-phase
      operation.
    - ``retrieve`` returns a list of memory dicts. We count retrieval calls
      and estimate context tokens via the caller-provided token_counter.
    - Final answer generation is left to the benchmark's shared QA prompt,
      so query-phase LLM token accounting happens in the runner, not here.
    """

    kind = MethodKind.STRUCTURED
    name = "memrl"

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[str] = None,
        retrieve_k: int = 3,
        token_counter: Optional[Any] = None,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: str = "http://localhost:30000/v1",
        llm_api_key: Optional[str] = None,
        embedding_model: str = "Qwen3-Embedding-4B",
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        save_dir: Optional[str] = None,
        **service_kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None and config_path:
            config = self._load_config(config_path)
        self._raw_config: Mapping[str, Any] = dict(config) if config else {}
        self._retrieve_k = int(retrieve_k)
        self._token_counter = token_counter
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.embedding_api_key = embedding_api_key or os.environ.get("EMBEDDING_API_KEY", "EMPTY")
        self.embedding_dim = int(embedding_dim) if embedding_dim else self._known_embedding_dim(embedding_model)
        self.save_dir = save_dir or tempfile.mkdtemp(prefix="memrl_method_")
        self._service_kwargs = service_kwargs

    def build(self, traj_text: str, *, task: str = "") -> Any:
        traj_text = self._truncate_build_trajectory(traj_text)
        try:
            MemoryService = _load_memory_service_cls()
        except ImportError:
            if os.environ.get("MEMRL_ALLOW_REPO_RUNTIME_FALLBACK", "1") not in {"1", "true", "TRUE", "yes"}:
                raise
            return self._build_repo_runtime(traj_text, task=task)

        init_kwargs = self._default_service_kwargs()
        init_kwargs.update(dict(self._raw_config))
        init_kwargs.update(self._service_kwargs)
        service = MemoryService(**init_kwargs)

        with self._counters.time_block("build_wallclock_seconds"):
            mem_id = service.build_memory(
                task_description=task,
                trajectory=traj_text,
            )

        self._counters.record_llm_call(
            prompt_tokens=self._count_tokens(traj_text),
            completion_tokens=0,
            phase="build",
        )
        return {"service": service, "mem_id": mem_id}

    def answer(self, memory: Any, question: str) -> str:
        if isinstance(memory, dict) and memory.get("repo_runtime_fallback"):
            return self._answer_repo_runtime(memory, question)
        service = memory["service"]

        with self._counters.time_block("wallclock_seconds"):
            results = service.retrieve(
                task_description=question,
                k=self._retrieve_k,
            )

        context = self._format_evidence(results)
        ctx_tokens = self._count_tokens(context)
        self._counters.record_retrieval(
            candidates_scored=0,
            evidence_injected=len(results) if isinstance(results, list) else 0,
            context_tokens=ctx_tokens,
        )
        return context

    def _build_repo_runtime(self, traj_text: str, *, task: str = "") -> Any:
        """Fallback when vendor MemRL exists but optional MemOS deps are absent."""
        from agentmem.backends.episodic import EpisodicMemoryStore
        from agentmem.memrl.runtime import MemRLConfig, MemRLRuntimeEngine

        cfg = MemRLConfig(
            phase1_topk=max(1, int(self._raw_config.get("phase1_topk", 20))),
            topk=max(1, int(self._raw_config.get("topk", self._retrieve_k))),
            epsilon=float(self._raw_config.get("epsilon", 0.1)),
            alpha=float(self._raw_config.get("alpha", 0.3)),
            gamma=float(self._raw_config.get("gamma", 0.0)),
            similarity_weight=float(self._raw_config.get("similarity_weight", 0.7)),
            utility_weight=float(self._raw_config.get("utility_weight", 0.3)),
            use_zscore_normalization=bool(self._raw_config.get("use_zscore_normalization", True)),
        )
        store = EpisodicMemoryStore()
        runtime = MemRLRuntimeEngine(store=store, config=cfg)
        chunks = self._split_experiences(traj_text)
        with self._counters.time_block("build_wallclock_seconds"):
            for idx, chunk in enumerate(chunks):
                runtime.add_experience(
                    intent=f"{task or 'trajectory'} chunk {idx}",
                    experience=chunk,
                    success=True,
                    metadata={"source": "repo_runtime_fallback", "chunk_idx": idx, "task": task},
                    task_id=f"chunk_{idx}",
                )
        self._counters.record_llm_call(
            prompt_tokens=self._count_tokens(traj_text),
            completion_tokens=0,
            phase="build",
        )
        return {
            "repo_runtime_fallback": True,
            "runtime": runtime,
            "config": cfg,
            "num_chunks": len(chunks),
        }

    def _answer_repo_runtime(self, memory: Any, question: str) -> str:
        runtime = memory["runtime"]
        cfg = memory["config"]
        with self._counters.time_block("wallclock_seconds"):
            result = runtime.retrieve(
                question,
                phase1_topk=cfg.phase1_topk,
                topk=cfg.topk,
            )
        selected = result.selected or result.candidates[: cfg.topk]
        context = "\n\n".join(str(getattr(candidate, "content", "") or "") for candidate in selected)
        self._counters.record_retrieval(
            candidates_scored=len(result.candidates),
            evidence_injected=len(selected),
            context_tokens=self._count_tokens(context),
        )
        return context

    def persistent_store_bytes(self, memory: Any) -> int:

        return 0

    def _default_service_kwargs(self) -> dict[str, Any]:
        root = Path(self.save_dir)
        root.mkdir(parents=True, exist_ok=True)
        runtime_dir = Path(tempfile.mkdtemp(prefix="memrl_runtime_", dir=str(root)))

        from memrl.providers.embedding import OpenAIEmbedder
        from memrl.providers.llm import OpenAILLM
        from memrl.service.strategies import StrategyConfiguration

        mos_config = {
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": self.llm_model,
                    "api_key": self.llm_api_key,
                    "api_base": self.llm_base_url,
                },
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": self.llm_model,
                            "api_key": self.llm_api_key,
                            "api_base": self.llm_base_url,
                        },
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "model_name_or_path": self.embedding_model,
                            "api_key": self.embedding_api_key,
                            "base_url": self.embedding_base_url,
                        },
                    },
                    "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
                },
            },
            "user_manager": {
                "backend": "sqlite",
                "config": {"db_path": str(runtime_dir / "users.db")},
            },
            "top_k": max(1, self._retrieve_k),
        }
        mos_config_path = runtime_dir / "mos_config.json"
        mos_config_path.write_text(json.dumps(mos_config), encoding="utf-8")

        return {
            "mos_config_path": str(mos_config_path),
            "llm_provider": OpenAILLM(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                model=self.llm_model,
                default_temperature=0.0,
                default_max_tokens=int(os.environ.get("MEMRL_MAX_TOKENS", "128")),
                token_log_dir=str(runtime_dir),
            ),
            "embedding_provider": OpenAIEmbedder(
                api_key=self.embedding_api_key,
                base_url=self.embedding_base_url,
                model=self.embedding_model,
                token_log_dir=str(runtime_dir),
            ),
            "strategy_config": StrategyConfiguration.main_combination(),
            "user_id": f"memrl_{os.getpid()}",
            "num_workers": 2,
            "db_max_concurrency": 2,
            "vector_dimension": self.embedding_dim,
        }

    @staticmethod
    def _format_evidence(results: Any) -> str:
        if isinstance(results, str):
            return results
        if isinstance(results, list):
            parts = []
            for r in results:
                if isinstance(r, dict):
                    parts.append(r.get("memory", r.get("content", str(r))))
                else:
                    parts.append(str(r))
            return "\n".join(parts)
        return str(results)

    def _count_tokens(self, text: Any) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0
        return len(str(text or "").split())

    @staticmethod
    def _known_embedding_dim(model: str) -> int:
        lowered = str(model or "").lower()
        if "qwen3-embedding-4b" in lowered:
            return 2560
        if "text-embedding-3-large" in lowered:
            return 3072
        if "text-embedding-3-small" in lowered:
            return 1536
        return 3072

    @staticmethod
    def _truncate_build_trajectory(text: str) -> str:
        max_tokens = int(os.environ.get("MEMRL_MAX_TRAJECTORY_TOKENS", "30000"))
        if max_tokens <= 0:
            return text
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(str(text or ""))
            if len(tokens) <= max_tokens:
                return str(text or "")
            return enc.decode(tokens[:max_tokens])
        except Exception:
            max_chars = int(os.environ.get("MEMRL_MAX_TRAJECTORY_CHARS", str(max_tokens * 4)))
            rendered = str(text or "")
            return rendered[:max_chars] if len(rendered) > max_chars else rendered

    @staticmethod
    def _split_experiences(text: str) -> list[str]:
        max_chars = max(512, int(os.environ.get("MEMRL_REPO_FALLBACK_CHUNK_CHARS", "6000")))
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for block in re.split(r"\n(?=(?:Turn|Step)\s+\d+\s*:)", str(text or "")):
            block = block.strip()
            if not block:
                continue
            if current and current_len + len(block) > max_chars:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            if len(block) > max_chars:
                for start in range(0, len(block), max_chars):
                    piece = block[start : start + max_chars].strip()
                    if piece:
                        chunks.append(piece)
                continue
            current.append(block)
            current_len += len(block)
        if current:
            chunks.append("\n".join(current))
        return chunks or [str(text or "").strip()]
