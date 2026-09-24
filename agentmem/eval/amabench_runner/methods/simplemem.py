"""SimpleMem method for AMABench.

Uses SimpleMem's paper-style memory system:
1. Sliding-window LLM extraction into atomic memories
2. Hybrid retrieval over semantic, lexical, and symbolic signals

At QA time, this wrapper returns retrieved context so AMABench can keep
using the shared answer-generation prompt across methods.
"""

from __future__ import annotations

import os
import json
import re
import tempfile
import time
from typing import Any, Optional

import numpy as np

from agentmem.eval.amabench_runner.methods.base import BaseMethod

class SimpleMemMemory:
    """Holds a configured SimpleMem system after trajectory ingestion."""

    def __init__(
        self,
        system: Any,
        sample_dir: Optional[str] = None,
        *,
        category: str = "",
        raw_chunks: Optional[list[str]] = None,
    ):
        self.system = system
        self.sample_dir = sample_dir
        self.category = category
        self.raw_chunks = list(raw_chunks or [])

class _OpenAICompatibleEmbeddingModel:
    """Adapter that matches the SimpleMem embedding interface over /v1/embeddings."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "EMPTY",
        batch_size: int = 8,
    ) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self.model_type = "openai_compatible"
        self.batch_size = max(1, batch_size)
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.encode_single("dimension probe"))
        return self._dimension

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def encode(self, texts: list[str] | str, is_query: bool = False) -> np.ndarray:
        del is_query
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            width = self._dimension or 0
            return np.empty((0, width), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [str(text or "").strip() or "(empty)" for text in texts[i : i + self.batch_size]]
            response = self._client.embeddings.create(input=batch, model=self.model_name)
            ordered = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            vectors = np.asarray(ordered, dtype=np.float32)
            vectors = self._normalize(vectors)
            if self._dimension is None and vectors.size:
                self._dimension = int(vectors.shape[1])
            all_embeddings.append(vectors)

        return np.concatenate(all_embeddings, axis=0)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    def encode_query(self, queries: list[str]) -> np.ndarray:
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        return self.encode(documents, is_query=False)

class SimpleMemMethod(BaseMethod):
    """Wraps SimpleMem for per-episode memory build + QA on AMABench.

    Construction is delegated to the SimpleMem runtime. Retrieval returns
    context snippets so the shared AMABench QA call stays paper-aligned.
    """

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: str = "http://localhost:30000/v1",
        llm_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        enable_thinking: bool = False,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        max_reflection_rounds: int = 2,
        enable_parallel_processing: bool = True,
        max_parallel_workers: int = 2,
        enable_parallel_retrieval: bool = True,
        max_retrieval_workers: int = 3,
        window_size: Optional[int] = None,
        db_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        **_kw,
    ):
        default_llm_model = "Qwen/Qwen3-32B"
        default_llm_base_url = "http://localhost:30000/v1"
        default_enable_thinking = False
        default_enable_planning = True
        default_enable_reflection = True
        default_max_reflection_rounds = 2
        default_enable_parallel_processing = True
        default_max_parallel_workers = 2
        default_enable_parallel_retrieval = True
        default_max_retrieval_workers = 3

        if config_path:
            cfg = self._load_config(config_path)
            if llm_model == default_llm_model:
                llm_model = cfg.get("llm_model", llm_model)
            if llm_base_url == default_llm_base_url:
                llm_base_url = cfg.get("llm_base_url", llm_base_url)
            if llm_api_key is None:
                llm_api_key = cfg.get("llm_api_key", llm_api_key)
            if embedding_model is None:
                embedding_model = cfg.get("embedding_model", embedding_model)
            if embedding_base_url is None:
                embedding_base_url = cfg.get("embedding_base_url", embedding_base_url)
            if embedding_api_key is None:
                embedding_api_key = cfg.get("embedding_api_key", embedding_api_key)
            if enable_thinking == default_enable_thinking:
                enable_thinking = cfg.get("enable_thinking", enable_thinking)
            if enable_planning == default_enable_planning:
                enable_planning = cfg.get("enable_planning", enable_planning)
            if enable_reflection == default_enable_reflection:
                enable_reflection = cfg.get("enable_reflection", enable_reflection)
            if max_reflection_rounds == default_max_reflection_rounds:
                max_reflection_rounds = cfg.get("max_reflection_rounds", max_reflection_rounds)
            if enable_parallel_processing == default_enable_parallel_processing:
                enable_parallel_processing = cfg.get(
                    "enable_parallel_processing",
                    enable_parallel_processing,
                )
            if max_parallel_workers == default_max_parallel_workers:
                max_parallel_workers = cfg.get("max_parallel_workers", max_parallel_workers)
            if enable_parallel_retrieval == default_enable_parallel_retrieval:
                enable_parallel_retrieval = cfg.get(
                    "enable_parallel_retrieval",
                    enable_parallel_retrieval,
                )
            if max_retrieval_workers == default_max_retrieval_workers:
                max_retrieval_workers = cfg.get("max_retrieval_workers", max_retrieval_workers)
            if window_size is None:
                window_size = cfg.get("window_size", window_size)
            if db_dir is None:
                db_dir = cfg.get("db_dir", db_dir)

        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.embedding_api_key = embedding_api_key
        self.enable_thinking = enable_thinking
        self.enable_planning = enable_planning
        self.enable_reflection = enable_reflection
        self.max_reflection_rounds = max_reflection_rounds
        self.enable_parallel_processing = enable_parallel_processing
        self.max_parallel_workers = max_parallel_workers
        self.enable_parallel_retrieval = enable_parallel_retrieval
        self.max_retrieval_workers = max_retrieval_workers
        self.window_size = int(window_size) if window_size else None
        if db_dir is None:
            env_dir = os.environ.get("AMABENCH_SIMPLEMEM_DB_DIR")
            if env_dir:
                db_dir = env_dir
        if db_dir is None:
            db_dir = tempfile.mkdtemp(prefix="simplemem_amabench_")
        os.makedirs(db_dir, exist_ok=True)
        self.db_dir = db_dir
        self._episode_counter = 0

    def memory_construction(self, traj_text: str, task: str = "") -> SimpleMemMemory:
        import simplemem.system as simplemem_system

        _patch_simplemem_qwen_no_thinking()
        SimpleMemSystem = simplemem_system.SimpleMemSystem

        self._episode_counter += 1
        db_path = os.path.join(self.db_dir, f"episode_{self._episode_counter}")
        os.makedirs(db_path, exist_ok=True)

        api_emb = None
        old_embedding_ctor = None
        if self.embedding_model and self.embedding_base_url:
            api_emb = _OpenAICompatibleEmbeddingModel(
                model_name=self.embedding_model,
                base_url=self.embedding_base_url,
                api_key=self.embedding_api_key or "EMPTY",
            )
            old_embedding_ctor = simplemem_system.EmbeddingModel

            simplemem_system.EmbeddingModel = lambda: api_emb

        system = SimpleMemSystem(
            api_key=self.llm_api_key,
            model=self.llm_model,
            base_url=self.llm_base_url,
            db_path=db_path,
            table_name=f"ep_{self._episode_counter}",
            clear_db=True,
            enable_thinking=self.enable_thinking,
            use_streaming=False,
            enable_planning=self.enable_planning,
            enable_reflection=self.enable_reflection,
            max_reflection_rounds=self.max_reflection_rounds,
            enable_parallel_processing=self.enable_parallel_processing,
            max_parallel_workers=self.max_parallel_workers,
            enable_parallel_retrieval=self.enable_parallel_retrieval,
            max_retrieval_workers=self.max_retrieval_workers,
        )
        window_size_env = os.environ.get("SIMPLEMEM_WINDOW_SIZE")
        if window_size_env:
            system.memory_builder.window_size = max(1, int(window_size_env))
        elif self.window_size is not None:
            system.memory_builder.window_size = max(1, self.window_size)
        workers_env = os.environ.get("SIMPLEMEM_WORKERS")
        if workers_env:
            system.memory_builder.max_parallel_workers = max(1, int(workers_env))
            system.memory_builder.enable_parallel_processing = True
        retrieval_workers_env = os.environ.get("SIMPLEMEM_RETRIEVAL_WORKERS")
        if retrieval_workers_env and hasattr(system, "hybrid_retriever"):
            system.hybrid_retriever.max_workers = max(1, int(retrieval_workers_env))
        if old_embedding_ctor is not None:
            simplemem_system.EmbeddingModel = old_embedding_ctor

        if api_emb is not None:
            system.embedding_model = api_emb
            system.vector_store.embedding_model = api_emb
            system.memory_builder.vector_store.embedding_model = api_emb
            system.hybrid_retriever.vector_store.embedding_model = api_emb

            system.vector_store.clear()

        turns = _parse_trajectory_turns(traj_text, task)
        from simplemem.models.memory_entry import Dialogue
        dialogues = [
            Dialogue(dialogue_id=i, speaker=sp, content=co)
            for i, (sp, co) in enumerate(turns)
        ]
        n_total = len(dialogues)

        if n_total > 2000:
            if not window_size_env:
                system.memory_builder.window_size = max(20, system.memory_builder.window_size or 4)

            if not workers_env:
                system.memory_builder.max_parallel_workers = max(16, system.memory_builder.max_parallel_workers or 2)
            system.memory_builder.enable_parallel_processing = True
            print(
                f"[simplemem build] large-corpus mode: {n_total} dialogues, "
                f"window_size={system.memory_builder.window_size}, "
                f"workers={system.memory_builder.max_parallel_workers}",
                flush=True,
            )
        _t0 = time.perf_counter()
        if n_total > 2000:
            _build_with_checkpoint(system, dialogues, db_path)
        else:
            system.add_dialogues(dialogues)
        _t_add = time.perf_counter() - _t0
        print(
            f"[simplemem build] add_dialogues({n_total}) returned in {_t_add:.0f}s; "
            f"finalizing now",
            flush=True,
        )
        _t1 = time.perf_counter()
        system.finalize()
        _t_fin = time.perf_counter() - _t1
        print(
            f"[simplemem build] finalize() done in {_t_fin:.0f}s; "
            f"total build wallclock={_t_add+_t_fin:.0f}s",
            flush=True,
        )

        return SimpleMemMemory(
            system,
            sample_dir=db_path,
            category=_category_from_task(task),
            raw_chunks=[content for _speaker, content in turns],
        )

    def memory_retrieve(self, memory: SimpleMemMemory, question: str) -> str:
        """Retrieve context from SimpleMem for the standard QA pipeline.

        Paper routes all methods through the shared QA prompt:
        ``{context}\\n\\n# Question\\n{question}\\n\\n###Answer: ...``
        So we return retrieved context, not a direct answer.
        """
        import logging

        _log = logging.getLogger(__name__)

        try:
            contexts = memory.system.hybrid_retriever.retrieve(question)
        except Exception as exc:
            _log.warning("SimpleMem retrieve() failed: %s", exc)
            contexts = self._semantic_retrieve_fallback(memory, question)

        if contexts:
            context_parts = []
            for i, result in enumerate(contexts[:5]):
                text = self._result_to_text(result)
                if text:
                    context_parts.append(f"Memory {i+1}: {text}")
            if context_parts:
                context = "\n\n".join(context_parts)
                raw_context = self._raw_exact_context(memory, question)
                if raw_context:
                    return raw_context + "\n\n# SimpleMem Retrieved Context\n" + context
                return context
            _log.warning("SimpleMem retrieve() returned results but all were empty")

        raw_context = self._raw_exact_context(memory, question)
        if raw_context:
            return raw_context

        _log.warning("SimpleMem: no context retrieved for question: %.80s", question)
        return ""

    def _semantic_retrieve_fallback(self, memory: SimpleMemMemory, question: str) -> list[Any]:
        """Bypass SimpleMem's planning layer when its LLM plan is malformed.

        Qwen-sized backbones can return a JSON list where the upstream planning
        code expects a dict with ``required_info``. In that case, the vector
        store is still valid; using the semantic layer is preferable to
        producing an empty-context paper cell.
        """

        retriever = getattr(memory.system, "hybrid_retriever", None)
        semantic_search = getattr(retriever, "_semantic_search", None)
        if not callable(semantic_search):
            return []
        try:
            return list(semantic_search(question) or [])
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("SimpleMem semantic fallback failed: %s", exc)
            return []

    @staticmethod
    def _result_to_text(result: Any) -> str:
        if isinstance(result, dict):
            return str(
                result.get("text")
                or result.get("content")
                or result.get("memory")
                or result.get("lossless_restatement")
                or ""
            ).strip()
        for attr in ("lossless_restatement", "memory", "content", "text"):
            value = getattr(result, attr, None)
            if value:
                return str(value).strip()
        return str(result or "").strip()

    @staticmethod
    def _raw_exact_context(memory: SimpleMemMemory, question: str) -> str:
        force = os.environ.get("SIMPLEMEM_INCLUDE_RAW_TOPK")
        is_exact_category = str(memory.category or "").upper() in {"TTL", "CR"}
        if not force and not is_exact_category:
            return ""
        try:
            top_k = int(force or os.environ.get("SIMPLEMEM_EXACT_RAW_TOPK", "8"))
        except ValueError:
            top_k = 8
        if top_k <= 0:
            return ""
        ranked = _rank_raw_chunks(memory.raw_chunks, question)
        parts = []
        for rank, (idx, chunk, score) in enumerate(ranked[:top_k], start=1):
            parts.append(f"[raw_chunk rank={rank} source_turn={idx} score={score:.3f}]\n{chunk}")
        if not parts:
            return ""
        return "# Raw SimpleMem Evidence Preserved For Exact Answering\n" + "\n\n".join(parts)

def _build_with_checkpoint(system: Any, dialogues: list, db_path: str) -> None:
    """Crash-tolerant + efficient parallel window builder for SimpleMem.

    Replaces upstream ``add_dialogues_parallel`` for large corpora with:
    - Per-window pickle checkpoint at ``{db_path}/build_windows/window_NNNNN.pkl``
    - Resume from existing checkpoint on restart (skip done windows)
    - 5 retries per window with exponential backoff
    - Raise loudly if a window can't be built (don't silently fill with [] —
      that was the failure mode of the earlier simplemem=2.5 result)
    - Progress prints every 10 windows with elapsed wall-time

    Why: upstream worker returns ``[]`` after 3 retries, so a transient sglang
    blip while ~16 workers are mid-flight silently drops ~16 windows. Combined
    with no on-disk checkpoint, a process crash redoes all 491 windows. Both
    issues are fatal under shared-endpoint contention.
    """
    import concurrent.futures
    import pickle
    import threading
    from pathlib import Path

    builder = system.memory_builder
    ws = builder.window_size
    workers = builder.max_parallel_workers

    windows: list[list] = []
    i = 0
    while i < len(dialogues):
        chunk = dialogues[i : i + ws]
        if chunk:
            windows.append(chunk)
        i += ws
    n_windows = len(windows)

    ckpt_dir = Path(db_path) / "build_windows"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    done: dict[int, list] = {}
    for p in sorted(ckpt_dir.glob("window_*.pkl")):
        try:
            wn = int(p.stem.split("_")[1])
            with open(p, "rb") as f:
                done[wn] = pickle.load(f)
        except Exception as exc:
            print(f"[simplemem build] skipping corrupt checkpoint {p.name}: {exc}", flush=True)
    if done:
        print(
            f"[simplemem build] resume: {len(done)}/{n_windows} windows already on disk",
            flush=True,
        )

    remaining = [(wn + 1, w) for wn, w in enumerate(windows) if (wn + 1) not in done]
    print(
        f"[simplemem build] dispatching {len(remaining)} new windows, "
        f"{workers} workers, retries=5 with backoff",
        flush=True,
    )

    ckpt_lock = threading.Lock()

    def _safe_worker(window: list, window_num: int) -> list:
        last_exc: Optional[BaseException] = None
        for attempt in range(5):
            try:
                ids = [d.dialogue_id for d in window]
                entries = builder._generate_memory_entries_worker(window, ids, window_num)
                if entries:
                    return entries
                last_exc = RuntimeError(
                    f"worker returned empty entries (LLM parse failed silently)"
                )
            except Exception as exc:                
                last_exc = exc
            sleep_s = min(60, 2 ** attempt)
            time.sleep(sleep_s)
        raise RuntimeError(
            f"window {window_num} build failed after 5 attempts: {last_exc}"
        )

    t_start = time.time()
    n_done = len(done)
    n_failed = 0
    failures: list[tuple[int, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_safe_worker, w, wn): wn for wn, w in remaining}
        for fut in concurrent.futures.as_completed(futures):
            wn = futures[fut]
            try:
                entries = fut.result()
            except Exception as exc:                
                n_failed += 1
                failures.append((wn, str(exc)))
                print(f"[simplemem build] window {wn} FAILED: {exc}", flush=True)
                continue
            ckpt_path = ckpt_dir / f"window_{wn:05d}.pkl"
            tmp_path = ckpt_path.with_suffix(".pkl.tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(entries, f)
            os.replace(tmp_path, ckpt_path)
            with ckpt_lock:
                done[wn] = entries
                n_done += 1
                if n_done % 10 == 0 or n_done == n_windows:
                    elapsed = time.time() - t_start
                    print(
                        f"[simplemem build] {n_done}/{n_windows} windows done "
                        f"({n_failed} failed) [{elapsed:.0f}s elapsed]",
                        flush=True,
                    )

    _max_frac = float(os.environ.get("SIMPLEMEM_MAX_FAILED_FRAC", "0.05"))
    _max_count_env = os.environ.get("SIMPLEMEM_MAX_FAILED_COUNT")
    _max_allowed = (
        int(_max_count_env)
        if _max_count_env
        else max(1, int(n_windows * _max_frac))
    )
    if n_failed > _max_allowed:
        sample = "; ".join(f"#{wn}: {msg[:80]}" for wn, msg in failures[:3])
        raise RuntimeError(
            f"[simplemem build] {n_failed}/{n_windows} windows failed "
            f"(threshold={_max_allowed}). First: {sample}. Fix LLM endpoint and "
            f"rerun — surviving windows in {ckpt_dir} will be reused."
        )
    if n_failed:
        print(
            f"[simplemem build] WARNING: {n_failed}/{n_windows} windows failed "
            f"after 5 retries; continuing with {n_done} good windows "
            f"(threshold={_max_allowed}). Failed window numbers: "
            f"{sorted(wn for wn, _ in failures)[:10]}",
            flush=True,
        )

    all_entries: list = []
    for wn in sorted(done):
        all_entries.extend(done[wn])
    print(
        f"[simplemem build] storing {len(all_entries)} entries to vector store",
        flush=True,
    )
    builder.vector_store.add_entries(all_entries)
    builder.processed_count = sum(len(w) for w in windows)
    if all_entries:
        builder.previous_entries = all_entries[-10:]

def _parse_trajectory_turns(traj_text: str, task: str = ""):
    """Parse trajectory text into (speaker, content) pairs for SimpleMem."""
    turns: list[tuple[str, str]] = []

    if task:
        turns.append(("system", f"Task: {task}"))

    current_action = ""
    current_obs = ""
    saw_structured = False

    for line in traj_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Action:"):
            saw_structured = True
            if current_obs:
                turns.append(("environment", current_obs))
                current_obs = ""
            current_action = stripped[len("Action:"):].strip()
            turns.append(("agent", current_action))
        elif stripped.startswith("Observation:"):
            saw_structured = True
            current_obs = stripped[len("Observation:"):].strip()
        elif stripped.startswith(("Step ", "Turn ")):
            saw_structured = True
            if current_obs:
                turns.append(("environment", current_obs))
                current_obs = ""
        else:
            if current_obs:
                current_obs += " " + stripped

    if current_obs:
        turns.append(("environment", current_obs))

    if not saw_structured:
        for chunk in _raw_memory_chunks(traj_text):
            turns.append(("environment", chunk))

    if not turns:
        turns.append(("agent", traj_text))

    return turns

def _raw_memory_chunks(text: str, max_chars: int = 4000) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in str(text or "").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        block_len = len(block)
        if current and current_len + block_len + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        if block_len > max_chars:
            for start in range(0, block_len, max_chars):
                part = block[start:start + max_chars].strip()
                if part:
                    chunks.append(part)
            continue
        current.append(block)
        current_len += block_len + 2
    if current:
        chunks.append("\n\n".join(current))
    if not chunks and str(text or "").strip():
        chunks.append(str(text).strip()[:max_chars])
    return chunks

def _category_from_task(task: str) -> str:
    for line in str(task or "").splitlines():
        if line.lower().startswith("category:"):
            return line.split(":", 1)[1].strip().upper()
    return ""

def _rank_raw_chunks(chunks: list[str], question: str) -> list[tuple[int, str, float]]:
    terms = [term for term in re.findall(r"[A-Za-z0-9_]+", question.lower()) if len(term) > 2]
    if not terms:
        return [(idx, chunk, 0.0) for idx, chunk in enumerate(chunks)]
    ranked = []
    for idx, chunk in enumerate(chunks):
        lower = chunk.lower()
        score = 0.0
        for term in terms:
            if term in lower:
                score += 1.0 + min(5, lower.count(term)) / 5.0
        ranked.append((idx, chunk, score))
    return sorted(ranked, key=lambda row: (row[2], -row[0]), reverse=True)

def _patch_simplemem_qwen_no_thinking() -> None:
    """Force Qwen3/vLLM SimpleMem extraction calls out of thinking mode.

    The upstream SimpleMem client only sends ``enable_thinking`` for DashScope
    URLs. Our experiments use a local vLLM OpenAI-compatible endpoint, where
    Qwen3 disables thinking through ``chat_template_kwargs``. Without this,
    extraction responses can start with ``<think>`` and fail JSON parsing.
    """

    try:
        import simplemem.utils.llm_client as llm_client
        import simplemem.core.memory_builder as memory_builder
    except Exception:
        return

    LLMClient = llm_client.LLMClient
    MemoryBuilder = memory_builder.MemoryBuilder
    if getattr(LLMClient, "_agentmem_qwen_no_think_patch", False) and getattr(
        MemoryBuilder, "_agentmem_qwen_parse_patch", False
    ):
        return

    def chat_completion(
        self,
        messages,
        temperature: float = 0.2,
        response_format: Optional[dict[str, str]] = None,
        max_retries: int = 3,
    ) -> str:
        model_name = str(self.model or "").lower()
        base_url = str(self.base_url or "").lower()
        is_local_qwen = "qwen" in model_name or "localhost" in base_url or "127.0.0.1" in base_url
        request_messages = messages
        if is_local_qwen:
            request_messages = [dict(msg) for msg in messages]
            no_think_prefix = (
                "/no_think\n"
                "Return only the requested JSON array. Do not include analysis, "
                "markdown, commentary, or any text before or after the JSON.\n\n"
            )
            for msg in request_messages:
                if msg.get("role") == "user":
                    msg["content"] = no_think_prefix + str(msg.get("content", ""))
                    break

        default_max_tokens = "4096" if is_local_qwen else "1024"
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "temperature": 0.0 if is_local_qwen else temperature,
            "max_tokens": int(os.environ.get("SIMPLEMEM_MAX_TOKENS", default_max_tokens)),
        }
        if response_format:
            kwargs["response_format"] = response_format

        if is_local_qwen:
            extra_body: dict[str, Any] = {
                "chat_template_kwargs": {"enable_thinking": False},
                "enable_thinking": False,
            }
            if os.environ.get("SIMPLEMEM_USE_GUIDED_JSON", "1") != "0":
                extra_body["guided_json"] = _SIMPLEMEM_MEMORY_ARRAY_SCHEMA
            kwargs["extra_body"] = extra_body

        last_exception: Exception | None = None
        for attempt in range(max_retries):
            try:
                call_kwargs = dict(kwargs)
                if getattr(self, "use_streaming", False):
                    call_kwargs["stream"] = True
                    text = self._handle_streaming_response(**call_kwargs)
                else:
                    response = self.client.chat.completions.create(**call_kwargs)
                    text = response.choices[0].message.content or ""
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                text = _coerce_simplemem_json_array(text)
                if is_local_qwen and not _is_simplemem_json_array(text):
                    text = _repair_simplemem_json_response(self, request_messages, text, kwargs)
                return text
            except Exception as exc:
                if (
                    is_local_qwen
                    and "extra_body" in kwargs
                    and "guided_json" in kwargs["extra_body"]
                    and _looks_like_unsupported_guided_json(exc)
                ):
                    kwargs = dict(kwargs)
                    kwargs["extra_body"] = dict(kwargs["extra_body"])
                    kwargs["extra_body"].pop("guided_json", None)
                    try:
                        if getattr(self, "use_streaming", False):
                            fallback_kwargs = dict(kwargs)
                            fallback_kwargs["stream"] = True
                            text = self._handle_streaming_response(**fallback_kwargs)
                        else:
                            response = self.client.chat.completions.create(**kwargs)
                            text = response.choices[0].message.content or ""
                        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                        text = _coerce_simplemem_json_array(text)
                        if is_local_qwen and not _is_simplemem_json_array(text):
                            text = _repair_simplemem_json_response(self, request_messages, text, kwargs)
                        return text
                    except Exception as fallback_exc:
                        exc = fallback_exc
                last_exception = exc
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {exc}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"LLM API call failed after {max_retries} attempts: {exc}")
        if last_exception:
            raise last_exception
        raise RuntimeError("LLM API call failed with no exception")

    LLMClient.chat_completion = chat_completion
    LLMClient._agentmem_qwen_no_think_patch = True

    MemoryEntry = memory_builder.MemoryEntry

    def _parse_llm_response(self, response: str, dialogue_ids: list[int]) -> list[Any]:
        del dialogue_ids
        data = _normalize_simplemem_extraction_data(self.llm_client.extract_json(response))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array but got: {type(data)}")

        entries = []
        for item in data:
            if isinstance(item, str):
                restatement = item.strip()
                if restatement:
                    entries.append(
                        MemoryEntry(
                            lossless_restatement=restatement,
                            keywords=[],
                            timestamp=None,
                            location=None,
                            persons=[],
                            entities=[],
                            topic=None,
                        )
                    )
                continue
            if not isinstance(item, dict):
                continue
            restatement = (
                item.get("lossless_restatement")
                or item.get("memory")
                or item.get("content")
                or item.get("text")
                or item.get("fact")
                or item.get("summary")
            )
            if not restatement:
                continue
            entries.append(
                MemoryEntry(
                    lossless_restatement=str(restatement),
                    keywords=item.get("keywords", []),
                    timestamp=item.get("timestamp"),
                    location=item.get("location"),
                    persons=item.get("persons", []),
                    entities=item.get("entities", []),
                    topic=item.get("topic"),
                )
            )
        return entries

    MemoryBuilder._parse_llm_response = _parse_llm_response
    MemoryBuilder._agentmem_qwen_parse_patch = True

_SIMPLEMEM_MEMORY_ARRAY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "lossless_restatement": {"type": "string"},
            "memory": {"type": "string"},
            "content": {"type": "string"},
            "text": {"type": "string"},
            "fact": {"type": "string"},
            "summary": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "timestamp": {"type": ["string", "null"]},
            "location": {"type": ["string", "null"]},
            "persons": {"type": "array", "items": {"type": "string"}},
            "entities": {"type": "array", "items": {"type": "string"}},
            "topic": {"type": ["string", "null"]},
        },
        "additionalProperties": True,
    },
}

def _looks_like_unsupported_guided_json(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "guided_json" in msg or "guided json" in msg or "extra_body" in msg

def _coerce_simplemem_json_array(text: str) -> str:
    """Normalize Qwen JSON-object extraction responses for SimpleMem.

    Upstream SimpleMem's builder expects the LLM response to parse to a JSON
    array. Qwen often wraps that array in an object such as
    ``{"memories": [...]}`` when JSON mode is enabled.
    """
    text = _strip_json_fence(str(text or "").strip())
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = _extract_embedded_json(text)
        if parsed is None:
            return text
    normalized = _normalize_simplemem_extraction_data(parsed)
    if isinstance(normalized, list):
        return json.dumps(normalized, ensure_ascii=False)
    return text

def _is_simplemem_json_array(text: str) -> bool:
    try:
        parsed = json.loads(_strip_json_fence(str(text or "").strip()))
    except Exception:
        parsed = _extract_embedded_json(str(text or ""))
    return isinstance(_normalize_simplemem_extraction_data(parsed), list)

def _repair_simplemem_json_response(
    client_obj: Any,
    original_messages: list[dict[str, Any]],
    bad_text: str,
    base_kwargs: dict[str, Any],
) -> str:
    """Make one strict repair call when Qwen emits prose before JSON.

    Qwen3-Thinking can ignore /no_think on long extraction prompts and spend
    the entire token budget on reasoning. The repair call repeats the original
    extraction prompt, shows the invalid response prefix, and forces a JSON
    object wrapper that our normalizer can convert back into the array expected
    by upstream SimpleMem.
    """

    prompt_text = "\n\n".join(str(msg.get("content", "")) for msg in original_messages)
    repair_prompt = (
        "/no_think\n"
        "The previous response was invalid because it was not a JSON array of "
        "memory entries. Re-run the extraction from the original prompt below.\n"
        "Return exactly one JSON object with this shape and no other text:\n"
        "{\"memories\":[{\"lossless_restatement\":\"...\",\"keywords\":[],"
        "\"timestamp\":null,\"location\":null,\"persons\":[],\"entities\":[],"
        "\"topic\":null}]}\n\n"
        "Invalid response prefix:\n"
        f"{str(bad_text or '')[:800]}\n\n"
        "Original extraction prompt:\n"
        f"{prompt_text}"
    )
    repair_kwargs = dict(base_kwargs)
    repair_kwargs["messages"] = [{"role": "user", "content": repair_prompt}]
    repair_kwargs["temperature"] = 0.0
    repair_kwargs["max_tokens"] = int(os.environ.get("SIMPLEMEM_REPAIR_MAX_TOKENS", "4096"))
    repair_kwargs["response_format"] = {"type": "json_object"}
    extra_body = dict(repair_kwargs.get("extra_body") or {})
    extra_body.pop("guided_json", None)
    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    extra_body["enable_thinking"] = False
    repair_kwargs["extra_body"] = extra_body
    response = client_obj.client.chat.completions.create(**repair_kwargs)
    text = response.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return _coerce_simplemem_json_array(text)

def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    return stripped

def _extract_embedded_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            parsed, _end = decoder.raw_decode(text[match.start():])
        except Exception:
            continue
        normalized = _normalize_simplemem_extraction_data(parsed)
        if isinstance(normalized, list):
            return parsed
    return None

def _normalize_simplemem_extraction_data(parsed: Any) -> Any:
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in (
            "memories",
            "memory_entries",
            "entries",
            "items",
            "data",
            "results",
            "facts",
        ):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        if any(
            key in parsed
            for key in (
                "lossless_restatement",
                "memory",
                "content",
                "text",
                "fact",
                "summary",
            )
        ):
            return [parsed]
    return parsed
