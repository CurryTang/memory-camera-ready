"""Corpus-level (HippoRAG-2 regime) adapters for HotpotQA.

All three classes load the same shared corpus (``datasets/hotpotqa/hotpotqa_corpus.json``,
9,811 paragraphs / ~1.4 M tokens) ONCE at init, build their method-specific index
ONCE, and serve every question from that pre-built memory. The build phase is
explicitly timed and recorded into ``EfficiencyCounters``.

Methods covered here:

- ``HotpotQAHippoRAGMethod`` — wraps the **official** HippoRAG-2 invocation from
  ``resource/From_RAG_to_Memory_.../code/HippoRAG/main.py``: same ``BaseConfig``
  parameters (retrieval_top_k=200, linking_top_k=5, qa_top_k=5, max_qa_steps=3,
  graph_type=facts_and_sim_passage_node_unidirectional). 100 % reproducible.
- ``HotpotQASimpleMemMethod`` — SimpleMem (density gating + atomic units +
  query-aware retrieval) over the shared corpus. Mirrors the AMABench
  ``SimpleMemMethod`` interface but feeds the shared 9,811-paragraph pool.
The longcontext + memrl agentic baselines live separately in
``agentmem/eval/hotpotqa_agentic.py`` (no pre-built index; agent browses the
corpus via grep/read tools).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from agentmem.eval.hotpotqa_agentic import _default_corpus_path
from agentmem.methods.base import EfficiencyCounters

@contextlib.contextmanager
def _capture_openai_build_usage(counters: EfficiencyCounters):
    """Monkey-patch openai SDK so chat + embedding usage during the build phase
    accumulates into ``counters.build_input_tokens`` / ``build_output_tokens``.

    Both HippoRAG-v2's chat client (``hipporag.llm.openai_gpt.CacheOpenAI``) and
    its embedding client (``hipporag.embedding_model.OpenAI.OpenAIEmbeddingModel``)
    talk to the OpenAI SDK via ``chat.completions.create`` / ``embeddings.create``,
    so patching the SDK class methods captures every call hipporag makes during
    indexing without touching vendor code. Restores originals on exit.

    Embedding usage is added as input tokens only (embeddings have no completion).
    """
    try:
        from openai.resources.chat.completions import Completions as _ChatCompletions
        from openai.resources.embeddings import Embeddings as _Embeddings
    except Exception:                                                
        yield
        return

    _orig_chat = _ChatCompletions.create
    _orig_embed = _Embeddings.create

    def _wrapped_chat(self, *a, **kw):
        resp = _orig_chat(self, *a, **kw)
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                p = getattr(u, "prompt_tokens", 0) or 0
                c = getattr(u, "completion_tokens", 0) or 0
                counters.build_input_tokens += int(p)
                counters.build_output_tokens += int(c)
                counters.total_input_tokens += int(p)
                counters.total_output_tokens += int(c)
        except Exception:
            pass
        return resp

    def _wrapped_embed(self, *a, **kw):
        resp = _orig_embed(self, *a, **kw)
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                p = getattr(u, "prompt_tokens", 0) or 0
                counters.build_input_tokens += int(p)
                counters.total_input_tokens += int(p)
        except Exception:
            pass
        return resp

    _ChatCompletions.create = _wrapped_chat
    _Embeddings.create = _wrapped_embed
    try:
        yield
    finally:
        _ChatCompletions.create = _orig_chat
        _Embeddings.create = _orig_embed

class HotpotQAAdapterError(RuntimeError):
    """Base class for missing-implementation errors in HotpotQA adapters."""

class MemRLNotConfiguredError(HotpotQAAdapterError):
    """Raised when HotpotQAMemRLMethod is missing the mos_config_path."""

def _load_corpus_docs(corpus_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return the list of {'idx', 'title', 'text'} dicts."""
    path = Path(corpus_path) if corpus_path else _default_corpus_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else [raw]
    return [d for d in items if isinstance(d, dict) and d.get("title") and d.get("text")]

def _docs_as_strings(docs: list[dict[str, Any]]) -> list[str]:
    """Match HippoRAG-2's main.py: ``f\"{title}\\n{text}\"`` per passage."""
    return [f"{d['title']}\n{d['text']}" for d in docs]

def _hipporag_rerank_dspy_file_path(llm_model: str) -> str | None:
    """Return an existing HippoRAG rerank prompt path, or None for built-in default."""
    override = os.environ.get("HIPPORAG_RERANK_DSPY_FILE_PATH")
    if override:
        path = Path(override).expanduser()
        if path.exists():
            return str(path)
        print(f"HotpotQA HippoRAG: ignoring missing HIPPORAG_RERANK_DSPY_FILE_PATH={override}")
    elif override == "":
        return None

    def _suffixes(model: str) -> list[str]:
        raw = model.rsplit("/", 1)[-1]
        normalized = raw.replace("_", "-")
        lower = normalized.lower()
        suffixes = [normalized, lower]

        suffixes.append("llama3.3-70B-Instruct")
        seen: set[str] = set()
        return [s for s in suffixes if not (s in seen or seen.add(s))]

    prompt_dirs: list[Path] = []
    try:
        import hipporag

        prompt_dirs.append(Path(hipporag.__file__).resolve().parent / "prompts" / "dspy_prompts")
    except Exception:
        pass
    repo_root = Path(__file__).resolve().parents[2]
    prompt_dirs.append(
        repo_root
        / "resource"
        / "From_RAG_to_Memory_Non_Parametric_Continual_Learning_for_Large_Language_Models"
        / "code"
        / "HippoRAG"
        / "src"
        / "hipporag"
        / "prompts"
        / "dspy_prompts"
    )
    prompt_dirs.append(Path.cwd() / "src" / "hipporag" / "prompts" / "dspy_prompts")

    for prompt_dir in prompt_dirs:
        for suffix in _suffixes(llm_model):
            candidate = prompt_dir / f"filter_{suffix}.json"
            if candidate.exists():
                return str(candidate)
    return None

class HotpotQAHippoRAGMethod:
    """Official HippoRAG-2 HotpotQA invocation (100 % reproducible).

    Mirrors ``resource/From_RAG_to_Memory_.../code/HippoRAG/main.py`` exactly:
    same ``BaseConfig`` parameters, same ``hipporag.index(docs)`` build, same
    ``hipporag.rag_qa`` retrieval. Build time and tokens are recorded.
    """

    name = "hipporag"

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: str | None = None,
        llm_api_key: str = "EMPTY",
        embedding_model: str = "nvidia/NV-Embed-v2",
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        save_dir: str | None = None,
        corpus_path: str | None = None,
        retrieval_top_k: int = 200,
        linking_top_k: int = 5,
        qa_top_k: int = 5,
        max_qa_steps: int = 3,
        graph_type: str = "facts_and_sim_passage_node_unidirectional",
        embedding_batch_size: int = 8,

        force_index_from_scratch: bool = False,
        force_openie_from_scratch: bool = False,
        openie_mode: str = "online",
        token_counter: Any = None,
        **_kw: Any,
    ) -> None:
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self._token_counter = token_counter
        self._counters = EfficiencyCounters()

        qa_top_k = int(os.environ.get("HOTPOTQA_HIPPORAG_QA_TOP_K", qa_top_k))
        linking_top_k = int(os.environ.get("HOTPOTQA_HIPPORAG_LINKING_TOP_K", linking_top_k))

        from hipporag.HippoRAG import HippoRAG
        from hipporag.utils.config_utils import BaseConfig

        if save_dir is None:
            save_dir = "outputs/hipporag_hotpotqa"

        if llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
        if embedding_api_key:
            os.environ["OPENAI_API_KEY"] = embedding_api_key
        if embedding_base_url:
            os.environ["OPENAI_BASE_URL"] = embedding_base_url
        elif llm_base_url:

            os.environ.setdefault("OPENAI_BASE_URL", llm_base_url)

        rerank_dspy_file_path = _hipporag_rerank_dspy_file_path(llm_model)
        if rerank_dspy_file_path:
            print(f"HotpotQA HippoRAG: rerank DSPy prompt={rerank_dspy_file_path}")
        else:
            print("HotpotQA HippoRAG: rerank DSPy prompt=HippoRAG built-in default")

        config = BaseConfig(
            save_dir=save_dir,
            llm_base_url=llm_base_url or "https://api.openai.com/v1",
            llm_name=llm_model,
            dataset="hotpotqa",
            embedding_model_name=embedding_model,
            embedding_base_url=embedding_base_url or llm_base_url or "https://api.openai.com/v1",
            force_index_from_scratch=force_index_from_scratch,
            force_openie_from_scratch=force_openie_from_scratch,
            rerank_dspy_file_path=rerank_dspy_file_path,
            retrieval_top_k=retrieval_top_k,
            linking_top_k=linking_top_k,
            max_qa_steps=max_qa_steps,
            qa_top_k=qa_top_k,
            graph_type=graph_type,
            embedding_batch_size=embedding_batch_size,
            max_new_tokens=None,
            corpus_len=0,                
            openie_mode=openie_mode,
        )

        self._docs_raw = _load_corpus_docs(corpus_path)
        docs = _docs_as_strings(self._docs_raw)
        config.corpus_len = len(docs)

        try:
            from hipporag.embedding_model import _get_embedding_model_class as _orig_get
            from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
            _embed_url = embedding_base_url or llm_base_url
            _embed_key = embedding_api_key or llm_api_key or "EMPTY"
            if _embed_url:
                def _patched(embedding_model_name: str = "Qwen3-Embedding-4B"):
                    cls = _orig_get(embedding_model_name)
                    if cls is OpenAIEmbeddingModel:

                        original_init = OpenAIEmbeddingModel.__init__
                        def _patched_init(self, *a, **kw):
                            kw.setdefault("base_url", _embed_url)
                            kw.setdefault("api_key", _embed_key)
                            return original_init(self, *a, **kw)
                        OpenAIEmbeddingModel.__init__ = _patched_init
                    return cls
                import hipporag.embedding_model as _em
                _em._get_embedding_model_class = _patched
        except Exception:                                
            pass

        t0 = time.perf_counter()
        with _capture_openai_build_usage(self._counters):
            self._hipporag = HippoRAG(global_config=config)
            self._hipporag.index(docs)
        self._build_seconds = time.perf_counter() - t0
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_llm_build", self._build_seconds)
        self._qa_top_k = qa_top_k

    @property
    def counters(self) -> EfficiencyCounters:
        return self._counters

    @property
    def build_seconds(self) -> float:
        return self._build_seconds

    def reset_counters(self) -> EfficiencyCounters:
        self._counters = EfficiencyCounters()
        return self._counters

    def build_from_sample(self, sample: dict[str, Any], *, sample_id: str) -> Any:

        return {"sample_id": sample_id}

    def build(self, traj_text: str, *, task: str = "") -> Any:
        return {"sample_id": task or "anonymous"}

    def answer(self, memory: Any, question: str) -> str:
        """Run hipporag.retrieve and return the concatenated retrieved passages.

        The outer ``run_hotpotqa50_method`` wraps this string in
        ``build_hotpot_answer_prompt(question, retrieved_context)`` and calls
        the answer LLM. We intentionally skip ``hipporag.rag_qa`` (which has its
        own answer prompt) so all corpus-level methods share the same final
        answer prompt — apples-to-apples.
        """
        t_tool = time.perf_counter()
        results = self._hipporag.retrieve(queries=[question], num_to_retrieve=self._qa_top_k)
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_tool", time.perf_counter() - t_tool)

        try:
            r = results[0]
            docs = list(getattr(r, "docs", None) or [])
        except Exception:
            docs = []
        return "\n\n".join(str(d) for d in docs[: self._qa_top_k])

class HotpotQASimpleMemMethod:
    """SimpleMem (density gating + atomic units) over the shared HotpotQA corpus.

    Build phase: ingest all 9,811 paragraphs, run density gating to drop low-
    information sentences, build atomic-unit dense index. Per question:
    query-aware retrieve top-k.
    """

    name = "simplemem"

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: str | None = None,
        llm_api_key: str = "EMPTY",
        embedding_model: str = "Qwen3-Embedding-4B",
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        top_k: int = 5,
        corpus_path: str | None = None,
        config_path: str | None = None,
        token_counter: Any = None,
        **_kw: Any,
    ) -> None:
        from agentmem.eval.amabench_runner.methods.simplemem import SimpleMemMethod

        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.top_k = top_k
        self._counters = EfficiencyCounters()

        self.inner = SimpleMemMethod(
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            top_k=top_k,
            config_path=config_path,
        )

        self._docs_raw = _load_corpus_docs(corpus_path)
        turn_lines: list[str] = []
        for idx, doc in enumerate(self._docs_raw):
            title = str(doc.get("title", "")).strip()
            text = str(doc.get("text", "")).strip()
            if not title or not text:
                continue
            flat_text = " ".join(text.split())
            turn_lines.append(f"Turn {idx}:")
            turn_lines.append(f"  Action: read_document")
            turn_lines.append(f"  Observation: Title: {title} | Text: {flat_text}")
        traj_text = "\n".join(turn_lines)
        t0 = time.perf_counter()
        self._memory = self.inner.memory_construction(traj_text=traj_text, task="hotpotqa_corpus")
        self._build_seconds = time.perf_counter() - t0
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_llm_build", self._build_seconds)

    @property
    def counters(self) -> EfficiencyCounters:
        return self._counters

    @property
    def build_seconds(self) -> float:
        return self._build_seconds

    def reset_counters(self) -> EfficiencyCounters:
        self._counters = EfficiencyCounters()
        return self._counters

    def build_from_sample(self, sample: dict[str, Any], *, sample_id: str) -> Any:
        return self._memory

    def build(self, traj_text: str, *, task: str = "") -> Any:
        return self._memory

    def answer(self, memory: Any, question: str) -> str:
        t_tool = time.perf_counter()
        retrieved = self.inner.memory_retrieve(memory, question)
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_tool", time.perf_counter() - t_tool)
        return str(retrieved or "")

class HotpotQALightMemMethod:
    """LightMem over the shared HotpotQA corpus, document-mode build.

    Bug fixed (Codex review): the AMABench LightMem wrapper's
    ``procedural_turns_from_text`` only consumes ``Action:``/``Observation:``/
    ``Turn``/``Step`` lines. HotpotQA paragraphs match none of those, so the
    procedural path silently dropped the entire corpus. We bypass the procedural
    wrapper and call ``document_messages_from_text`` + ``build_from_messages``
    directly, mirroring the rest of the inner ``LightMemMethod.memory_construction``
    pipeline.
    """

    name = "lightmem"

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: str | None = None,
        llm_api_key: str = "EMPTY",
        embedding_model: str = "Qwen3-Embedding-4B",
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        retrieve_limit: int = 5,
        corpus_path: str | None = None,
        config_path: str | None = None,
        chunk_chars: int = 2800,
        overlap_chars: int = 250,
        max_turns: int | None = None,
        token_counter: Any = None,
        **kwargs: Any,
    ) -> None:
        retrieve_limit = int(os.environ.get("HOTPOTQA_LIGHTMEM_RETRIEVE_LIMIT", retrieve_limit))

        from agentmem.eval.amabench_runner.methods.lightmem import (
            LightMemMethod,
            _EpisodeLightMem,
        )
        from agentmem.methods.lightmem import LightMemMethod as UnifiedLightMemMethod
        from agentmem.methods.lightmem_support import (
            PROCEDURAL_LIGHTMEM_PROMPTS,
            document_messages_from_text,
            ensure_embedding_dims,
            make_lightmem_config,
            sanitize_collection_name,
        )

        self._counters = EfficiencyCounters()

        kwargs.setdefault("max_turns", max_turns)
        self.inner = LightMemMethod(
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            retrieve_limit=retrieve_limit,
            config_path=config_path,
            **kwargs,
        )

        self._docs_raw = _load_corpus_docs(corpus_path)
        traj_text = "\n\n".join(_docs_as_strings(self._docs_raw))
        messages = document_messages_from_text(
            traj_text,
            domain="hotpotqa",
            sub_domain="corpus",
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
        )

        collection_name = sanitize_collection_name(
            f"{self.inner.collection_prefix}_corpus_v1"
        )
        dims = ensure_embedding_dims(
            model=self.inner.embedding_model,
            base_url=self.inner.embedding_base_url,
            api_key=self.inner.embedding_api_key or "EMPTY",
            configured_dims=int(self.inner.embedding_dims) if self.inner.embedding_dims else None,
        )
        cfg = make_lightmem_config(
            collection_name=collection_name,
            root_dir=self.inner.storage_root,
            llm_model=self.inner.llm_model,
            llm_base_url=self.inner.llm_base_url,
            llm_api_key=self.inner.llm_api_key,
            embedding_model=self.inner.embedding_model,
            embedding_base_url=self.inner.embedding_base_url,
            embedding_api_key=self.inner.embedding_api_key or "EMPTY",
            embedding_dims=dims,
            pre_compress=self.inner.pre_compress,
            topic_segment=self.inner.topic_segment,
            precomp_topic_shared=self.inner.precomp_topic_shared,
            messages_use=self.inner.messages_use,
            metadata_generate=self.inner.metadata_generate,
            text_summary=self.inner.text_summary,
            extract_threshold=self.inner.extract_threshold,
            extraction_mode=self.inner.extraction_mode,
            llm_max_tokens=self.inner.llm_max_tokens,
            topic_segmenter=self.inner.topic_segmenter_config,
            pre_compressor=self.inner.pre_compressor_config,
        )
        method = UnifiedLightMemMethod(
            config=cfg,
            retrieve_limit=self.inner.retrieve_limit,
            metadata_generate_prompt=PROCEDURAL_LIGHTMEM_PROMPTS,
            direct_store_when_unsegmented=True,
        )
        t0 = time.perf_counter()
        memory = method.build_from_messages(messages, task="hotpotqa_corpus")
        if self.inner.construct_update_queue:
            memory.construct_update_queue_all_entries(
                top_k=self.inner.update_queue_top_k,
                keep_top_n=self.inner.update_queue_keep_top_n,
            )
        if self.inner.offline_update:
            memory.offline_update_all_entries(score_threshold=self.inner.update_score_threshold)
        self._build_seconds = time.perf_counter() - t0
        self._counters.add_seconds("w_llm_build", self._build_seconds)
        self._memory = _EpisodeLightMem(method=method, memory=memory)

    @property
    def counters(self) -> EfficiencyCounters:
        return self._counters

    @property
    def build_seconds(self) -> float:
        return self._build_seconds

    def reset_counters(self) -> EfficiencyCounters:
        self._counters = EfficiencyCounters()
        return self._counters

    def build_from_sample(self, sample: dict[str, Any], *, sample_id: str) -> Any:
        return self._memory

    def build(self, traj_text: str, *, task: str = "") -> Any:
        return self._memory

    def answer(self, memory: Any, question: str) -> str:
        t_tool = time.perf_counter()
        retrieved = self.inner.memory_retrieve(memory, question)
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_tool", time.perf_counter() - t_tool)
        return str(retrieved or "")

class HotpotQAMemRLMethod:
    """MemRL HotpotQA adapter — lightweight in-process variant.

    Build phase: ingest all 9,811 paragraphs as MemRL experiences into an
    EpisodicMemoryStore (lexical BM25-style search, no LLM/embedding calls).
    Per question: two-phase retrieve (recall + value-aware ranking) returns
    top-K paragraphs as context for the runner's answer LLM.

    Cold-start defaults (q_init_pos=1.0); no PPO checkpoint needed. The
    runtime utility update loop is wired but inactive at eval time since we
    have no per-question reward signal.
    """

    name = "memrl"

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",                                             
        llm_base_url: str | None = None,
        llm_api_key: str = "EMPTY",
        embedding_model: str | None = None,             
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        retrieve_k: int = 5,
        phase1_topk: int = 20,
        corpus_path: str | None = None,
        config_path: str | None = None,
        token_counter: Any = None,
        **_kw: Any,
    ) -> None:
        from agentmem.backends.episodic import EpisodicMemoryStore
        from agentmem.memrl import MemRLConfig, MemRLRuntimeEngine

        self._counters = EfficiencyCounters()
        self._token_counter = token_counter
        self.retrieve_k = int(retrieve_k)
        self.phase1_topk = int(phase1_topk)

        self._docs_raw = _load_corpus_docs(corpus_path)
        store = EpisodicMemoryStore()
        runtime = MemRLRuntimeEngine(
            store=store,
            config=MemRLConfig(
                topk=self.retrieve_k,
                phase1_topk=self.phase1_topk,
            ),
        )

        t0 = time.perf_counter()
        for idx, doc in enumerate(self._docs_raw):
            title = str(doc.get("title", "")).strip()
            text = str(doc.get("text", "")).strip()
            experience = f"[{title}] {text}"
            runtime.add_experience(
                intent=title or f"doc_{idx}",
                experience=experience,
                success=True,                             
                metadata={
                    "source": "hotpotqa_corpus",
                    "doc_idx": idx,
                    "title": title,
                },
                task_id=f"doc_{idx}",
            )
        self._build_seconds = time.perf_counter() - t0
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_tool", self._build_seconds)
        self._counters.build_input_tokens += self._count_tokens(
            "\n\n".join(_docs_as_strings(self._docs_raw))
        )
        self._counters.total_input_tokens += self._counters.build_input_tokens
        self._runtime = runtime
        self._shared_memory = {"runtime": runtime}
        print(f"[memrl build] DONE in {self._build_seconds:.1f}s — {len(self._docs_raw)} docs ingested", flush=True)

    @property
    def counters(self) -> EfficiencyCounters:
        return self._counters

    @property
    def build_seconds(self) -> float:
        return self._build_seconds

    def reset_counters(self) -> EfficiencyCounters:
        self._counters = EfficiencyCounters()
        return self._counters

    def build_from_sample(self, sample: dict[str, Any], *, sample_id: str) -> Any:
        return self._shared_memory

    def build(self, traj_text: str, *, task: str = "") -> Any:
        return self._shared_memory

    def answer(self, memory: Any, question: str) -> str:
        runtime = (memory or self._shared_memory)["runtime"]
        t_tool = time.perf_counter()
        result = runtime.retrieve(
            question, phase1_topk=self.phase1_topk, topk=self.retrieve_k
        )
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_tool", time.perf_counter() - t_tool)

        chosen = result.selected or result.candidates[: self.retrieve_k]
        context_parts = []
        for cand in chosen:
            meta = (cand.metadata or {}) if hasattr(cand, "metadata") else {}
            content = str(meta.get("experience") or meta.get("full_content") or cand.content or "")
            if content:
                context_parts.append(content)
        context = "\n\n".join(context_parts)
        self._counters.record_retrieval(
            candidates_scored=len(result.candidates),
            evidence_injected=len(chosen),
            context_tokens=self._count_tokens(context),
        )
        return context

    def _count_tokens(self, text: Any) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0
        return len(str(text or "").split())

class _HotpotQAMemRLServiceMethod:
    """MemRL HotpotQA adapter — frozen paragraph-level corpus memory."""

    name = "memrl_service"

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: str | None = None,
        llm_api_key: str = "EMPTY",
        embedding_model: str = "text-embedding-3-large",
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        retrieve_k: int = 5,
        corpus_path: str | None = None,
        mos_config_path: str | None = None,
        warmed_store_path: str | None = None,
        config_path: str | None = None,
        token_counter: Any = None,
        **service_kwargs: Any,
    ) -> None:
        self._counters = EfficiencyCounters()
        self._token_counter = token_counter
        self.retrieve_k = int(retrieve_k)

        cfg_path = Path(
            mos_config_path
            or config_path
            or (Path(__file__).resolve().parents[2] / "configs" / "memrl" / "hotpotqa_qwen3_32b.yaml")
        )
        if not cfg_path.exists():
            raise MemRLNotConfiguredError(
                f"MemRL requires mos_config_path; missing config file: {cfg_path}"
            )
        self._config = self._load_memrl_yaml(cfg_path)
        self._docs_raw = _load_corpus_docs(corpus_path)
        self._service = self._init_memrl_service(
            cfg_path=cfg_path,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            service_kwargs=service_kwargs,
        )

        t0 = time.perf_counter()
        if warmed_store_path and Path(warmed_store_path).exists():
            self._service.load_checkpoint_snapshot(str(Path(warmed_store_path)))
        else:
            for idx, doc in enumerate(self._docs_raw):
                self._service.build_memory(
                    task_description="HotpotQA passage",
                    trajectory=f"Title: {doc['title']}\nText: {doc['text']}",
                    metadata={"source_benchmark": "hotpotqa", "corpus_idx": idx},
                )
        self._build_seconds = time.perf_counter() - t0
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_llm_build", self._build_seconds)
        self._counters.build_input_tokens += self._count_tokens(
            "\n\n".join(_docs_as_strings(self._docs_raw))
        )
        self._counters.total_input_tokens += self._counters.build_input_tokens
        self._shared_memory = {"service": self._service}

    @staticmethod
    def _load_memrl_yaml(path: Path) -> dict[str, Any]:
        try:
            import yaml
        except Exception as exc:
            raise MemRLNotConfiguredError("MemRL requires PyYAML to read mos_config_path YAML.") from exc
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise MemRLNotConfiguredError(f"Failed to read MemRL config: {path}") from exc
        if not isinstance(data, dict):
            raise MemRLNotConfiguredError(f"MemRL config must be a mapping: {path}")
        return data

    def _init_memrl_service(
        self,
        *,
        cfg_path: Path,
        llm_model: str,
        llm_base_url: str | None,
        llm_api_key: str,
        embedding_model: str,
        embedding_base_url: str | None,
        embedding_api_key: str | None,
        service_kwargs: dict[str, Any],
    ) -> Any:
        vendor_root = Path(__file__).resolve().parents[2] / "vendor" / "memrl"
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        try:
            from memrl.providers.embedding import OpenAIEmbedder
            from memrl.providers.llm import OpenAILLM
            from memrl.service import MemoryService
            from memrl.service.strategies import BuildStrategy, RetrieveStrategy, StrategyConfiguration, UpdateStrategy
            from memrl.service.value_driven import RLConfig
        except Exception as exc:
            raise MemRLNotConfiguredError(
                "MemRL upstream classes are not importable. Install vendor/memrl editably via pixi."
            ) from exc

        llm_cfg = dict(self._config.get("llm") or {})
        emb_cfg = dict(self._config.get("embedding") or {})
        mem_cfg = dict(self._config.get("memory") or {})
        exp_cfg = dict(self._config.get("experiment") or {})
        rl_cfg_raw = dict(self._config.get("rl_config") or {})

        llm_provider = OpenAILLM(
            api_key=llm_api_key or llm_cfg.get("api_key") or "EMPTY",
            base_url=llm_base_url or llm_cfg.get("base_url"),
            model=llm_model or llm_cfg.get("model") or "Qwen/Qwen3-32B",
            default_temperature=float(llm_cfg.get("temperature", 0.0)),
            default_max_tokens=int(llm_cfg.get("max_tokens", 4096)),
        )
        embedding_provider = OpenAIEmbedder(
            api_key=embedding_api_key or emb_cfg.get("api_key") or llm_api_key or "EMPTY",
            base_url=embedding_base_url or emb_cfg.get("base_url"),
            model=embedding_model or emb_cfg.get("model") or "text-embedding-3-large",
            max_text_len=int(emb_cfg.get("max_text_len", 4096)),
        )

        runtime_dir = Path(tempfile.mkdtemp(prefix="hotpotqa_memrl_mos_"))
        mos_json_path = runtime_dir / "mos_config.json"
        mos_json = {
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": llm_model or llm_cfg.get("model") or "Qwen/Qwen3-32B",
                    "api_key": llm_api_key or llm_cfg.get("api_key") or "EMPTY",
                    "api_base": llm_base_url or llm_cfg.get("base_url"),
                },
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": llm_model or llm_cfg.get("model") or "Qwen/Qwen3-32B",
                            "api_key": llm_api_key or llm_cfg.get("api_key") or "EMPTY",
                            "api_base": llm_base_url or llm_cfg.get("base_url"),
                        },
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "model_name_or_path": embedding_model or emb_cfg.get("model") or "text-embedding-3-large",
                            "api_key": embedding_api_key or emb_cfg.get("api_key") or llm_api_key or "EMPTY",
                            "base_url": embedding_base_url or emb_cfg.get("base_url"),
                        },
                    },
                    "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
                },
            },
            "user_manager": {"backend": "sqlite", "config": {"db_path": str(runtime_dir / "users.db")}},
            "top_k": self.retrieve_k,
        }
        mos_json_path.write_text(json.dumps(mos_json, indent=2), encoding="utf-8")

        rl_allowed = set(RLConfig.__dataclass_fields__)
        rl_config = RLConfig(**{k: v for k, v in rl_cfg_raw.items() if k in rl_allowed})
        kwargs = dict(service_kwargs)
        kwargs.setdefault("max_keywords", int(mem_cfg.get("max_keywords", 8)))
        kwargs.setdefault("add_similarity_threshold", float(mem_cfg.get("add_similarity_threshold", 0.90)))
        kwargs.setdefault("memory_confidence", float(mem_cfg.get("memory_confidence", 100.0)))
        kwargs.setdefault("enable_value_driven", bool(exp_cfg.get("enable_value_driven", False)))
        kwargs.setdefault("rl_config", rl_config)
        kwargs.setdefault("sim_norm_mean", mem_cfg.get("sim_norm_mean", 0.5187))
        kwargs.setdefault("sim_norm_std", mem_cfg.get("sim_norm_std", 0.1203))
        return MemoryService(
            mos_config_path=str(mos_json_path),
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            strategy_config=StrategyConfiguration(
                BuildStrategy(mem_cfg.get("build_strategy", "trajectory")),
                RetrieveStrategy(mem_cfg.get("retrieve_strategy", "query")),
                UpdateStrategy(mem_cfg.get("update_strategy", "adjustment")),
            ),
            user_id=str(mem_cfg.get("user_id", "hotpotqa_memrl")),
            num_workers=int(exp_cfg.get("batch_size", 8)),
            **kwargs,
        )

    @property
    def counters(self) -> EfficiencyCounters:
        return self._counters

    @property
    def build_seconds(self) -> float:
        return self._build_seconds

    def reset_counters(self) -> EfficiencyCounters:
        self._counters = EfficiencyCounters()
        return self._counters

    def build_from_sample(self, sample: dict[str, Any], *, sample_id: str) -> Any:
        return self._shared_memory

    def build(self, traj_text: str, *, task: str = "") -> Any:
        return self._shared_memory

    def answer(self, memory: Any, question: str) -> str:
        service = (memory or self._shared_memory)["service"]
        t_tool = time.perf_counter()
        retrieved = service.retrieve(task_description=question, k=self.retrieve_k)
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_tool", time.perf_counter() - t_tool)
        context = self._format_evidence(retrieved)
        self._counters.record_retrieval(
            candidates_scored=0,
            evidence_injected=len(retrieved) if isinstance(retrieved, list) else 0,
            context_tokens=self._count_tokens(context),
        )
        return context

    @staticmethod
    def _format_evidence(results: Any) -> str:
        if isinstance(results, str):
            return results
        if isinstance(results, list):
            parts = []
            for item in results:
                if isinstance(item, dict):
                    parts.append(str(item.get("memory") or item.get("content") or item.get("full_content") or item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(results or "")

    def _count_tokens(self, text: Any) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0
        return len(str(text or "").split())
