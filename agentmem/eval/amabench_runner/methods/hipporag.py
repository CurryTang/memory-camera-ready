"""HippoRAGv2 retrieval method for AMAbench.

Keep the wrapper close to the paper-described/default HippoRAG behavior:
- per-turn passages for indexing
- internal retrieval uses linking_top_k=5 and retrieval_top_k=200
- QA context returns the top 5 passages
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional

from agentmem.eval.amabench_runner.methods.base import BaseMethod

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PERSISTENT_ROOT = os.environ.get("AGENTMEM_PERSISTENT_ROOT")
_HIPPORAG_FALLBACK_SRC_CANDIDATES = [
    _REPO_ROOT / "vendor" / "hipporag" / "src",
]
if _PERSISTENT_ROOT:
    _HIPPORAG_FALLBACK_SRC_CANDIDATES.append(
        Path(_PERSISTENT_ROOT) / "vendor" / "hipporag" / "src"
    )

def _install_vllm_stub() -> None:
    """Block HippoRAG's optional offline-vLLM imports from loading local vllm."""
    import types as _types

    _vllm = _types.ModuleType("vllm")
    _vllm.SamplingParams = type("SamplingParams", (), {})                              
    _vllm.LLM = type("LLM", (), {})                              
    _vllm.__path__ = []                                               
    sys.modules["vllm"] = _vllm
    for _name in (
        "vllm._C",
        "vllm.model_executor",
        "vllm.model_executor.guided_decoding",
        "vllm.model_executor.guided_decoding.guided_fields",
    ):
        _mod = _types.ModuleType(_name)
        if _name.endswith("guided_fields"):
            _mod.GuidedDecodingRequest = type("GuidedDecodingRequest", (), {})                              
        sys.modules[_name] = _mod

def _ensure_hipporag_importable() -> None:

    _install_vllm_stub()

    try:
        import ipdb as _ipdb

        def _no_set_trace(*a, **kw):
            raise RuntimeError("ipdb.set_trace hit in hipporag encode path")
        _ipdb.set_trace = _no_set_trace
    except Exception:
        pass
    try:
        import hipporag              

        _patch_openie_json_parsers()

        try:
            from hipporag.embedding_model import _get_embedding_model_class as _orig_get
            from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel

            def _patched_get(embedding_model_name: str = "nvidia/NV-Embed-v2"):
                try:
                    return _orig_get(embedding_model_name)
                except AssertionError:

                    return OpenAIEmbeddingModel

            import hipporag.embedding_model as _em
            _em._get_embedding_model_class = _patched_get

            for _mod_name in list(sys.modules.keys()):
                if _mod_name.endswith("HippoRAG") or _mod_name.endswith("StandardRAG"):
                    _mod = sys.modules[_mod_name]
                    if hasattr(_mod, "_get_embedding_model_class"):
                        setattr(_mod, "_get_embedding_model_class", _patched_get)

            try:
                from hipporag import HippoRAG as _HippoRAG_cls
                _HippoRAG_cls.__init__.__globals__["_get_embedding_model_class"] = _patched_get
            except Exception:
                pass
            try:
                from hipporag import StandardRAG as _StandardRAG_cls
                _StandardRAG_cls.__init__.__globals__["_get_embedding_model_class"] = _patched_get
            except Exception:
                pass
        except Exception:
            pass                                                                         

        return
    except ImportError:
        pass
    for candidate in _HIPPORAG_FALLBACK_SRC_CANDIDATES:
        if (candidate / "hipporag" / "__init__.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise ImportError(
        "hipporag is not importable. Install the official PyPI package "
        "(``pip install hipporag`` / pixi dep) or provide a vendor/hipporag/src "
        "checkout."
    )

def _patch_openie_json_parsers() -> None:
    """Accept common OpenAI-compatible JSON variants in HippoRAG OpenIE.

    The official parser expects a bare object like
    ``{"named_entities": [...]}``. Several OpenAI-compatible local models
    return valid fenced JSON, and for NER often return the entity list itself.
    This keeps the HippoRAG prompts and extraction pipeline unchanged while
    making the response parser tolerant to those equivalent JSON encodings.
    """
    try:
        from hipporag.information_extraction import openie_openai as _openie_mod
    except Exception:
        return

    original = getattr(_openie_mod, "_extract_ner_from_response", None)
    if getattr(original, "_agentmem_patched", False):
        return

    def _strip_fence(text: str) -> str:
        stripped = str(text or "").strip()
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else stripped

    def _coerce_entities(obj: Any) -> list[str]:
        if isinstance(obj, dict):
            obj = obj.get("named_entities", obj.get("entities", []))
        if isinstance(obj, list):
            entities: list[str] = []
            for x in obj:
                if x is Ellipsis:
                    continue
                value = str(x).strip()
                if value and value != "...":
                    entities.append(value)
            return entities
        return []

    def _patched_extract_ner(real_response: str) -> list[str]:
        if callable(original):
            try:
                return _coerce_entities(original(real_response))
            except Exception:
                pass

        cleaned = _strip_fence(real_response)
        try:
            entities = _coerce_entities(json.loads(cleaned))
            if entities:
                return entities
        except Exception:
            pass

        object_match = re.search(r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}', cleaned, re.DOTALL)
        if object_match:
            try:
                return _coerce_entities(json.loads(object_match.group(0)))
            except Exception:
                pass

        array_match = re.search(r"\[[\s\S]*\]", cleaned)
        if array_match:
            try:
                return _coerce_entities(json.loads(array_match.group(0)))
            except Exception:
                pass
        return []

    _patched_extract_ner._agentmem_patched = True                              
    _openie_mod._extract_ner_from_response = _patched_extract_ner
    try:
        _openie_mod.OpenIE.ner.__globals__["_extract_ner_from_response"] = _patched_extract_ner
    except Exception:
        pass

class HippoRAGMemory:
    def __init__(
        self,
        hipporag_instance: Any,
        *,
        fallback_docs: list[str] | None = None,
        fallback_reason: str | None = None,
    ):
        self.hipporag = hipporag_instance
        self.fallback_docs = fallback_docs or []
        self.fallback_reason = fallback_reason

@contextmanager
def _temporary_api_env(api_key: Optional[str]):
    if not api_key:
        yield
        return

    old_openai = os.environ.get("OPENAI_API_KEY")
    old_openrouter = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENROUTER_API_KEY"] = api_key
    try:
        yield
    finally:
        if old_openai is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_openai
        if old_openrouter is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = old_openrouter

class HippoRAGMethod(BaseMethod):
    """Wraps HippoRAGv2 for per-episode index→retrieve on AMABench.

    The goal here is fidelity, not task-specific prompt engineering:
    trajectory turns are indexed as individual passages and QA consumes the
    top 5 passages, matching the AMABench appendix description.
    """

    def __init__(
        self,
        top_k: int = 5,
        chunk_turns: int = 1,
        retrieval_top_k: int = 200,
        linking_top_k: int = 5,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: str = "http://localhost:30000/v1",
        llm_api_key: Optional[str] = None,
        embedding_model: str = "Qwen3-Embedding-4B",
        embedding_base_url: Optional[str] = "http://localhost:30001/v1",
        embedding_api_key: Optional[str] = None,
        save_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        **_kw,
    ):
        _ensure_hipporag_importable()
        default_top_k = 5
        default_chunk_turns = 1
        default_retrieval_top_k = 200
        default_linking_top_k = 5
        default_llm_model = "Qwen/Qwen3-32B"
        default_llm_base_url = "http://localhost:30000/v1"
        default_embedding_model = "Qwen3-Embedding-4B"

        if config_path:
            cfg = self._load_config(config_path)
            if top_k == default_top_k:
                top_k = cfg.get("top_k", top_k)
            if chunk_turns == default_chunk_turns:
                chunk_turns = cfg.get("chunk_turns", chunk_turns)
            if retrieval_top_k == default_retrieval_top_k:
                retrieval_top_k = cfg.get("retrieval_top_k", retrieval_top_k)
            if linking_top_k == default_linking_top_k:
                linking_top_k = cfg.get("linking_top_k", linking_top_k)
            if llm_model == default_llm_model:
                llm_model = cfg.get("llm_model", llm_model)
            if llm_base_url == default_llm_base_url:
                llm_base_url = cfg.get("llm_base_url", llm_base_url)
            if llm_api_key is None:
                llm_api_key = cfg.get("llm_api_key", llm_api_key)
            if embedding_model == default_embedding_model:
                embedding_model = cfg.get("embedding_model", embedding_model)
            if embedding_base_url is None:
                embedding_base_url = cfg.get("embedding_base_url", embedding_base_url)
            if embedding_api_key is None:
                embedding_api_key = cfg.get("embedding_api_key", embedding_api_key)
            if save_dir is None:
                save_dir = cfg.get("save_dir", save_dir)

        self.top_k = top_k
        self.chunk_turns = chunk_turns
        self.retrieval_top_k = retrieval_top_k
        self.linking_top_k = linking_top_k
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.embedding_api_key = embedding_api_key
        if save_dir is None:
            env_dir = os.environ.get("AMABENCH_HIPPORAG_SAVE_DIR")
            if env_dir:
                save_dir = env_dir
        if save_dir is None:
            save_dir = tempfile.mkdtemp(prefix="hipporag_amabench_")
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self._episode_counter = 0

    def memory_construction(self, traj_text: str, task: str = "") -> HippoRAGMemory:
        try:
            from hipporag import HippoRAG
            from hipporag.utils.config_utils import BaseConfig
        except ModuleNotFoundError as exc:
            checked = ", ".join(str(path) for path in _HIPPORAG_FALLBACK_SRC_CANDIDATES)
            if exc.name == "hipporag":
                raise ModuleNotFoundError(
                    "HippoRAG package not found. Checked source roots: "
                    f"{checked}. Sync the HippoRAG source tree or install it with "
                    "`pip install -e resource/.../HippoRAG` on the target machine."
                ) from exc
            raise ModuleNotFoundError(
                "HippoRAG import failed while loading the source tree. "
                f"Checked source roots: {checked}. Missing dependency: {exc.name!r}. "
                "Install the HippoRAG runtime extras on the target machine "
                "(for example `igraph`, `litellm`, and `gritlm`)."
            ) from exc

        self._episode_counter += 1
        episode_dir = os.path.join(self.save_dir, f"episode_{self._episode_counter}")
        os.makedirs(episode_dir, exist_ok=True)

        config_kwargs = dict(
            save_dir=episode_dir,
            llm_name=self.llm_model,
            llm_base_url=self.llm_base_url,
            embedding_model_name=self.embedding_model,
            embedding_base_url=self.embedding_base_url,
            linking_top_k=self.linking_top_k,
            retrieval_top_k=self.retrieval_top_k,
            qa_top_k=self.top_k,
            max_qa_steps=1,
            openie_mode="online",
            temperature=0,
            embedding_batch_size=int(os.environ.get("AMABENCH_HIPPORAG_EMBED_BATCH_SIZE", "1")),
            embedding_max_seq_len=int(os.environ.get("AMABENCH_HIPPORAG_EMBED_MAX_TOKENS", "32000")),
        )

        if self.embedding_api_key:
            try:
                import inspect
                if "embedding_api_key" in inspect.signature(BaseConfig.__init__).parameters:
                    config_kwargs["embedding_api_key"] = self.embedding_api_key
            except Exception:
                pass
        config = BaseConfig(**config_kwargs)

        docs = _split_trajectory_chunked(traj_text, task, self.chunk_turns)

        try:
            from hipporag.embedding_model import _get_embedding_model_class as _cur
            from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel

            _orig = getattr(_cur, "_hipporag_orig", _cur)

            def _patched(embedding_model_name: str = "nvidia/NV-Embed-v2"):
                try:
                    return _orig(embedding_model_name)
                except AssertionError:
                    return OpenAIEmbeddingModel

            _patched._hipporag_orig = _orig                              
            HippoRAG.__init__.__globals__["_get_embedding_model_class"] = _patched
        except Exception:
            pass
        with _temporary_api_env(self.llm_api_key):
            hipporag = HippoRAG(global_config=config)
            try:
                hipporag.index(docs=docs)
            except ZeroDivisionError as exc:

                return HippoRAGMemory(
                    hipporag,
                    fallback_docs=docs,
                    fallback_reason=f"empty_openie_graph: {exc}",
                )
            except Exception as exc:
                message = str(exc).lower()
                overflow_markers = (
                    "context length",
                    "maximum context",
                    "max context",
                    "too many tokens",
                    "token indices sequence length",
                    "40961",
                )
                if not any(marker in message for marker in overflow_markers):
                    raise
                return HippoRAGMemory(
                    hipporag,
                    fallback_docs=docs,
                    fallback_reason=f"openie_context_overflow: {exc}",
                )

        return HippoRAGMemory(hipporag)

    def memory_retrieve(self, memory: HippoRAGMemory, question: str) -> str:
        if memory.fallback_docs:
            ranked = _rank_docs_lexical(memory.fallback_docs, question)
            return "\n\n".join(ranked[: self.top_k])
        if _hipporag_dense_only(memory.hipporag):
            return "\n\n".join(_dense_or_lexical_docs(memory.hipporag, question, self.top_k))
        try:
            solutions = memory.hipporag.retrieve(queries=[question])
        except (AssertionError, ValueError) as exc:
            message = str(exc)
            empty_graph = (
                "No phrases found in the graph" in message
                or "shapes (0,)" in message
                or "not aligned: 0" in message
            )
            if not empty_graph:
                raise
            return "\n\n".join(_dense_or_lexical_docs(memory.hipporag, question, self.top_k))
        if isinstance(solutions, tuple):
            solutions = solutions[0]

        if solutions and solutions[0].docs:
            return "\n\n".join(solutions[0].docs[: self.top_k])
        return ""

def _split_trajectory_chunked(
    traj_text: str, task: str = "", chunk_turns: int = 3
) -> List[str]:
    """Split trajectory text into ordered passage documents.

    `chunk_turns=1` keeps one turn per passage, which is the closest match to
    the paper's "top 5 passages" retrieval setup on trajectory text.
    """

    turns: list[str] = []
    current: list[str] = []
    structured_markers = 0
    for line in traj_text.split("\n"):
        if line.strip().startswith(("Turn ", "Step ")):
            structured_markers += 1
            if current:
                turns.append("\n".join(current))
                current = []
        current.append(line)
    if current:
        turns.append("\n".join(current))

    if not turns or structured_markers == 0:
        raw = traj_text.strip()
        if not raw:
            return []
        words = raw.split()
        chunk_words = max(64, int(os.environ.get("AMABENCH_HIPPORAG_RAW_CHUNK_WORDS", "1600")))
        overlap_words = max(0, int(os.environ.get("AMABENCH_HIPPORAG_RAW_CHUNK_OVERLAP_WORDS", "160")))
        step = max(1, chunk_words - overlap_words)
        docs = []
        for idx, start in enumerate(range(0, len(words), step), start=1):
            chunk = " ".join(words[start : start + chunk_words]).strip()
            if not chunk:
                continue
            prefix = f"# Task\n{task}\n\n" if task and idx == 1 else ""
            docs.append(f"{prefix}Raw haystack chunk {idx}:\n{chunk}")
            if start + chunk_words >= len(words):
                break
        return _limit_hipporag_docs(docs if docs else [raw])

    docs: list[str] = []
    for i in range(0, len(turns), chunk_turns):
        chunk = "\n".join(turns[i : i + chunk_turns])
        docs.extend(_split_long_doc(chunk))

    return _limit_hipporag_docs(docs)

def _split_long_doc(text: str) -> list[str]:
    """Keep each HippoRAG passage inside the embedding endpoint budget."""
    token_cap = int(
        os.environ.get("HIPPO_OPENIE_MAX_INPUT_TOKENS")
        or os.environ.get("AMABENCH_HIPPORAG_MAX_INPUT_TOKENS")
        or "2000"
    )
    text = str(text or "").strip()
    if not text:
        return []
    overlap_tokens = max(0, int(os.environ.get("AMABENCH_HIPPORAG_DOC_OVERLAP_TOKENS", "256")))
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= token_cap:
            return [text]
        step = max(1, token_cap - min(overlap_tokens, token_cap // 2))
        chunks: list[str] = []
        for start in range(0, len(tokens), step):
            chunk = enc.decode(tokens[start : start + token_cap]).strip()
            if chunk:
                chunks.append(chunk)
            if start + token_cap >= len(tokens):
                break
        return chunks
    except Exception:
        pass

    max_chars = max(1024, int(os.environ.get("AMABENCH_HIPPORAG_DOC_MAX_CHARS", str(token_cap * 4))))
    overlap_chars = max(0, int(os.environ.get("AMABENCH_HIPPORAG_DOC_OVERLAP_CHARS", str(overlap_tokens * 4))))
    if len(text) <= max_chars:
        return [text]
    step = max(1, max_chars - overlap_chars)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        if start + max_chars >= len(text):
            break
    return chunks

def _limit_hipporag_docs(docs: list[str]) -> list[str]:
    max_docs = int(os.environ.get("AMABENCH_HIPPORAG_MAX_DOCS", "80") or 0)
    if max_docs > 0 and len(docs) > max_docs:
        return docs[:max_docs]
    return docs

def _hipporag_dense_only(hipporag: Any) -> bool:
    try:
        info = hipporag.get_graph_info()
        return int(info.get("num_phrase_nodes") or 0) == 0 or int(info.get("num_extracted_triples") or 0) == 0
    except Exception:
        return False

def _dense_or_lexical_docs(hipporag: Any, question: str, top_k: int) -> list[str]:
    try:
        if not getattr(hipporag, "ready_to_retrieve", False):
            hipporag.prepare_retrieval_objects()
        sorted_doc_ids, _ = hipporag.dense_passage_retrieval(question)
        docs = [
            hipporag.chunk_embedding_store.get_row(hipporag.passage_node_keys[int(idx)])["content"]
            for idx in list(sorted_doc_ids)[:top_k]
        ]
        if docs:
            return docs
    except Exception:
        pass
    try:
        all_docs = list(hipporag.chunk_embedding_store.get_text_for_all_rows().values())
        contents = [str(row.get("content") or "") for row in all_docs]
    except Exception:
        contents = []
    return _rank_docs_lexical(contents, question)[:top_k]

def _rank_docs_lexical(docs: list[str], query: str) -> list[str]:
    query_tokens = set(re.findall(r"[A-Za-z0-9]+", query.lower()))
    if not query_tokens:
        return docs

    def score(doc: str) -> tuple[int, int]:
        doc_tokens = set(re.findall(r"[A-Za-z0-9]+", doc.lower()))
        return (len(query_tokens & doc_tokens), len(doc))

    return sorted(docs, key=score, reverse=True)
