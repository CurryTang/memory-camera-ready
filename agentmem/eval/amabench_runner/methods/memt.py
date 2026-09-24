"""Mem-T method for AMAbench.

Uses the native Mem-T runtime engine (formation → update → ChromaDB storage)
for memory construction, and multi-step tool-based retrieval for QA.

Mem-T's retrieve_and_answer produces the final answer internally, so we
return it with the ###Answer: prefix to bypass the runner's LLM call.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from agentmem.eval.amabench_runner.methods.base import BaseMethod

MEMT_ANSWER_PREFIX = "###Answer: "
_DEFAULT_LARGE_CORPUS_PARALLEL = 32
_MEMT_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "SILICON_API_KEY",
    "SILICON_BASE_URL",
    "MEMT_EMBEDDING_PROVIDER",
    "MEMT_EMBEDDING_MODEL",
)
_MEMT_ENV_LOCK = threading.Lock()
_MEMT_CLIENT_PATCHED = False
_MEMT_MAB_PROMPT_PATCHED = False

def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _patch_memt_client_context_budget() -> None:
    """Cap Mem-T user messages before upstream OpenAI-compatible calls.

    The upstream client only caps output tokens. On long MemoryAgentBench
    haystacks, vLLM rejects formation prompts that exceed Qwen3-32B's 32768
    context by one token and the client retries the deterministic 400 error.
    """
    global _MEMT_CLIENT_PATCHED
    if _MEMT_CLIENT_PATCHED:
        return
    try:
        from llm_api import OpenAIAPIClient
    except Exception:
        return

    original_construct = OpenAIAPIClient._construct_messages
    original_get_completion = OpenAIAPIClient.get_completion

    def _construct_messages(self: Any, prompt_or_messages: Any, system_prompt: Optional[str]) -> list[dict]:
        messages = original_construct(self, prompt_or_messages, system_prompt)
        max_tokens = int(os.environ.get("MEMT_MAX_INPUT_TOKENS", "30000"))
        if max_tokens <= 0:
            return messages
        max_chars = int(os.environ.get("MEMT_MAX_INPUT_CHARS", str(max_tokens * 4)))
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = None

        trimmed = []
        for msg in messages:
            item = dict(msg)
            if item.get("role") == "user":
                content = str(item.get("content", ""))
                if enc is not None:
                    tokens = enc.encode(content)
                    if len(tokens) > max_tokens:
                        item["content"] = enc.decode(tokens[:max_tokens])
                elif len(content) > max_chars:
                    item["content"] = content[:max_chars]
            trimmed.append(item)
        return trimmed

    def get_completion(self: Any, prompt_or_messages: Any, *args: Any, **kwargs: Any) -> str:
        return original_get_completion(self, _cap_prompt_or_messages(prompt_or_messages), *args, **kwargs)

    OpenAIAPIClient._construct_messages = _construct_messages
    OpenAIAPIClient.get_completion = get_completion
    _MEMT_CLIENT_PATCHED = True

def _cap_prompt_or_messages(prompt_or_messages: Any) -> Any:
    max_tokens = int(os.environ.get("MEMT_MAX_INPUT_TOKENS", "30000"))
    if max_tokens <= 0:
        return prompt_or_messages
    max_total_tokens = int(os.environ.get("MEMT_MAX_TOTAL_INPUT_TOKENS", str(max_tokens)))
    max_chars = int(os.environ.get("MEMT_MAX_INPUT_CHARS", str(max_tokens * 4)))
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

    def trim_text(value: Any) -> str:
        content = str(value or "")
        if enc is not None:
            tokens = enc.encode(content)
            if len(tokens) > max_tokens:
                return enc.decode(tokens[:max_tokens])
            return content
        return content[:max_chars] if len(content) > max_chars else content

    if isinstance(prompt_or_messages, str):
        return trim_text(prompt_or_messages)
    if isinstance(prompt_or_messages, list):
        out: list[dict[str, Any]] = []
        for msg in prompt_or_messages:
            item = dict(msg)
            if item.get("role") == "user":
                item["content"] = trim_text(item.get("content", ""))
            out.append(item)
        if enc is not None and max_total_tokens > 0:
            out = _cap_message_list_total_tokens(out, max_total_tokens, enc)
        elif max_total_tokens > 0:
            out = _cap_message_list_total_chars(out, max_total_tokens * 4)
        return out
    return prompt_or_messages

def _cap_message_list_total_tokens(messages: list[dict[str, Any]], max_total_tokens: int, enc: Any) -> list[dict[str, Any]]:
    """Trim accumulated ReAct observations to keep the whole chat below budget."""

    def count(msgs: list[dict[str, Any]]) -> int:

        return sum(len(enc.encode(str(msg.get("content", "")))) + 8 for msg in msgs)

    out = [dict(msg) for msg in messages]
    if count(out) <= max_total_tokens:
        return out

    candidate_indices = [
        idx
        for idx, msg in enumerate(out)
        if idx >= 2 and msg.get("role") == "user" and "<observation>" in str(msg.get("content", ""))
    ]
    candidate_indices += [
        idx
        for idx, msg in enumerate(out)
        if idx >= 2 and idx not in candidate_indices and msg.get("role") in {"user", "assistant"}
    ]

    for idx in candidate_indices:
        if count(out) <= max_total_tokens:
            break
        content = str(out[idx].get("content", ""))
        tokens = enc.encode(content)
        if not tokens:
            continue
        overflow = count(out) - max_total_tokens
        keep = max(256, len(tokens) - overflow - 128)
        if keep < len(tokens):
            out[idx]["content"] = enc.decode(tokens[:keep])

    while len(out) > 2 and count(out) > max_total_tokens:
        del out[2]
    return out

def _cap_message_list_total_chars(messages: list[dict[str, Any]], max_total_chars: int) -> list[dict[str, Any]]:
    out = [dict(msg) for msg in messages]

    def count(msgs: list[dict[str, Any]]) -> int:
        return sum(len(str(msg.get("content", ""))) for msg in msgs)

    for idx, msg in enumerate(out):
        if count(out) <= max_total_chars:
            break
        if idx < 2 or msg.get("role") not in {"user", "assistant"}:
            continue
        content = str(msg.get("content", ""))
        overflow = count(out) - max_total_chars
        keep = max(1024, len(content) - overflow - 512)
        if keep < len(content):
            out[idx]["content"] = content[:keep]
    while len(out) > 2 and count(out) > max_total_chars:
        del out[2]
    return out

def _patch_memt_client_instance(client: Any) -> None:
    if client is None or getattr(client, "_agentmem_context_capped", False):
        return
    original = client.get_completion

    def get_completion(prompt_or_messages: Any, *args: Any, **kwargs: Any) -> str:
        return original(_cap_prompt_or_messages(prompt_or_messages), *args, **kwargs)

    client.get_completion = get_completion
    client._agentmem_context_capped = True

def _patch_memt_memoryagentbench_prompt() -> None:
    """Teach the vendored Mem-T prompt formatter MemoryAgentBench output rules."""
    global _MEMT_MAB_PROMPT_PATCHED
    if _MEMT_MAB_PROMPT_PATCHED:
        return
    try:
        import memory_retrieval
    except Exception:
        return

    original = memory_retrieval.get_final_result_format

    def get_final_result_format(benchmark_name: str, category: str = "") -> str:
        if str(benchmark_name).lower() == "amabench":
            category_norm = str(category or "").upper()
            if category_norm == "TTL":
                return (
                    "The Final Result's Format Must Follow These Rules:\n"
                    "1. This is a MemoryAgentBench test-time-learning query.\n"
                    "2. The retrieved examples map utterances to labels or outputs.\n"
                    "3. Return exactly the learned label/output requested by the query.\n"
                    "4. Do not return the semantic real-world answer, explanation, or extra words."
                )
            return (
                "The Final Result's Format Must Follow These Rules:\n"
                "1. Answer using only the retrieved MemoryAgentBench memories.\n"
                "2. Return only the concise final answer string.\n"
                "3. Do not include explanation or tool traces."
            )
        return original(benchmark_name, category)

    memory_retrieval.get_final_result_format = get_final_result_format
    _MEMT_MAB_PROMPT_PATCHED = True

@contextmanager
def _memt_env(
    api_key: str,
    base_url: str,
    *,
    embedding_model: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
):
    """Temporarily expose endpoint settings expected by the original Mem-T client."""
    with _MEMT_ENV_LOCK:
        saved = {key: os.environ.get(key) for key in _MEMT_ENV_KEYS}
        openai_base_url = embedding_base_url or base_url
        openai_api_key = embedding_api_key or api_key
        overrides = {

            "OPENAI_API_KEY": openai_api_key,
            "OPENAI_BASE_URL": openai_base_url,
            "OPENAI_API_BASE": openai_base_url,

            "SILICON_API_KEY": api_key,
            "SILICON_BASE_URL": base_url,
        }
        if embedding_model:
            overrides["MEMT_EMBEDDING_MODEL"] = embedding_model
        if embedding_base_url:
            overrides["MEMT_EMBEDDING_PROVIDER"] = "openai"
        try:
            for key, value in overrides.items():
                os.environ[key] = value
            yield
        finally:
            for key, old_value in saved.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

class MemTMemory:
    """Holds Mem-T runtime state after construction."""

    def __init__(self, engine: Any, sample_id: str, category: str = "", raw_chunks: Optional[list[str]] = None):
        self.engine = engine
        self.sample_id = sample_id
        self.category = category
        self.raw_chunks = list(raw_chunks or [])

        self.lock = threading.Lock()

class MemTMethod(BaseMethod):
    """Wraps Mem-T runtime for per-episode memory build + QA on AMABench.

    Mem-T produces answers directly via retrieve_and_answer(), so
    memory_retrieve returns the full answer prefixed with '###Answer: '
    so the runner picks it up without an extra LLM call.
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        db_type: str = "persistent",
        db_host: str = "localhost",
        db_port: int = 8070,
        db_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        k_turns: int = 4,
        update_retrieval_topk: int = 3,
        retrieval_topk: int = 5,
        max_tool_steps: int = 3,
        benchmark_name: str = "amabench",
        disable_thinking: bool = True,
        max_concurrent_requests: int = 8,
        large_corpus: Optional[bool] = None,
        disable_per_fact_merge: Optional[bool] = None,
        build_session_summary: Optional[bool] = None,
        large_corpus_parallel: Optional[int] = None,
        config_path: Optional[str] = None,
        **_kw,
    ):
        default_llm_model = "Qwen/Qwen3-32B"
        default_llm_base_url = "http://localhost:30000/v1"
        llm_model = llm_model or os.environ.get("RQ1_LLM_MODEL", default_llm_model)
        llm_base_url = llm_base_url or os.environ.get("RQ1_LLM_URL", default_llm_base_url)
        embedding_model = embedding_model or os.environ.get("RQ1_EMB_MODEL", "Qwen3-Embedding-4B")
        embedding_base_url = embedding_base_url or os.environ.get("RQ1_EMB_URL")
        embedding_api_key = embedding_api_key or os.environ.get("RQ1_EMB_API_KEY")
        if config_path:
            cfg = self._load_config(config_path)
            if llm_model == os.environ.get("RQ1_LLM_MODEL", default_llm_model):
                llm_model = cfg.get("llm_model", llm_model)
            if llm_base_url == os.environ.get("RQ1_LLM_URL", default_llm_base_url):
                llm_base_url = cfg.get("llm_base_url", llm_base_url)
            llm_api_key = cfg.get("llm_api_key", llm_api_key)
            db_type = cfg.get("db_type", db_type)
            db_host = cfg.get("db_host", db_host)
            db_port = cfg.get("db_port", db_port)
            db_path = cfg.get("db_path", db_path)
            embedding_model = cfg.get("embedding_model", embedding_model)
            embedding_base_url = cfg.get("embedding_base_url", embedding_base_url)
            embedding_api_key = cfg.get("embedding_api_key", embedding_api_key)
            k_turns = cfg.get("k_turns", k_turns)
            update_retrieval_topk = cfg.get("update_retrieval_topk", update_retrieval_topk)
            retrieval_topk = cfg.get("retrieval_topk", retrieval_topk)
            max_tool_steps = cfg.get("max_tool_steps", max_tool_steps)
            benchmark_name = cfg.get("benchmark_name", benchmark_name)
            disable_thinking = cfg.get("disable_thinking", disable_thinking)
            max_concurrent_requests = cfg.get("max_concurrent_requests", max_concurrent_requests)
            large_corpus = cfg.get("large_corpus", cfg.get("fast_build", large_corpus))
            disable_per_fact_merge = cfg.get("disable_per_fact_merge", disable_per_fact_merge)
            build_session_summary = cfg.get("build_session_summary", build_session_summary)
            large_corpus_parallel = cfg.get("large_corpus_parallel", large_corpus_parallel)

        if large_corpus is None and (build_session_summary is True or disable_per_fact_merge is False):
            large_corpus = False

        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key or os.environ.get("RQ1_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.db_type = db_type
        self.db_host = db_host
        self.db_port = db_port
        self.db_path = db_path or tempfile.mkdtemp(prefix="memt_amabench_")
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.embedding_api_key = embedding_api_key
        self.k_turns = k_turns
        self.update_retrieval_topk = update_retrieval_topk
        self.retrieval_topk = retrieval_topk
        self.max_tool_steps = max_tool_steps
        self.benchmark_name = benchmark_name
        self.disable_thinking = disable_thinking
        self.max_concurrent_requests = max_concurrent_requests
        self.large_corpus = _env_bool("MEMT_LARGE_CORPUS", default=True) if large_corpus is None else bool(large_corpus)
        self.disable_per_fact_merge = (
            self.large_corpus if disable_per_fact_merge is None else bool(disable_per_fact_merge)
        )
        self.build_session_summary = (
            not self.large_corpus if build_session_summary is None else bool(build_session_summary)
        )
        if not self.disable_per_fact_merge or self.build_session_summary:
            self.large_corpus = False
        self.large_corpus_parallel = max(
            1,
            int(
                large_corpus_parallel
                if large_corpus_parallel is not None
                else os.environ.get("MEMT_LARGE_CORPUS_PARALLEL", _DEFAULT_LARGE_CORPUS_PARALLEL)
            ),
        )
        self._episode_counter = 0
        self._last_trajectory: Optional[dict] = None
        self.last_trace: list[dict[str, Any]] = []
        self.last_route: dict[str, Any] = {}

    def _make_engine(self):
        """Create a fresh MemTRuntimeEngine for one episode."""
        from agentmem.memt import MemTConfig, MemTRuntimeEngine
        _patch_memt_client_context_budget()
        _patch_memt_memoryagentbench_prompt()

        config = MemTConfig(
            db_type=self.db_type,
            db_host=self.db_host,
            db_port=self.db_port,
            db_path=self.db_path,
            k_turns=self.k_turns,
            update_retrieval_topk=self.update_retrieval_topk,
            retrieval_topk=self.retrieval_topk,
            max_tool_steps=self.max_tool_steps,
            max_context_tokens=int(os.environ.get("MEMT_RETRIEVAL_CONTEXT_TOKENS", "2048")),
            strong_model=self.llm_model,
            data_name=self.benchmark_name,
            temperature=0.0,
        )

        with _memt_env(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            embedding_model=self.embedding_model,
            embedding_base_url=self.embedding_base_url,
            embedding_api_key=self.embedding_api_key,
        ):
            engine = MemTRuntimeEngine(config=config)
        _patch_memt_client_instance(getattr(engine, "_llm", None))
        return engine

    def memory_construction(self, traj_text: str, task: str = "") -> MemTMemory:
        self._episode_counter += 1
        sample_id = f"amabench_ep_{self._episode_counter}"
        category = _category_from_task(task)

        engine = self._make_engine()

        sample = _traj_text_to_sample(traj_text, task, sample_id)
        with _memt_env(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            embedding_model=self.embedding_model,
            embedding_base_url=self.embedding_base_url,
            embedding_api_key=self.embedding_api_key,
        ):
            if self.large_corpus:
                self._large_corpus_build(engine, sample, sample_id)
            else:
                engine.build_from_sample(sample)

        if hasattr(engine, "get_bank"):
            try:
                memory_counts = engine.get_bank(sample_id).get_counts()
            except Exception:
                memory_counts = {}
            self._last_trajectory = {
                "memt_num_sessions": len(sample.get("conversation", [])),
                "memt_num_batches": sum(
                    (len(session.get("session_turns", [])) + self.k_turns - 1) // self.k_turns
                    for session in sample.get("conversation", [])
                ),
                "memt_memory_counts": memory_counts,
                "memt_large_corpus": self.large_corpus,
            }
        else:
            self._last_trajectory = None

        return MemTMemory(
            engine=engine,
            sample_id=sample_id,
            category=category,
            raw_chunks=_raw_exact_records(traj_text)
            or [turn["text"] for session in sample.get("conversation", []) for turn in session.get("session_turns", [])],
        )

    def _large_corpus_build(self, engine: Any, sample: dict[str, Any], sample_id: str) -> None:
        """Build AMABench MemT memory with bounded prompt growth.

        This keeps the load-bearing raw-turn and fact/experience memories used
        by retrieval, while skipping the per-session running summary and
        per-fact merge/update LLM calls that dominate AMABench build time.
        """
        builder = getattr(engine, "_builder")
        builder._init_sample_collections(sample_id)
        c_turns = f"{sample_id}_{builder.BASE_C_TURNS}"
        c_facts = f"{sample_id}_{builder.BASE_C_FACTS}"
        c_exp = f"{sample_id}_{builder.BASE_C_EXPERIENCES}"

        jobs: list[tuple[dict[str, Any], int, list[dict[str, Any]]]] = []
        for session in sample.get("conversation", []):
            turns = list(session.get("session_turns", []))
            for start in range(0, len(turns), builder.k_turns):
                jobs.append((session, start // builder.k_turns, turns[start:start + builder.k_turns]))

        def _process_batch(
            session: dict[str, Any],
            batch_idx: int,
            batch: list[dict[str, Any]],
        ) -> tuple[int, int]:
            session_id = str(session.get("session_id") or "episode_0")
            metadata = session.get("metadata", {}) or {}
            turn_datetime = metadata.get("session_time", "")
            speaker_a = metadata.get("speaker_a", "agent")
            speaker_b = metadata.get("speaker_b", "environment")
            current_text = ""
            current_turns_list = []
            batch_turn_ids = []
            for turn in batch:
                turn_text = turn.get("text", "")
                turn_speaker = turn.get("speaker", "Unknown")
                turn_speak_to = speaker_b if turn_speaker == speaker_a else speaker_a
                formatted_turn = f"{turn_speaker} speak to {turn_speak_to} at {turn_datetime}: {turn_text}"
                current_text += formatted_turn + "\n"
                current_turns_list.append(formatted_turn)
                batch_turn_ids.append(turn.get("turn_id", ""))

            batch_id = f"{session_id}_batch_{batch_idx}"
            builder.vector_db.add(
                c_turns,
                ids=[batch_id],
                documents=[current_text],
                metadatas=[{
                    "id": batch_id,
                    "col_name": c_turns,
                    "session_id": session_id,
                    "turn_time": turn_datetime,
                    "source_turn_ids": [batch_turn_ids],
                    "original_turns": current_turns_list,
                }],
            )

            formation_messages = builder.formation.construct_prompt(current_text, "", "")
            formation_response = builder.formation.llm_executor.get_completion(formation_messages)
            facts_added = 0
            experiences_added = 0
            for tc in builder._parse_tool_calls(formation_response):
                result_obj = builder.formation.execute_tool(tc["name"], tc["arguments"])
                if not result_obj or result_obj.get("type") not in {"fact", "experience"}:
                    continue
                col_name = c_facts if result_obj["type"] == "fact" else c_exp
                mem_id = f"{session_id}_b{batch_idx}_{uuid.uuid4().hex[:8]}"
                builder.vector_db.add(
                    col_name,
                    ids=[mem_id],
                    documents=[result_obj.get("document", "")],
                    metadatas=[{
                        "id": mem_id,
                        "col_name": col_name,
                        "session_id": session_id,
                        "turn_time": turn_datetime,
                        "source_turn_ids": batch_turn_ids,
                        "original_text": current_text,
                        "original_turns": current_turns_list,
                        "memory_content": result_obj.get("document", ""),
                    }],
                )
                if result_obj["type"] == "fact":
                    facts_added += 1
                else:
                    experiences_added += 1
            return facts_added, experiences_added

        if not jobs:
            return
        workers = min(self.large_corpus_parallel, len(jobs))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_batch, session, batch_idx, batch) for session, batch_idx, batch in jobs]
            for future in as_completed(futures):
                future.result()

    def memory_retrieve(self, memory: MemTMemory, question: str) -> str:
        """Mem-T answers directly via retrieve_and_answer."""
        with memory.lock:
            with _memt_env(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                embedding_model=self.embedding_model,
                embedding_base_url=self.embedding_base_url,
                embedding_api_key=self.embedding_api_key,
            ):
                result = memory.engine.retrieve_and_answer(
                    question,
                    sample_id=memory.sample_id,
                    category=memory.category,
                )

        if isinstance(result, dict):
            answer = result.get("answer", "")
            traces = result.get("traces") or []
        else:
            answer = getattr(result, "answer", "")
            traces = getattr(result, "traces", []) or []
        self.last_trace = list(traces) if isinstance(traces, list) else []
        self.last_route = {
            "tool_trace": [_trace_tool_name(item) for item in self.last_trace],
            "chosen_class": _trace_to_rq2_class(self.last_trace),
        }
        raw_context = self._raw_exact_context(memory, question, answer)
        if raw_context.startswith("# Raw MemT Exact Evidence"):
            return raw_context
        if str(memory.category or "").upper() == "TTL":
            ttl_context = _ttl_label_bank_context(memory.raw_chunks, question)
            if ttl_context:
                return ttl_context
        if str(memory.category or "").upper() == "LRU" and _env_bool("MEMT_LRU_RAW_FALLBACK", default=True):
            lru_context = _lru_raw_context(memory.raw_chunks, question)
            if lru_context:
                return lru_context
        if raw_context and not str(answer or "").strip():
            return raw_context
        return f"{MEMT_ANSWER_PREFIX}{answer}"

    def last_trajectory(self) -> Optional[dict]:
        return self._last_trajectory

    def _raw_exact_context(self, memory: MemTMemory, question: str, answer: str) -> str:
        force = os.environ.get("MEMT_INCLUDE_RAW_TOPK")
        is_exact_category = str(memory.category or "").upper() in {"TTL", "CR"}
        if not force and not is_exact_category:
            return ""
        try:
            top_k = int(force or os.environ.get("MEMT_EXACT_RAW_TOPK", "8"))
        except ValueError:
            top_k = 8
        exact_segments = _matching_label_segments(memory.raw_chunks, question)
        if exact_segments:
            try:
                max_segments = int(os.environ.get("MEMT_EXACT_RAW_SEGMENTS", "3"))
            except ValueError:
                max_segments = 3
            parts = [
                f"[raw_exact rank={rank} source_turn={idx} score={score:.3f}]\n{segment}"
                for rank, (idx, segment, score) in enumerate(exact_segments[:max(1, max_segments)], start=1)
            ]
            return "# Raw MemT Exact Evidence For Label Answering\n" + "\n\n".join(parts)
        ranked = _rank_raw_chunks(memory.raw_chunks, question)
        parts = []
        for rank, (idx, chunk, score) in enumerate(ranked[:top_k], start=1):
            parts.append(f"[raw_chunk rank={rank} source_turn={idx} score={score:.3f}]\n{chunk}")
        if not parts:
            return ""
        return "# Raw MemT Evidence Preserved For Exact Answering\n" + "\n\n".join(parts)

def _traj_text_to_sample(traj_text: str, task: str, sample_id: str) -> dict:
    """Convert AMABench trajectory text into Mem-T sample format."""
    turns = []
    current_turn_idx = 0

    for line in traj_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("Turn ", "Step ")):

            try:
                idx_label = stripped.split(":")[0].strip()
                idx_str = idx_label.replace("Turn ", "").replace("Step ", "").strip()
                current_turn_idx = int(idx_str)
            except (ValueError, IndexError):
                pass
        elif stripped.startswith("Action:"):
            action = stripped[len("Action:"):].strip()
            turns.append({
                "turn_id": f"turn_{current_turn_idx}_action",
                "speaker": "agent",
                "text": f"[Turn {current_turn_idx}] Action: {action}",
            })
        elif stripped.startswith("Observation:"):
            obs = stripped[len("Observation:"):].strip()
            turns.append({
                "turn_id": f"turn_{current_turn_idx}_obs",
                "speaker": "environment",
                "text": f"[Turn {current_turn_idx}] Observation: {obs}",
            })

    if not turns:
        for idx, chunk in enumerate(_raw_memory_chunks(traj_text)):
            turns.append({
                "turn_id": f"turn_{idx}",
                "speaker": "environment",
                "text": f"[Turn {idx}] Observation: {chunk}",
            })

    metadata = {
        "sample_id": sample_id,
        "speaker_a": "agent",
        "speaker_b": "environment",
        "session_time": "",
    }
    if task:
        metadata["task"] = task

    return {
        "sample_id": sample_id,
        "conversation": [{
            "session_id": "episode_0",
            "session_turns": turns,
            "metadata": metadata,
        }],
    }

def _category_from_task(task: str) -> str:
    for line in str(task or "").splitlines():
        if line.lower().startswith("category:"):
            return line.split(":", 1)[1].strip().upper()
    return ""

def _raw_memory_chunks(text: str, max_chars: int | None = None) -> list[str]:
    if max_chars is None:
        max_chars = int(os.environ.get("MEMT_RAW_CHUNK_CHARS", "4000"))
    max_chars = max(512, int(max_chars))
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

def _raw_exact_records(text: str) -> list[str]:
    """Keep MemoryAgentBench question/label records small enough for exact TTL/CR retrieval."""
    records: list[str] = []
    pending: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            if pending:
                records.append("\n".join(pending))
                pending = []
            continue
        if pending and _memt_label_like_line(line):
            pending.append(line)
            records.append("\n".join(pending))
            pending = []
            continue
        if pending:
            records.append("\n".join(pending))
        pending = [line]
    if pending:
        records.append("\n".join(pending))
    paired = [record for record in records if "\n" in record or re.search(r"\blabel\s*:", record, flags=re.I)]
    return paired if len(paired) >= 10 else []

def _memt_label_like_line(line: str) -> bool:
    line = str(line or "").strip()
    if not line:
        return False
    if re.match(r"^(label|answer|output|target|gold|class|result)\s*[:=]", line, flags=re.I):
        return True
    if len(line) <= 80 and not line.endswith("?") and re.fullmatch(r"[A-Za-z0-9_.:/,+\\-\\s%$#()]+", line):
        return True
    return False

_QUERY_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "the", "and", "for", "with", "about", "that", "this", "there", "their",
    "does", "did", "are", "was", "were", "is", "can", "could", "would",
    "should", "much", "many", "name", "tell", "give", "need", "your",
}

def _content_terms(text: str) -> list[str]:
    return [
        term for term in re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())
        if len(term) > 2 and term not in _QUERY_STOPWORDS and term != "label"
    ]

def _rank_raw_chunks(chunks: list[str], question: str) -> list[tuple[int, str, float]]:
    terms = _content_terms(question)
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

def _lru_raw_context(chunks: list[str], question: str) -> str:
    """Return bounded raw evidence for long-range understanding tasks."""
    if not chunks:
        return ""
    try:
        max_chars = int(os.environ.get("MEMT_LRU_RAW_MAX_CHARS", "24000"))
    except ValueError:
        max_chars = 24000
    try:
        top_k = int(os.environ.get("MEMT_LRU_RAW_TOPK", "10"))
    except ValueError:
        top_k = 10
    max_chars = max(4096, max_chars)
    top_k = max(1, top_k)

    ranked = _rank_raw_chunks(chunks, question)
    if ranked and ranked[0][2] > 0:
        selected = sorted(ranked[:top_k], key=lambda row: row[0])
    else:
        selected = _spread_chunks(chunks, top_k)

    parts = [
        "# Raw MemT Long-Range Evidence",
        "Use the memorized context below to answer the long-range understanding question. Follow the question's output format exactly.",
        f"Question: {question}",
    ]
    for idx, chunk, score in selected:
        candidate = f"[lru_chunk source_turn={idx} score={score:.3f}]\n{chunk}"
        if sum(len(part) + 2 for part in parts) + len(candidate) + 2 > max_chars:
            remaining = max_chars - sum(len(part) + 2 for part in parts) - len(f"[lru_chunk source_turn={idx} score={score:.3f}]\n") - 4
            if remaining > 512:
                parts.append(f"[lru_chunk source_turn={idx} score={score:.3f}]\n{chunk[:remaining].strip()}")
            break
        parts.append(candidate)
    return "\n\n".join(parts) if len(parts) > 3 else ""

def _spread_chunks(chunks: list[str], top_k: int) -> list[tuple[int, str, float]]:
    if len(chunks) <= top_k:
        return [(idx, chunk, 0.0) for idx, chunk in enumerate(chunks)]
    if top_k == 1:
        indices = [0]
    else:
        last = len(chunks) - 1
        indices = sorted({round(i * last / (top_k - 1)) for i in range(top_k)})
    return [(idx, chunks[idx], 0.0) for idx in indices]

def _ttl_label_bank_context(chunks: list[str], question: str) -> str:
    """Build a compact in-context label bank for MemoryAgentBench TTL tasks."""
    try:
        top_k = int(os.environ.get("MEMT_TTL_LABEL_BANK_TOPK", "24"))
    except ValueError:
        top_k = 24
    try:
        per_label = int(os.environ.get("MEMT_TTL_LABEL_BANK_PER_LABEL", "2"))
    except ValueError:
        per_label = 2
    try:
        max_chars = int(os.environ.get("MEMT_TTL_LABEL_BANK_MAX_CHARS", "24000"))
    except ValueError:
        max_chars = 24000

    records: list[tuple[int, str, str]] = []
    for idx, chunk in enumerate(chunks):
        match = re.search(r"\blabel:\s*([^\s]+)", chunk, flags=re.I)
        if not match:
            continue
        records.append((idx, match.group(1), chunk.strip()))
    if not records:
        return ""

    selected: dict[int, str] = {}
    for idx, _, record, _score in _rank_ttl_records(records, question)[: max(1, top_k)]:
        selected[idx] = record

    by_label: dict[str, list[tuple[int, str, float]]] = {}
    for idx, label, record, score in _rank_ttl_records(records, question):
        bucket = by_label.setdefault(label, [])
        if len(bucket) < max(1, per_label):
            bucket.append((idx, record, score))
    for label in sorted(by_label, key=lambda item: (len(item), item)):
        for idx, record, _score in by_label[label]:
            selected.setdefault(idx, record)

    parts = [
        "# Raw MemT TTL Label Bank",
        "Infer the requested learned label from these MemoryAgentBench examples. Return only the label value.",
        f"Question: {question}",
    ]
    for idx, record in sorted(selected.items()):
        candidate = f"[ttl_example source_turn={idx}]\n{record}"
        if sum(len(part) + 2 for part in parts) + len(candidate) + 2 > max_chars:
            break
        parts.append(candidate)
    return "\n\n".join(parts)

def _rank_ttl_records(records: list[tuple[int, str, str]], question: str) -> list[tuple[int, str, str, float]]:
    terms = _content_terms(question)
    ranked: list[tuple[int, str, str, float]] = []
    for idx, label, record in records:
        lower = record.lower()
        score = 0.0
        for term in terms:
            if term in lower:
                score += 1.0 + min(5, lower.count(term)) / 5.0
        ranked.append((idx, label, record, score))
    return sorted(ranked, key=lambda row: (row[3], -row[0]), reverse=True)

def _matching_label_segments(chunks: list[str], question: str) -> list[tuple[int, str, float]]:
    q_terms = _content_terms(question)
    if not q_terms:
        return []
    question_norm = " ".join(q_terms)
    matches: list[tuple[int, str, float]] = []
    for idx, chunk in enumerate(chunks):
        segments = _split_label_segments(chunk)
        for segment in segments:
            if not segment:
                continue
            segment = re.sub(r"^\[Turn\s+\d+\]\s+Observation:\s*", "", segment).strip()
            segment_terms = _content_terms(segment)
            if not segment_terms:
                continue
            segment_norm = " ".join(segment_terms)
            overlap = sum(1 for term in q_terms if term in segment_terms)
            coverage = overlap / max(1, len(q_terms))
            substring_bonus = 2.0 if question_norm in segment_norm else 0.0
            ordered_bonus = 0.0
            if len(q_terms) >= 2:
                for left, right in zip(q_terms, q_terms[1:]):
                    if f"{left} {right}" in segment_norm:
                        ordered_bonus += 0.2
            if coverage >= 0.6 or (overlap >= 2 and len(q_terms) <= 4) or substring_bonus:
                matches.append((idx, segment, coverage + substring_bonus + ordered_bonus))
    return sorted(matches, key=lambda row: row[2], reverse=True)

def _split_label_segments(chunk: str) -> list[str]:
    records = _raw_exact_records(str(chunk or ""))
    if records:
        return records
    segments: list[str] = []
    start = 0
    for label_match in re.finditer(r"\blabel:\s*[^\s]+", str(chunk or ""), flags=re.I):
        segment = str(chunk or "")[start:label_match.end()].strip()
        start = label_match.end()
        if segment:
            segments.append(segment)
    return segments

def _trace_tool_name(item: Any) -> str:
    if isinstance(item, dict):
        call = item.get("tool_call") or {}
        if isinstance(call, dict):
            name = call.get("name")
            if name:
                return str(name).strip().lower()
    return ""

def _trace_to_rq2_class(trace: list[dict[str, Any]]) -> str:
    tools = [_trace_tool_name(item) for item in trace]
    tools = [tool for tool in tools if tool]
    if not tools or tools[0] == "finish":
        return "none"
    if "search_experiences" in tools:
        return "procedural"
    semantic_tools = {"search_turns", "search_facts", "search_personas", "search_summary"}
    if any(tool in semantic_tools for tool in tools):
        return "semantic"
    return "semantic"
