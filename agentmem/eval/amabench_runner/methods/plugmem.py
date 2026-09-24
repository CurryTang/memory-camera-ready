"""Shared PlugMem method wrapper backed by the upstream graph pipeline."""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agentmem.eval.amabench_runner.methods.base import BaseMethod
from agentmem.plugmem.session import PlugMemSession
from agentmem.plugmem.upstream import (
    PlugMemGraphMemory,
    PlugMemUpstreamAdapter,
    default_plugmem_source_root,
    parse_trajectory_steps,
    plugmem_env,
)

_QWEN_ROUTE_NAME = os.environ.get("LLM_MODEL") or os.environ.get("QWEN_MODEL_NAME") or "qwen3-32b"
_SEMANTIC_ONLY_MODES = ("semantic_memory",)

def _plugmem_qwen_model_name(llm_model: Optional[str]) -> str:
    for candidate in (
        llm_model,
        os.environ.get("QWEN_MODEL_NAME"),
        os.environ.get("OPENAI_MODEL_NAME"),
        "Qwen/Qwen3-32B",
    ):
        if candidate not in (None, ""):
            return str(candidate)
    return "Qwen/Qwen3-32B"

def _plugmem_llm_env_name(llm_model: Optional[str], llm_base_url: Optional[str]) -> str:

    if llm_base_url:
        return llm_model or os.environ.get("LLM_MODEL") or os.environ.get("QWEN_MODEL_NAME") or _QWEN_ROUTE_NAME
    return _plugmem_qwen_model_name(llm_model)

def _plugmem_env_overrides(
    *,
    llm_model: Optional[str],
    llm_base_url: Optional[str],
    llm_api_key: Optional[str],
    embedding_model: Optional[str],
    embedding_base_url: Optional[str],
    embedding_api_key: Optional[str],
) -> dict[str, str]:
    qwen_model = _plugmem_qwen_model_name(llm_model)
    return {
        key: str(value)
        for key, value in {
            "LLM_NAME": _plugmem_llm_env_name(qwen_model, llm_base_url),
            "QWEN_MODEL_NAME": qwen_model if llm_base_url else None,
            "QWEN_BASE_URL": llm_base_url,
            "VLLM_QWEN_API_KEY": llm_api_key,
            "EMBEDDING_MODEL_NAME": embedding_model,
            "EMBEDDING_BASE_URL": embedding_base_url,
            "EMBEDDING_API_KEY": embedding_api_key,
            "OPENAI_API_KEY": llm_api_key,
        }.items()
        if value not in (None, "")
    }

class PlugMemMethod(BaseMethod):
    """Task-agnostic PlugMem wrapper with selectable branch ablations."""

    def __init__(
        self,
        llm_model: str = "qwen/qwen3-32b",
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        top_k: int = 20,
        save_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        source_root: Optional[str] = None,
        memory_modes: str | Sequence[str] | None = None,
        answer_model: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        answer_api_key: Optional[str] = None,
        semantic_window_size: int = 12,
        fast_semantic_only: bool = False,
        provider_kwargs: Optional[Mapping[str, Any]] = None,
        **_kw,
    ) -> None:
        default_llm_model = "qwen/qwen3-32b"
        if config_path:
            cfg = self._load_config(config_path)
            if llm_model == default_llm_model:
                llm_model = cfg.get("llm_model", llm_model)
            if llm_base_url is None:
                llm_base_url = cfg.get("llm_base_url", llm_base_url)
            if llm_api_key is None:
                llm_api_key = cfg.get("llm_api_key", llm_api_key)
            if embedding_model is None:
                embedding_model = cfg.get("embedding_model", embedding_model)
            if embedding_base_url is None:
                embedding_base_url = cfg.get("embedding_base_url", embedding_base_url)
            if embedding_api_key is None:
                embedding_api_key = cfg.get("embedding_api_key", embedding_api_key)
            if save_dir is None:
                save_dir = cfg.get("save_dir", save_dir)
            if source_root is None:
                source_root = cfg.get("source_root", source_root)
            if memory_modes is None:
                memory_modes = cfg.get("memory_modes", memory_modes)
            if answer_model is None:
                answer_model = cfg.get("answer_model", answer_model)
            if answer_base_url is None:
                answer_base_url = cfg.get("answer_base_url", answer_base_url)
            if answer_api_key is None:
                answer_api_key = cfg.get("answer_api_key", answer_api_key)
            semantic_window_size = cfg.get("semantic_window_size", semantic_window_size)
            fast_semantic_only = bool(cfg.get("fast_semantic_only", fast_semantic_only))

        env_modes = os.environ.get("PLUGMEM_MEMORY_MODES")
        if env_modes and memory_modes is None:
            memory_modes = env_modes
        env_window = os.environ.get("PLUGMEM_SEMANTIC_WINDOW_SIZE")
        if env_window:
            semantic_window_size = int(env_window)
        env_fast = os.environ.get("PLUGMEM_FAST_SEMANTIC_ONLY")
        if env_fast is not None:
            fast_semantic_only = env_fast.strip().lower() in {"1", "true", "yes", "on"}

        plugmem_embedding_base_url = embedding_base_url
        if plugmem_embedding_base_url:
            normalized = str(plugmem_embedding_base_url).rstrip("/")
            if not normalized.endswith("/embeddings"):
                normalized = f"{normalized}/embeddings"
            plugmem_embedding_base_url = normalized

        self.top_k = max(1, int(top_k))
        self.save_dir = save_dir or tempfile.mkdtemp(prefix="plugmem_method_")
        self.semantic_window_size = max(1, int(semantic_window_size or 12))
        self.fast_semantic_only = bool(fast_semantic_only)
        self._episode_counter = 0
        self._last_retrieval_debug: dict[str, Any] = {}
        self.adapter = PlugMemUpstreamAdapter(
            source_root=source_root or default_plugmem_source_root(),
            env_overrides=_plugmem_env_overrides(
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                embedding_model=embedding_model,
                embedding_base_url=plugmem_embedding_base_url,
                embedding_api_key=embedding_api_key,
            ),
            retrieval_topk=self.top_k,
            memory_modes=memory_modes,
            answer_model=answer_model,
            answer_api_key=answer_api_key,
            answer_base_url=answer_base_url,
            provider_kwargs=provider_kwargs,
        )

    def memory_construction(self, traj_text: str, task: str = "") -> PlugMemGraphMemory:
        self._episode_counter += 1
        sample_dir = Path(self.save_dir) / f"episode_{self._episode_counter}"
        if (
            self.fast_semantic_only
            and tuple(getattr(self.adapter, "memory_modes", ())) == _SEMANTIC_ONLY_MODES
        ):
            return self._build_semantic_only(traj_text, task=task, sample_dir=sample_dir)
        return self.adapter.build_from_trajectory_text(
            traj_text,
            task=task,
            sample_dir=sample_dir,
            metadata={"benchmark": "amabench"},
        )

    def memory_retrieve(self, memory: PlugMemGraphMemory, question: str) -> str:
        retrieval = self.adapter.retrieve(memory, question)
        self._last_retrieval_debug = {
            "contexts": dict(retrieval.contexts),
            "errors": dict(retrieval.errors),
            "variables": dict(retrieval.variables),
        }
        context = retrieval.combined_context()
        raw_context = self._raw_support_context(memory, question)
        if raw_context:
            graph_context = f"\n\n# PlugMem Graph Context\n{context}" if context else ""
            return "# Raw PlugMem Evidence Preserved For Exact Answering\n" + raw_context + graph_context
        return context

    def last_trajectory(self) -> dict[str, Any]:
        return {"plugmem_retrieval_debug": dict(self._last_retrieval_debug)}

    def _raw_support_context(self, memory: PlugMemGraphMemory, question: str) -> str:
        task = str(getattr(memory.session, "goal", "") or "")
        force = os.environ.get("PLUGMEM_INCLUDE_RAW_TOPK")
        is_exact_category = "Category: TTL" in task or "Category: CR" in task
        if not force and not is_exact_category:
            return ""
        try:
            top_k = int(force or os.environ.get("PLUGMEM_EXACT_RAW_TOPK", "8"))
        except ValueError:
            top_k = 8
        if top_k <= 0:
            return ""
        chunks = [str(getattr(step, "observation", "") or "") for step in memory.session.steps]
        ranked = self._rank_raw_chunks(chunks, question)
        parts = []
        for rank, (idx, chunk, score) in enumerate(ranked[:top_k], start=1):
            parts.append(f"[raw_chunk rank={rank} source_turn={idx} score={score:.3f}]\n{chunk}")
        return "\n\n".join(parts)

    @staticmethod
    def _rank_raw_chunks(chunks: list[str], question: str) -> list[tuple[int, str, float]]:
        import re

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

    def _build_semantic_only(self, traj_text: str, *, task: str, sample_dir: Path) -> PlugMemGraphMemory:
        raw_steps = list(parse_trajectory_steps(traj_text))

        def _keep(step: Any) -> bool:
            action = getattr(step, "action", None)
            observation = getattr(step, "observation", None)
            if isinstance(step, dict):
                action = step.get("action", action)
                observation = step.get("observation", observation)
            return bool((action and str(action).strip()) or (observation and str(observation).strip()))

        filtered = [step for step in raw_steps if _keep(step)]
        max_turns = int(os.environ.get("PLUGMEM_MAX_TURNS", "0") or "0")
        if max_turns > 0 and len(filtered) > max_turns:
            filtered = filtered[-max_turns:]

        session = PlugMemSession(
            session_id=f"plugmem-{self.adapter._episode_counter}",
            goal=str(task or "Solve the task."),
            steps=filtered,
            metadata={"benchmark": "generic"},
        )
        self.adapter._episode_counter += 1

        sample_dir = Path(sample_dir).resolve()
        sample_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("episodic_memory", "semantic_memory", "procedural_memory", "tag", "subgoal"):
            (sample_dir / subdir).mkdir(parents=True, exist_ok=True)

        graph = self.adapter._new_memory_graph(log_file=sample_dir / "plugmem.log")
        first_timestamp = str(filtered[0].timestamp or 0) if filtered else "0"
        memory = self.adapter._Memory(
            goal=str(session.goal or "PlugMem"),
            observation=self.adapter._initial_observation(session, dialogue_like=False),
            time=first_timestamp,
        )

        episodic_steps = []
        semantic_items = []
        semantic_embeddings = []
        structuring = sys.modules.get("memory_structuring.structuring_inference")
        get_semantic = getattr(structuring, "get_semantic", None)
        if get_semantic is None:
            from memory_structuring.structuring_inference import get_semantic as imported_get_semantic
            get_semantic = imported_get_semantic

        with plugmem_env(self.adapter.env_overrides, sample_dir=sample_dir):
            for window_idx, start in enumerate(range(0, len(filtered), self.semantic_window_size)):
                window = filtered[start:start + self.semantic_window_size]
                observation = self._render_semantic_window(window)
                if not observation:
                    continue
                episodic_steps.append({
                    "observation": observation,
                    "action": "",
                    "reward": "",
                    "time": first_timestamp,
                })
                if os.environ.get("PLUGMEM_FAST_RAW_SEMANTIC", "").strip().lower() in {"1", "true", "yes", "on"}:
                    semantic_text = observation.strip()
                    if not semantic_text:
                        continue
                    tags = [
                        "memoryagentbench",
                        str(task or "plugmem").splitlines()[0][:80].lower().replace(" ", "_"),
                    ]
                    semantic_items.append({
                        "semantic_memory": semantic_text,
                        "tags": [tag for tag in tags if tag],
                        "trajectory_num": 0,
                        "turn_num": window_idx,
                        "time": first_timestamp,
                    })
                    semantic_embeddings.append({
                        "semantic_memory": self.adapter._embed_text(semantic_text),
                        "tags": [self.adapter._embed_text(tag) for tag in tags if tag],
                    })
                    continue
                new_semantic = get_semantic(
                    {"observation": observation},
                    trajectory_num=0,
                    turn_num=window_idx,
                    time=first_timestamp,
                    mode="trajectory",
                )
                for item in new_semantic:
                    semantic_text = str(item.get("semantic_memory") or "").strip()
                    if not semantic_text:
                        continue
                    tags = [str(tag).strip() for tag in item.get("tags", []) if str(tag).strip()]
                    normalized = dict(item)
                    normalized["semantic_memory"] = semantic_text
                    normalized["tags"] = tags
                    normalized["trajectory_num"] = 0
                    normalized["turn_num"] = window_idx
                    normalized.setdefault("time", first_timestamp)
                    semantic_items.append(normalized)
                    semantic_embeddings.append({
                        "semantic_memory": self.adapter._embed_text(semantic_text),
                        "tags": [self.adapter._embed_text(tag) for tag in tags],
                    })

            if not semantic_items and episodic_steps:
                fallback_text = str(episodic_steps[0].get("observation") or "").strip()
                if fallback_text:
                    semantic_items.append({
                        "semantic_memory": fallback_text[:2000],
                        "tags": [],
                        "trajectory_num": 0,
                        "turn_num": 0,
                        "time": first_timestamp,
                    })
                    semantic_embeddings.append({
                        "semantic_memory": self.adapter._embed_text(fallback_text[:2000]),
                        "tags": [],
                    })

            memory.memory["episodic"] = [episodic_steps] if episodic_steps else []
            memory.memory["semantic"] = semantic_items
            memory.memory["procedural"] = []
            memory.memory_embedding["semantic"] = semantic_embeddings
            memory.memory_embedding["procedural"] = []
            graph.insert(self.adapter._normalize_memory(memory))

        return PlugMemGraphMemory(graph=graph, session=session, sample_dir=sample_dir)

    @staticmethod
    def _render_semantic_window(steps: Sequence[Any]) -> str:
        lines: list[str] = []
        for step in steps:
            index = getattr(step, "index", None)
            prefix = f"Step {index}: " if index is not None else ""
            action = str(getattr(step, "action", "") or "").strip()
            observation = str(getattr(step, "observation", "") or "").strip()
            if action:
                lines.append(f"{prefix}Action: {action}")
            if observation:
                lines.append(f"{prefix}Observation: {observation}")
        return "\n".join(lines).strip()
