"""Cross-episode memory backends for ALFWorld evaluation."""

from __future__ import annotations

import logging
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agentmem.backends.base import MemoryRecord
from agentmem.backends.episodic import EpisodicMemoryStore
from agentmem.memrl.runtime import MemRLConfig, MemRLRuntimeEngine
from agentmem.memrl.task_adaptation import (
    format_alfworld_memrl_prompt,
    normalize_alfworld_task_description,
)

logger = logging.getLogger(__name__)

PlugMemMethod: Any | None = None

@dataclass
class EpisodeMemoryRecord:
    episode_idx: int
    task_desc: str
    task_type: str
    success: bool
    memory: Any
    retrieval_preview: str = ""

class BaseAlfWorldMemoryBackend:
    """Shared cross-episode wrapper over per-trajectory memory methods."""

    name = "base"

    _MAX_CONTEXT_CHARS_PER_EPISODE = 3000
    _MAX_TOTAL_MEMORY_CHARS = 8000

    def __init__(self, method: Any, top_k: int = 3) -> None:
        self.method = method
        self.top_k = max(1, int(top_k))
        self.records: list[EpisodeMemoryRecord] = []

    def get_memory_prompt(self, task_desc: str, task_type: str = "") -> str:
        query = str(task_desc or "").strip()
        if not query:
            return ""

        scored: list[tuple[float, EpisodeMemoryRecord, str]] = []
        for record in self.records:
            try:
                context = str(self.method.memory_retrieve(record.memory, query) or "").strip()
            except Exception as exc:
                logger.warning("%s retrieve failed for episode %d: %s", self.name, record.episode_idx, exc)
                continue
            if not context:
                continue
            if len(context) > self._MAX_CONTEXT_CHARS_PER_EPISODE:
                context = context[: self._MAX_CONTEXT_CHARS_PER_EPISODE - 15].rstrip() + "... [truncated]"
            score = self._score_record(query=query, task_type=task_type, record=record, context=context)
            scored.append((score, record, context))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: self.top_k]
        parts = [
            f"Memory system: {self.memory_summary()}",
            self.usage_guidance(),
            f"Relevant memories from past ALFWorld episodes ({self.name}):",
        ]
        for _, record, context in selected:
            header = (
                f"Episode {record.episode_idx} | task_type={record.task_type or 'generic'} "
                f"| success={record.success}"
            )
            parts.append(f"{header}\nTask: {record.task_desc}\n{context}")
        result = "\n\n".join(parts)
        if len(result) > self._MAX_TOTAL_MEMORY_CHARS:
            result = result[: self._MAX_TOTAL_MEMORY_CHARS - 15].rstrip() + "... [truncated]"
        return result

    def update_from_episode(
        self,
        episode_idx: int,
        task_desc: str,
        task_type: str,
        success: bool,
        trajectory: list[dict[str, Any]],
    ) -> None:
        traj_text = format_trajectory_for_memory(trajectory)
        if not traj_text.strip():
            return

        try:
            memory = self.method.memory_construction(traj_text, task=task_desc)
        except Exception as exc:
            logger.warning("%s construction failed for episode %d: %s", self.name, episode_idx, exc)
            return

        record = EpisodeMemoryRecord(
            episode_idx=episode_idx,
            task_desc=str(task_desc or "").strip(),
            task_type=str(task_type or "").strip(),
            success=bool(success),
            memory=memory,
            retrieval_preview=traj_text[:400],
        )
        self.records.append(record)

    def _score_record(
        self,
        query: str,
        task_type: str,
        record: EpisodeMemoryRecord,
        context: str,
    ) -> float:
        task_bonus = 0.25 if task_type and task_type == record.task_type else 0.0
        success_bonus = 0.15 if record.success else 0.0
        query_tokens = _normalize_tokens(query)
        record_tokens = _normalize_tokens(f"{record.task_desc}\n{context}")
        overlap = _jaccard(query_tokens, record_tokens)
        return overlap + task_bonus + success_bonus - 0.001 * max(0, len(context) - 2000)

    def memory_summary(self) -> str:
        return "retrieved notes from past ALFWorld episodes"

    def usage_guidance(self) -> str:
        return (
            "How to use this memory: extract reusable strategy, search order, or failure avoidance; "
            "adapt it to the current room and object names; trust the current observation and admissible "
            "actions over memory if they conflict."
        )

class HippoRAGv2MemoryBackend(BaseAlfWorldMemoryBackend):
    name = "hipporagv2"

    def __init__(self, top_k: int = 3, save_dir: Optional[str] = None, **kwargs: Any) -> None:
        from agentmem.eval.amabench_runner.methods.hipporag import HippoRAGMethod

        if save_dir:
            kwargs.setdefault("save_dir", str(save_dir))
        super().__init__(HippoRAGMethod(**kwargs), top_k=top_k)

    def memory_summary(self) -> str:
        return "graph-linked retrieved passages from past episodes"

    def usage_guidance(self) -> str:
        return (
            "How to use HippoRAGv2 memory: look for linked subgoals, likely object/receptacle locations, "
            "and action sequences that solved similar tasks; use the graph-linked evidence to plan the next "
            "few steps, but do not assume old object ids or room layouts still hold exactly."
        )

class SimpleMemMemoryBackend(BaseAlfWorldMemoryBackend):
    name = "simplemem"

    def __init__(self, top_k: int = 3, **kwargs: Any) -> None:
        from agentmem.eval.amabench_runner.methods.simplemem import SimpleMemMethod

        super().__init__(SimpleMemMethod(**kwargs), top_k=top_k)

    def memory_summary(self) -> str:
        return "distilled atomic memories and retrieved episode snippets"

    def usage_guidance(self) -> str:
        return (
            "How to use SimpleMem memory: extract concise facts, useful search heuristics, and mistakes to avoid; "
            "convert them into the next concrete action for the current task instead of copying them literally; "
            "prefer high-level tactics over exact entity matches."
        )

class DCILiteMemoryBackend(BaseAlfWorldMemoryBackend):
    name = "dci_lite"

    def __init__(self, top_k: int = 3, context_level: str = "level3", **kwargs: Any) -> None:
        from agentmem.methods import build_method

        kwargs.setdefault("context_level", context_level)
        kwargs.setdefault("max_context_chars", 8000)
        super().__init__(build_method("dci_lite", **kwargs), top_k=top_k)

    def memory_summary(self) -> str:
        return "direct-corpus interaction over prior episode traces"

    def usage_guidance(self) -> str:
        return (
            "How to use DCI memory: treat prior episodes as a raw searchable corpus; reuse only grounded "
            "search/action patterns and failure fixes, then verify every next step against the current "
            "observation and admissible actions."
        )

class DCILiteSummarizeMemoryBackend(DCILiteMemoryBackend):
    name = "dci_lite_sum"

    def __init__(self, top_k: int = 3, **kwargs: Any) -> None:
        super().__init__(top_k=top_k, context_level="level4", **kwargs)

class LightMemMemoryBackend(BaseAlfWorldMemoryBackend):
    name = "lightmem"

    def __init__(self, top_k: int = 3, **kwargs: Any) -> None:
        from agentmem.eval.amabench_runner.methods.lightmem import LightMemMethod

        super().__init__(LightMemMethod(**kwargs), top_k=top_k)

    def memory_summary(self) -> str:
        return "compressed topic summaries and retrieved episode snippets"

    def usage_guidance(self) -> str:
        return (
            "How to use LightMem memory: extract concise facts, search heuristics, and mistakes to avoid from the "
            "compressed summaries; convert them into the next concrete action for the current task instead of "
            "copying them literally; prefer stable tactics and state updates over exact stale entity matches."
        )

class AMAAgentMemoryBackend(BaseAlfWorldMemoryBackend):
    name = "ama_agent"

    _MAX_CONTEXT_CHARS_PER_EPISODE = 2000
    _MAX_TOTAL_MEMORY_CHARS = 6000

    def __init__(self, top_k: int = 3, save_dir: Optional[str] = None, **kwargs: Any) -> None:
        from agentmem.eval.amabench_runner.methods.ama_agent import AMAAgentMethod

        kwargs.setdefault("chunk_size", 1024)
        super().__init__(AMAAgentMethod(**kwargs), top_k=top_k)
        self._index_save_dir: Optional[Path] = Path(save_dir) if save_dir else None
        if self._index_save_dir is not None:
            self._index_save_dir.mkdir(parents=True, exist_ok=True)
        self._episode_counter = 0

    def update_from_episode(
        self,
        episode_idx: int,
        task_desc: str,
        task_type: str,
        success: bool,
        trajectory: list[dict[str, Any]],
    ) -> None:
        prior_len = len(self.records)
        super().update_from_episode(
            episode_idx=episode_idx,
            task_desc=task_desc,
            task_type=task_type,
            success=success,
            trajectory=trajectory,
        )
        if self._index_save_dir is None or len(self.records) == prior_len:
            return
        self._episode_counter += 1
        episode_dir = self._index_save_dir / f"episode_{self._episode_counter}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        latest = self.records[-1]
        try:
            with (episode_dir / "state.pkl").open("wb") as handle:
                pickle.dump(latest.memory, handle)
        except Exception as exc:
            logger.warning(
                "%s failed to dump index for episode %d: %s",
                self.name, episode_idx, exc,
            )
            return
        meta = {
            "episode_idx": int(episode_idx),
            "task_desc": latest.task_desc,
            "task_type": latest.task_type,
            "success": bool(latest.success),
            "trajectory_preview": latest.retrieval_preview,
        }
        (episode_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def memory_summary(self) -> str:
        return "structured agent memories with task-level causal hints"

    def usage_guidance(self) -> str:
        return (
            "How to use AMA-style memory: identify the latent subgoal and the causal dependencies between steps, "
            "then use the retrieved plan pattern to decide what must happen before the current goal can succeed; "
            "keep the action grounded in the current observation."
        )

    def get_memory_prompt(self, task_desc: str, task_type: str = "") -> str:
        query = str(task_desc or "").strip()
        if not query:
            return ""

        scored: list[tuple[float, EpisodeMemoryRecord, str]] = []
        for record in self.records:
            try:
                context = str(self.method.memory_retrieve(record.memory, query) or "").strip()
            except Exception as exc:
                logger.warning("%s retrieve failed for episode %d: %s", self.name, record.episode_idx, exc)
                continue
            if not context:
                continue

            if len(context) > self._MAX_CONTEXT_CHARS_PER_EPISODE:
                context = context[: self._MAX_CONTEXT_CHARS_PER_EPISODE - 15].rstrip() + "... [truncated]"
            score = self._score_record(query=query, task_type=task_type, record=record, context=context)
            scored.append((score, record, context))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: self.top_k]
        parts = [
            f"Memory system: {self.memory_summary()}",
            self.usage_guidance(),
            f"Relevant memories from past ALFWorld episodes ({self.name}):",
        ]
        for _, record, context in selected:
            header = (
                f"Episode {record.episode_idx} | task_type={record.task_type or 'generic'} "
                f"| success={record.success}"
            )
            parts.append(f"{header}\nTask: {record.task_desc}\n{context}")

        result = "\n\n".join(parts)
        if len(result) > self._MAX_TOTAL_MEMORY_CHARS:
            result = result[: self._MAX_TOTAL_MEMORY_CHARS - 15].rstrip() + "... [truncated]"
        return result

class MemTMemoryBackend(BaseAlfWorldMemoryBackend):
    name = "memt"

    _COLLECTION_ORDER = ("experiences", "facts", "summary", "turns", "personas")

    def __init__(self, top_k: int = 3, **kwargs: Any) -> None:
        from agentmem.eval.amabench_runner.methods.memt import MemTMethod

        super().__init__(MemTMethod(**kwargs), top_k=top_k)

    def memory_summary(self) -> str:
        return "multi-collection memories retrieved from past episodes"

    def usage_guidance(self) -> str:
        return (
            "How to use Mem-T memory: treat experiences as reusable procedures, facts as object-location cues, "
            "and summaries as compressed task plans; combine them into the next grounded action instead of "
            "copying stale entities or room layouts literally."
        )

    def get_memory_prompt(self, task_desc: str, task_type: str = "") -> str:
        query = str(task_desc or "").strip()
        if not query:
            return ""

        scored: list[tuple[float, EpisodeMemoryRecord, str]] = []
        for record in self.records:
            context = self._retrieve_record_context(record, query=query)
            if not context:
                continue
            score = self._score_record(query=query, task_type=task_type, record=record, context=context)
            scored.append((score, record, context))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: self.top_k]
        parts = [
            f"Memory system: {self.memory_summary()}",
            self.usage_guidance(),
            f"Relevant memories from past ALFWorld episodes ({self.name}):",
        ]
        for _, record, context in selected:
            header = (
                f"Episode {record.episode_idx} | task_type={record.task_type or 'generic'} "
                f"| success={record.success}"
            )
            parts.append(f"{header}\nTask: {record.task_desc}\n{context}")
        return "\n\n".join(parts)

    def _retrieve_record_context(self, record: EpisodeMemoryRecord, query: str) -> str:
        try:
            bank = record.memory.engine.get_bank(record.memory.sample_id)
        except Exception as exc:
            logger.warning("%s failed to access memory bank for episode %d: %s", self.name, record.episode_idx, exc)
            return ""

        snippets: list[str] = []
        seen: set[str] = set()
        per_collection_top_k = max(1, min(2, self.top_k))
        for collection in self._COLLECTION_ORDER:
            try:
                search_result = bank.search(collection, query=query, top_k=per_collection_top_k)
            except Exception as exc:
                logger.warning(
                    "%s search failed for episode %d collection %s: %s",
                    self.name,
                    record.episode_idx,
                    collection,
                    exc,
                )
                continue

            documents = search_result.get("documents", [])
            if documents and isinstance(documents[0], list):
                documents = documents[0]

            for document in documents or []:
                text = str(document or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                snippets.append(f"[{collection}] {text}")

        joined = "\n".join(snippets[: max(3, self.top_k * 2)])
        if len(joined) > 3000:
            joined = joined[:2985].rstrip() + "... [truncated]"
        return joined

class PlugMemMemoryBackend(BaseAlfWorldMemoryBackend):
    name = "plugmem"

    def __init__(self, top_k: int = 3, save_dir: Optional[str] = None, **kwargs: Any) -> None:
        method_cls = PlugMemMethod
        if method_cls is None:
            from agentmem.eval.amabench_runner.methods.plugmem import PlugMemMethod as method_cls

        if save_dir:
            kwargs.setdefault("save_dir", str(save_dir))
        super().__init__(method_cls(**kwargs), top_k=top_k)

    def memory_summary(self) -> str:
        return "procedural and semantic memory snippets distilled from past episodes"

    def usage_guidance(self) -> str:
        return (
            "How to use PlugMem memory: use procedural memory for step ordering and semantic memory for object or "
            "receptacle cues; combine them to choose the next action, but ignore stale details when the current "
            "environment state disagrees."
        )

class MemRLAlfWorldMemoryBackend(BaseAlfWorldMemoryBackend):
    name = "memrl"

    def __init__(self, top_k: int = 3, config_path: Optional[str] = None, **kwargs: Any) -> None:
        del kwargs
        self.top_k = max(1, int(top_k))
        self.records: list[EpisodeMemoryRecord] = []
        self.store = EpisodicMemoryStore()
        self._config = self._default_config()
        self.runtime = MemRLRuntimeEngine(store=self.store, config=self._config)
        self._pending_selected_ids: dict[str, list[str]] = {}
        if config_path:
            self.load_state(config_path)

    def _default_config(self) -> MemRLConfig:
        return MemRLConfig(
            phase1_topk=max(8, self.top_k * 3),
            topk=self.top_k,
            alpha=0.3,
            gamma=0.0,
            epsilon=0.0,
            similarity_weight=0.5,
            utility_weight=0.5,
            dedup_by_task_id=True,
        )

    def memory_summary(self) -> str:
        return "memrl-style episodic ALFWorld memory with utility-ranked retrieval"

    def usage_guidance(self) -> str:
        return (
            "How to use MemRL memory: extract reusable plan structure, object search order, and failure warnings; "
            "adapt them to the current room state and admissible commands instead of copying old actions literally."
        )

    def get_memory_prompt(self, task_desc: str, task_type: str = "") -> str:
        query = normalize_alfworld_task_description(task_desc, task_type)
        if not query:
            return ""
        retrieval = self.runtime.retrieve(query, topk=self.top_k)
        selected = list(retrieval.selected or [])
        self._pending_selected_ids[query] = [str(c.memory_id) for c in selected]

        successful: list[str] = []
        failed: list[str] = []
        for cand in selected:
            meta = dict(cand.metadata or {})
            content = str(meta.get("full_content") or cand.content or "").strip()
            if not content:
                continue

            if len(content) > self._MAX_CONTEXT_CHARS_PER_EPISODE:
                content = content[: self._MAX_CONTEXT_CHARS_PER_EPISODE - 15].rstrip() + "... [truncated]"
            header = (
                f"Task={meta.get('task_desc') or meta.get('intent') or query} | "
                f"task_type={meta.get('task_type') or task_type or 'generic'} | "
                f"utility={float(cand.utility):.3f}"
            )
            block = f"{header}\n{content}"
            if bool(meta.get("success", True)):
                successful.append(block)
            else:
                failed.append(block)
        result = format_alfworld_memrl_prompt(successful, failed)
        if len(result) > self._MAX_TOTAL_MEMORY_CHARS:
            result = result[: self._MAX_TOTAL_MEMORY_CHARS - 15].rstrip() + "... [truncated]"
        return result

    def update_from_episode(
        self,
        episode_idx: int,
        task_desc: str,
        task_type: str,
        success: bool,
        trajectory: list[dict[str, Any]],
    ) -> None:
        task_key = normalize_alfworld_task_description(task_desc, task_type)
        selected_ids = self._pending_selected_ids.pop(task_key, [])
        reward = 1.0 if success else -1.0
        if selected_ids:
            self.runtime.update_utilities(memory_ids=selected_ids, reward=reward)

        trajectory_text = format_trajectory_for_memory(trajectory)
        if not trajectory_text.strip():
            return

        experience = (
            f"Task: {task_desc}\n"
            f"Task type: {task_type or 'generic'}\n"
            f"Episode: {episode_idx}\n"
            f"Success: {bool(success)}\n\n"
            f"Trajectory:\n{trajectory_text}"
        )
        self.runtime.add_experience(
            intent=task_key,
            experience=experience,
            success=bool(success),
            metadata={
                "source": "alfworld_episode",
                "task_desc": str(task_desc or "").strip(),
                "task_type": str(task_type or "").strip(),
                "episode_idx": int(episode_idx),
                "success": bool(success),
                "full_content": experience,
            },
            retrieved_memory_ids=selected_ids,
            task_id=f"alfworld_episode_{episode_idx}",
        )
        self.records.append(
            EpisodeMemoryRecord(
                episode_idx=episode_idx,
                task_desc=str(task_desc or "").strip(),
                task_type=str(task_type or "").strip(),
                success=bool(success),
                memory=experience,
                retrieval_preview=trajectory_text[:400],
            )
        )

    def load_state(self, path: str | Path) -> None:
        """Load a train-only MemRL warmstart store produced by the holdout script."""
        state_path = Path(path)
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        config_payload = dict(payload.get("memrl_config") or {})
        merged_config = {
            **self._default_config().__dict__,
            **{k: v for k, v in config_payload.items() if hasattr(MemRLConfig, "__dataclass_fields__") and k in MemRLConfig.__dataclass_fields__},
        }
        self._config = MemRLConfig(**merged_config)
        self.store = EpisodicMemoryStore()
        self.records = []
        for idx, item in enumerate(payload.get("records") or []):
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            metadata = dict(item.get("metadata") or {})
            record = MemoryRecord(
                content=content,
                metadata=metadata,
                id=str(item.get("id") or f"memrl_holdout_{idx}"),
            )
            self.store.add(record)
            self.records.append(
                EpisodeMemoryRecord(
                    episode_idx=int(metadata.get("episode_idx", idx) or idx),
                    task_desc=str(metadata.get("task_desc") or metadata.get("intent") or ""),
                    task_type=str(metadata.get("task_type") or ""),
                    success=bool(metadata.get("success", False)),
                    memory=content,
                    retrieval_preview=content[:400],
                )
            )
        self.runtime = MemRLRuntimeEngine(store=self.store, config=self._config)
        self._pending_selected_ids = {}

    def export_state(self, path: str | Path, *, source: str = "") -> None:
        """Write the current MemRL store as a portable JSON warmstart artifact."""
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "id": rec.id,
                "content": rec.content,
                "metadata": rec.metadata,
            }
            for rec in self.store.iter_chronological()
        ]
        payload = {
            "format": "agentmem_alfworld_memrl_state_v1",
            "source": source,
            "top_k": self.top_k,
            "memrl_config": self._config.__dict__,
            "records": records,
        }
        state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def create_memory_backend(
    backend: str,
    *,
    top_k: int = 3,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    config_path: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> BaseAlfWorldMemoryBackend:
    key = str(backend or "").strip().lower().replace("-", "_")
    kwargs = {
        "config_path": config_path,
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "llm_api_key": llm_api_key,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
        "embedding_api_key": embedding_api_key,
    }
    kwargs = {name: value for name, value in kwargs.items() if value not in (None, "")}
    extra_kwargs: dict[str, Any] = {}
    if save_dir and key in {"hipporag", "hipporagv2", "plugmem", "ama_agent", "ama"}:
        extra_kwargs["save_dir"] = str(save_dir)

    if key in {"hipporag", "hipporagv2"}:
        return HippoRAGv2MemoryBackend(top_k=top_k, **kwargs, **extra_kwargs)
    if key == "plugmem":
        return PlugMemMemoryBackend(top_k=top_k, **kwargs, **extra_kwargs)
    if key == "simplemem":
        return SimpleMemMemoryBackend(top_k=top_k, **kwargs)
    if key == "dci_lite":
        return DCILiteMemoryBackend(top_k=top_k, **kwargs)
    if key == "dci_lite_sum":
        return DCILiteSummarizeMemoryBackend(top_k=top_k, **kwargs)
    if key in {"automem", "dci_memory"}:

        return DCILiteMemoryBackend(top_k=top_k, **kwargs)
    if key == "lightmem":
        return LightMemMemoryBackend(top_k=top_k, **kwargs)
    if key in {"ama_agent", "ama"}:
        return AMAAgentMemoryBackend(top_k=top_k, **kwargs, **extra_kwargs)
    if key in {"memt", "mem_t", "mem-t"}:
        return MemTMemoryBackend(top_k=top_k, **kwargs)
    if key == "memrl":
        return MemRLAlfWorldMemoryBackend(top_k=top_k, **kwargs)
    raise ValueError(f"Unknown ALFWorld memory backend: {backend}")

def format_trajectory_for_memory(trajectory: list[dict[str, Any]]) -> str:
    """Convert an ALFWorld episode trajectory into the shared turn text format."""
    lines: list[str] = []
    turn_idx = 0
    for step in trajectory:
        action = str(step.get("action") or "").strip()
        observation = str(step.get("observation") or "").strip()
        if not action:
            continue
        lines.append(f"Turn {turn_idx}:")
        lines.append(f"Action: {action}")
        lines.append(f"Observation: {observation}")
        turn_idx += 1
    return "\n".join(lines)

def _normalize_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 1
    }

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)
