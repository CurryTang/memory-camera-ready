from __future__ import annotations

import os
import re
import inspect
import importlib.util
import json
import requests
import sys
import tempfile
import threading
import __future__
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from agentmem.plugmem.bridge import PlugMemBridge
from agentmem.plugmem.session import PlugMemSession, PlugMemStep
from agentmem.providers.base import Message
from agentmem.providers.openai_compat import OpenAICompatibleProvider

_PLUGMEM_ENV_KEYS = (
    "LLM_NAME",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_DISABLE_REASONING",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "QWEN_MODEL_NAME",
    "QWEN_BASE_URL",
    "VLLM_QWEN_API_KEY",
    "AZURE_ENDPOINT",
    "AZURE_DPSK_API_KEY",
    "AZURE_DPSK_ENDPOINT",
    "TOKEN_USAGE_FILE",
    "WRITE",
    "DIR_PATH",
)
_PLUGMEM_LOCK = threading.RLock()

_MODE_ALIASES = {
    "semantic": "semantic_memory",
    "semantic_memory": "semantic_memory",
    "episodic": "episodic_memory",
    "episodic_memory": "episodic_memory",
    "procedural": "procedural_memory",
    "procedural_memory": "procedural_memory",
}
_MODE_ORDER = ("semantic_memory", "episodic_memory", "procedural_memory")

def _preload_plugmem_utils_for_py39(source_root: Path) -> None:
    """Load PlugMem's top-level ``utils.py`` on Python versions before 3.10.

    The vendored PlugMem source uses PEP 604 annotations such as
    ``str | None`` without ``from __future__ import annotations``. Python 3.9
    evaluates those annotations at import time and raises ``TypeError``. Keep
    the compatibility patch in this wrapper instead of editing vendored source.
    """
    if sys.version_info >= (3, 10) or "utils" in sys.modules:
        return

    utils_path = source_root / "utils.py"
    if not utils_path.exists():
        return

    spec = importlib.util.spec_from_file_location("utils", utils_path)
    if spec is None:
        return
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(utils_path)
    module.__package__ = ""
    source = utils_path.read_text(encoding="utf-8")
    code = compile(
        source,
        str(utils_path),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    sys.modules["utils"] = module
    exec(code, module.__dict__)

def default_plugmem_source_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "vendor"
        / "plugmem"
    )

def resolve_plugmem_source_root(source_root: Path | str | None) -> Path:
    raw = Path(source_root) if source_root is not None else default_plugmem_source_root()
    return PlugMemBridge.resolve_source_root(raw)

def normalize_plugmem_modes(modes: str | Sequence[str] | None) -> tuple[str, ...]:
    if modes is None:
        return _MODE_ORDER
    if isinstance(modes, str):
        raw_items = [item.strip() for item in modes.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip() for item in modes if str(item).strip()]
    if not raw_items:
        return _MODE_ORDER
    if len(raw_items) == 1 and raw_items[0].lower() in {"all", "default"}:
        return _MODE_ORDER

    selected: list[str] = []
    for item in raw_items:
        key = item.lower().replace("-", "_")
        mode = _MODE_ALIASES.get(key)
        if mode is None:
            raise ValueError(f"Unsupported PlugMem memory mode: {item}")
        if mode not in selected:
            selected.append(mode)
    return tuple(selected)

def ensure_plugmem_memory_graph_compat(
    memory_graph_cls: type[Any],
    *,
    embedding_fn: Optional[Any] = None,
) -> type[Any]:
    """Backfill helpers missing from older PlugMem revisions.

    Some upstream revisions call ``MemoryGraph._parallel_get_embeddings()`` from
    one disk loader without defining the helper. We patch the class
    at runtime so the parent repo stays self-contained and the vendored submodule
    does not need local source edits.
    """
    if hasattr(memory_graph_cls, "_parallel_get_embeddings"):
        return memory_graph_cls

    if embedding_fn is None:
        module = sys.modules.get(memory_graph_cls.__module__)
        embedding_fn = getattr(module, "get_embedding", None)
    if embedding_fn is None:
        raise ImportError(
            "Could not resolve PlugMem get_embedding for MemoryGraph compatibility patch."
        )

    def _parallel_get_embeddings(self: Any, texts: Sequence[str]) -> dict[str, Any]:
        cache: dict[str, Any] = {}
        for text in texts:
            normalized = str(text or "").strip()
            if not normalized or normalized in cache:
                continue
            cache[normalized] = embedding_fn(normalized)
        return cache

    setattr(memory_graph_cls, "_parallel_get_embeddings", _parallel_get_embeddings)
    return memory_graph_cls

def ensure_plugmem_semantic_node_compat(semantic_node_cls: type[Any]) -> type[Any]:
    """Accept newer dump-loader kwargs on older PlugMem SemanticNode revisions."""
    init = getattr(semantic_node_cls, "__init__", None)
    if init is None:
        return semantic_node_cls
    signature = inspect.signature(init)
    if "date" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return semantic_node_cls

    def _patched_init(self: Any, *args: Any, date: Any = None, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        setattr(self, "date", date)

    setattr(semantic_node_cls, "__init__", _patched_init)
    return semantic_node_cls

def ensure_plugmem_episodic_node_compat(episodic_node_cls: type[Any]) -> type[Any]:
    """Backfill attributes expected by newer dump loaders on old EpisdoicNode."""
    init = getattr(episodic_node_cls, "__init__", None)
    if init is None:
        return episodic_node_cls

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        if not hasattr(self, "semantic_nodes"):
            setattr(self, "semantic_nodes", [])

    setattr(episodic_node_cls, "__init__", _patched_init)
    return episodic_node_cls

def ensure_plugmem_procedural_node_compat(procedural_node_cls: type[Any]) -> type[Any]:
    """Tolerate older dumps that omit procedural return values."""
    init = getattr(procedural_node_cls, "__init__", None)
    if init is None:
        return procedural_node_cls

    def _patched_init(self: Any, procedural_memory: Any, *args: Any, **kwargs: Any) -> None:
        normalized = procedural_memory
        if isinstance(procedural_memory, Mapping) and "return" not in procedural_memory:
            normalized = dict(procedural_memory)
            normalized.setdefault("return", 0)
        init(self, normalized, *args, **kwargs)
        if not hasattr(self, "Return"):
            setattr(self, "Return", 0)

    setattr(procedural_node_cls, "__init__", _patched_init)
    return procedural_node_cls

def ensure_plugmem_subgoal_node_compat(subgoal_node_cls: type[Any]) -> type[Any]:
    """Accept both dict and list subgoal dump formats produced by PlugMem."""

    def _coerce_subgoal(payload: Any, subgoal_id: Any) -> str:
        if isinstance(payload, Mapping):
            return str(payload.get("subgoal") or "")
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            fallback = ""
            for item in payload:
                if isinstance(item, Mapping):
                    if not fallback:
                        fallback = str(item.get("subgoal") or "")
                    if item.get("subgoal_id") == subgoal_id:
                        return str(item.get("subgoal") or "")
                elif item and not fallback:
                    fallback = str(item)
            return fallback
        return str(payload or "")

    def _patched_get_subgoal(self: Any) -> str:
        dir_path = os.environ.get("DIR_PATH", None)
        if dir_path:
            path = Path(dir_path) / "subgoal" / f"subgoal_{self.subgoal_id}.json"
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    return _coerce_subgoal(json.load(f), self.subgoal_id)
        return _coerce_subgoal(getattr(self, "subgoal", ""), getattr(self, "subgoal_id", None))

    setattr(subgoal_node_cls, "get_subgoal", _patched_get_subgoal)
    return subgoal_node_cls

def _openai_compat_get_embedding(text: Any, embedding_model: Optional[str] = None) -> Any:
    url = os.environ["EMBEDDING_BASE_URL"]
    model_name = embedding_model or os.environ.get("EMBEDDING_MODEL_NAME") or "nvidia/NV-Embed-v2"
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("EMBEDDING_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        url,
        headers=headers,
        json={"model": model_name, "input": str(text or "")[:8192]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def install_plugmem_embedding_compat(module_names: Sequence[str]) -> None:
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "get_embedding"):
            setattr(module, "get_embedding", _openai_compat_get_embedding)

@dataclass
class PlugMemGraphMemory:
    graph: Any
    session: PlugMemSession
    sample_dir: Path

@dataclass
class PlugMemRetrievalResult:
    question: str
    contexts: dict[str, str]
    prompts: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    variables: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def combined_context(self) -> str:
        parts: list[str] = []
        labels = {
            "semantic_memory": "Semantic Memory",
            "episodic_memory": "Episodic Memory",
            "procedural_memory": "Procedural Memory",
        }
        for mode in _MODE_ORDER:
            text = str(self.contexts.get(mode, "") or "").strip()
            if not text:
                continue
            parts.append(f"{labels[mode]}:\n{text}")
        return "\n\n".join(parts)

@contextmanager
def plugmem_env(env_overrides: Mapping[str, Optional[str]], sample_dir: Path | None = None):
    merged = {str(key): value for key, value in env_overrides.items()}
    if sample_dir is not None:
        merged["DIR_PATH"] = str(Path(sample_dir).resolve())

    with _PLUGMEM_LOCK:
        saved = {key: os.environ.get(key) for key in _PLUGMEM_ENV_KEYS}
        try:
            for key, value in merged.items():
                if value is None:
                    continue
                os.environ[key] = str(value)
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

class PlugMemUpstreamAdapter:
    """Shared adapter over the upstream PlugMem graph construction and retrieval stack."""

    def __init__(
        self,
        *,
        source_root: Path | str | None = None,
        env_overrides: Optional[Mapping[str, str]] = None,
        retrieval_topk: int = 20,
        memory_modes: str | Sequence[str] | None = None,
        answer_model: Optional[str] = None,
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        provider_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.source_root = resolve_plugmem_source_root(source_root)
        source_root_str = str(self.source_root)
        if source_root_str not in sys.path:
            sys.path.insert(0, source_root_str)

        _saved_utils = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "utils" or name.startswith("utils.")
        }

        _hidden_paths: list[tuple[int, str]] = []
        for _idx in range(len(sys.path) - 1, -1, -1):
            _p = sys.path[_idx]
            if _p == source_root_str:
                continue
            _candidate = os.path.join(_p, "utils") if _p else "utils"
            if os.path.isdir(_candidate):
                _hidden_paths.append((_idx, sys.path.pop(_idx)))

        if sys.path[0] != source_root_str:
            try:
                sys.path.remove(source_root_str)
            except ValueError:
                pass
            sys.path.insert(0, source_root_str)
        try:
            _preload_plugmem_utils_for_py39(self.source_root)
            from memory_retrieving.graph_node import EpisdoicNode, ProceduralNode, SemanticNode, SubgoalNode
            from memory_retrieving.memory_graph import MemoryGraph
            from memory_retrieving.value_longmemeval import (
                ProceduralEqual,
                ProceduralRelevant,
                SemanticEqual,
                SemanticRelevant,
                SubgoalEqual,
                SubgoalRelevant,
                TagEqual,
                TagRelevant,
            )
            from memory_structuring.memory import Memory
            _plugmem_imports_done = True
        finally:

            for _idx, _p in sorted(_hidden_paths):
                sys.path.insert(_idx, _p)
            for name in list(sys.modules):
                if name == "utils" or name.startswith("utils."):
                    sys.modules.pop(name, None)
            sys.modules.update(_saved_utils)

        install_plugmem_embedding_compat(
            (
                "utils",
                "memory_structuring.memory",
                "memory_retrieving.memory_graph",
            )
        )
        memory_graph_module = sys.modules.get(MemoryGraph.__module__)
        self._get_embedding = getattr(memory_graph_module, "get_embedding", None)
        self._Memory = Memory
        self._EpisodicNode = ensure_plugmem_episodic_node_compat(EpisdoicNode)
        self._ProceduralNode = ensure_plugmem_procedural_node_compat(ProceduralNode)
        self._SemanticNode = ensure_plugmem_semantic_node_compat(SemanticNode)
        self._SubgoalNode = ensure_plugmem_subgoal_node_compat(SubgoalNode)
        self._MemoryGraph = ensure_plugmem_memory_graph_compat(
            MemoryGraph,
            embedding_fn=self._get_embedding,
        )
        self._TagEqual = TagEqual
        self._TagRelevant = TagRelevant
        self._SemanticEqual = SemanticEqual
        self._SemanticRelevant = SemanticRelevant
        self._SubgoalEqual = SubgoalEqual
        self._SubgoalRelevant = SubgoalRelevant
        self._ProceduralEqual = ProceduralEqual
        self._ProceduralRelevant = ProceduralRelevant
        self.env_overrides = dict(env_overrides or {})
        self.retrieval_topk = max(1, int(retrieval_topk))
        self.memory_modes = normalize_plugmem_modes(memory_modes)
        self._episode_counter = 0

        self._provider: Optional[OpenAICompatibleProvider] = None
        if answer_model:
            provider_args = dict(provider_kwargs or {})
            self._provider = OpenAICompatibleProvider(
                api_key=answer_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
                model=answer_model,
                base_url=answer_base_url,
                disable_thinking=True,
                **provider_args,
            )

    def _new_memory_graph(self, *, log_file: Path | str | None = None) -> Any:
        return self._MemoryGraph(
            tag_equal=self._TagEqual(),
            tag_relevant=self._TagRelevant(k=min(5, self.retrieval_topk)),
            semantic_equal=self._SemanticEqual(),
            semantic_relevant=self._SemanticRelevant(k=self.retrieval_topk),
            subgoal_equal=self._SubgoalEqual(),
            subgoal_relevant=self._SubgoalRelevant(),
            procedural_equal=self._ProceduralEqual(),
            procedural_relevant=self._ProceduralRelevant(k=min(5, self.retrieval_topk)),
            log_file=str(log_file) if log_file is not None else None,
        )

    @staticmethod
    def _dialogue_like(session: PlugMemSession) -> bool:
        benchmark = str(session.metadata.get("benchmark", "") or "").lower()
        if benchmark == "locomo":
            return True
        if benchmark == "amabench":
            return False
        return bool(session.steps) and all(str(step.action or "") == "dialogue_turn" for step in session.steps)

    @staticmethod
    def _render_observation(step: PlugMemStep) -> str:
        speaker = str(step.speaker or "").strip() or "Agent"
        observation = str(step.observation or "").strip()
        action = str(step.action or "").strip()

        step_tag = f"[step {step.index}] " if step.index is not None else ""
        if action == "dialogue_turn":
            return f"{speaker} says: {observation}".strip()
        if observation:
            return f"{step_tag}{speaker} observes: {observation}".strip()
        if action:
            return f"{step_tag}{speaker} performs: {action}".strip()
        return f"{step_tag}{speaker} observes nothing"

    @staticmethod
    def _render_action(step: PlugMemStep) -> str:
        speaker = str(step.speaker or "").strip() or "Agent"
        action = str(step.action or "").strip()
        observation = str(step.observation or "").strip()
        step_tag = f"[step {step.index}] " if step.index is not None else ""
        if action == "dialogue_turn":
            return f"{speaker} says: {observation}".strip() if observation else f"{speaker} says nothing"
        if action:
            return f"{step_tag}{speaker} action: {action}".strip()
        if observation:
            return f"{step_tag}{speaker} reports: {observation}".strip()
        return f"{step_tag}{speaker} acts"

    @staticmethod
    def _initial_observation(session: PlugMemSession, *, dialogue_like: bool) -> str:
        goal = str(session.goal or "PlugMem").strip() or "PlugMem"
        if dialogue_like:
            participants = sorted(
                {
                    str(step.speaker or "").strip()
                    for step in session.steps
                    if str(step.speaker or "").strip()
                }
            )
            if participants:
                return f"Conversation begins. Goal: {goal}. Participants: {', '.join(participants)}"
            return f"Conversation begins. Goal: {goal}"

        task_type = str(session.metadata.get("task_type", "") or "").strip()
        domain = str(session.metadata.get("domain", "") or "").strip()
        parts = [f"Episode begins. Goal: {goal}"]
        if task_type:
            parts.append(f"Task type: {task_type}")
        if domain:
            parts.append(f"Domain: {domain}")
        return "\n".join(parts)

    def _transitions(self, session: PlugMemSession) -> tuple[str, list[tuple[str, str]], str]:
        if not session.steps:
            return str(session.goal or "PlugMem"), [], "0"

        dialogue_like = self._dialogue_like(session)
        initial_observation = self._initial_observation(session, dialogue_like=dialogue_like)
        transitions: list[tuple[str, str]] = []

        if dialogue_like:
            if len(session.steps) == 1:
                step = session.steps[0]
                transitions.append((self._render_action(step), self._render_observation(step)))
            else:
                for current_step, next_step in zip(session.steps[:-1], session.steps[1:]):
                    transitions.append(
                        (self._render_action(current_step), self._render_observation(next_step))
                    )
        else:
            for step in session.steps:
                transitions.append((self._render_action(step), self._render_observation(step)))

        first_timestamp = str(session.steps[0].timestamp or 0)
        return initial_observation, transitions, first_timestamp

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _coerce_tags(value: Any) -> list[str]:
        if isinstance(value, str):
            raw_items = re.split(r"[,;]", value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            raw_items = [str(item) for item in value]
        else:
            raw_items = []
        tags: list[str] = []
        for item in raw_items:
            tag = str(item).strip().strip("[]\"'`")
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _embed_text(self, text: str) -> Any:
        if self._get_embedding is None:
            return None
        normalized = str(text or "").strip()
        if not normalized:
            return self._get_embedding("")
        return self._get_embedding(normalized)

    def _normalize_memory(self, memory: Any) -> Any:
        payload = getattr(memory, "memory", None)
        embeddings = getattr(memory, "memory_embedding", None)
        if not isinstance(payload, dict) or not isinstance(embeddings, dict):
            return memory

        episodic_raw = list(payload.get("episodic") or [])
        normalized_episodic: list[list[dict[str, Any]]] = []
        for trajectory in episodic_raw:
            if isinstance(trajectory, Mapping):
                steps_raw = [trajectory]
            elif isinstance(trajectory, Sequence) and not isinstance(trajectory, (str, bytes)):
                steps_raw = list(trajectory)
            else:
                text = str(trajectory or "").strip()
                steps_raw = [{"observation": text}] if text else []
            steps: list[dict[str, Any]] = []
            for step in steps_raw:
                if isinstance(step, Mapping):
                    normalized_step = dict(step)
                else:
                    text = str(step or "").strip()
                    if not text:
                        continue
                    normalized_step = {"observation": text}
                normalized_step.setdefault("action", "")
                normalized_step.setdefault("reward", "")
                normalized_step.setdefault("time", getattr(memory, "time", 0))
                normalized_step.setdefault("subgoal", str(getattr(memory, "goal", "") or "complete the task"))
                normalized_step.setdefault("state", "")
                steps.append(normalized_step)
            if steps:
                normalized_episodic.append(steps)

        semantic_raw = list(payload.get("semantic") or [])
        normalized_semantic: list[dict[str, Any]] = []
        for item in semantic_raw:
            if isinstance(item, Mapping):
                semantic_item = dict(item)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and item:
                semantic_item = {
                    "semantic_memory": str(item[0]),
                    "tags": list(item[1]) if len(item) > 1 and isinstance(item[1], Sequence) else [],
                }
            else:
                continue
            semantic_text = str(
                semantic_item.get("semantic_memory") or semantic_item.get("statement") or ""
            ).strip()
            if not semantic_text:
                continue
            semantic_item["semantic_memory"] = semantic_text
            semantic_item["tags"] = self._coerce_tags(semantic_item.get("tags"))
            semantic_item["trajectory_num"] = self._coerce_int(semantic_item.get("trajectory_num"), 0)
            semantic_item["turn_num"] = self._coerce_int(semantic_item.get("turn_num"), 0)
            semantic_item.setdefault("time", getattr(memory, "time", 0))
            normalized_semantic.append(semantic_item)

        procedural_raw = list(payload.get("procedural") or [])
        normalized_procedural: list[dict[str, Any]] = []
        for item in procedural_raw:
            if isinstance(item, Mapping):
                procedural_item = dict(item)
            else:
                continue
            procedural_text = str(
                procedural_item.get("procedural_memory")
                or procedural_item.get("experience")
                or procedural_item.get("insight")
                or ""
            ).strip()
            if not procedural_text:
                continue
            procedural_item["procedural_memory"] = procedural_text
            subgoal = str(
                procedural_item.get("subgoal") or procedural_item.get("sub_goal") or ""
            ).strip()
            procedural_item["subgoal"] = subgoal or "complete the task"
            procedural_item["trajectory_num"] = self._coerce_int(procedural_item.get("trajectory_num"), 0)
            procedural_item.setdefault("time", getattr(memory, "time", 0))
            procedural_item["return"] = self._coerce_int(procedural_item.get("return"), 0)
            normalized_procedural.append(procedural_item)

        if not normalized_episodic and (normalized_semantic or normalized_procedural):
            normalized_episodic = [[{
                "observation": str(getattr(memory, "observation_t0", "") or ""),
                "action": "",
                "reward": "",
                "time": getattr(memory, "time", 0),
            }]]

        max_traj_index = max(len(normalized_episodic) - 1, 0)
        for semantic_item in normalized_semantic:
            trajectory_num = min(max(semantic_item["trajectory_num"], 0), max_traj_index)
            turn_count = len(normalized_episodic[trajectory_num]) if normalized_episodic else 1
            semantic_item["trajectory_num"] = trajectory_num
            semantic_item["turn_num"] = min(max(semantic_item["turn_num"], 0), max(turn_count - 1, 0))
        for procedural_item in normalized_procedural:
            procedural_item["trajectory_num"] = min(
                max(procedural_item["trajectory_num"], 0),
                max_traj_index,
            )

        semantic_embeddings_raw = list(embeddings.get("semantic") or [])
        normalized_semantic_embeddings: list[dict[str, Any]] = []
        for idx, semantic_item in enumerate(normalized_semantic):
            raw_embedding = semantic_embeddings_raw[idx] if idx < len(semantic_embeddings_raw) else {}
            emb_mapping = dict(raw_embedding) if isinstance(raw_embedding, Mapping) else {}
            semantic_embedding = emb_mapping.get("semantic_memory")
            if semantic_embedding is None:
                semantic_embedding = self._embed_text(semantic_item["semantic_memory"])
            tag_embeddings = list(emb_mapping.get("tags") or [])
            if len(tag_embeddings) != len(semantic_item["tags"]):
                tag_embeddings = [self._embed_text(tag) for tag in semantic_item["tags"]]
            normalized_semantic_embeddings.append(
                {
                    "semantic_memory": semantic_embedding,
                    "tags": tag_embeddings,
                }
            )

        procedural_embeddings_raw = list(embeddings.get("procedural") or [])
        normalized_procedural_embeddings: list[dict[str, Any]] = []
        for idx, procedural_item in enumerate(normalized_procedural):
            raw_embedding = procedural_embeddings_raw[idx] if idx < len(procedural_embeddings_raw) else {}
            emb_mapping = dict(raw_embedding) if isinstance(raw_embedding, Mapping) else {}
            subgoal_embedding = emb_mapping.get("subgoal")
            if subgoal_embedding is None:
                subgoal_embedding = self._embed_text(procedural_item["subgoal"])
            procedural_embedding = emb_mapping.get("procedural_memory")
            if procedural_embedding is None:
                procedural_embedding = self._embed_text(procedural_item["procedural_memory"])
            normalized_procedural_embeddings.append(
                {
                    "subgoal": subgoal_embedding,
                    "procedural_memory": procedural_embedding,
                }
            )

        payload["episodic"] = normalized_episodic
        payload["semantic"] = normalized_semantic
        payload["procedural"] = normalized_procedural
        embeddings["semantic"] = normalized_semantic_embeddings
        embeddings["procedural"] = normalized_procedural_embeddings
        return memory

    def build_memory(self, session: PlugMemSession) -> Any:
        initial_observation, transitions, first_timestamp = self._transitions(session)
        memory = self._Memory(goal=str(session.goal or "PlugMem"), observation=initial_observation, time=first_timestamp)
        for action_text, observation_text in transitions:
            memory.append(action_t0=action_text, observation_t1=observation_text)
        memory.close()
        return self._normalize_memory(memory)

    def _build_memory_from_chat_session(
        self,
        turns: Sequence[Mapping[str, Any]],
        *,
        time_value: Optional[str],
        goal: str,
    ) -> Any | None:
        normalized_turns = [
            {
                "role": str(turn.get("role", "") or "").strip().lower(),
                "content": str(turn.get("content", "") or "").strip(),
            }
            for turn in turns
            if str(turn.get("content", "") or "").strip()
        ]
        if not normalized_turns:
            return None

        memory_time = f"Date: {time_value}" if time_value else "0"
        if normalized_turns[0]["role"] == "user":
            memory = self._Memory(
                goal=goal,
                observation=normalized_turns[0]["content"],
                time=memory_time,
            )
            start_index = 1
        else:
            memory = self._Memory(
                goal=goal,
                observation="User: ...",
                time=memory_time,
            )
            start_index = 0

        pending_action: Optional[str] = None
        for turn in normalized_turns[start_index:]:
            if turn["role"] == "assistant":
                pending_action = f"Agent Say: {turn['content']}"
                continue
            if pending_action is None:
                raise ValueError("Encountered user turn before any assistant action in chat session.")
            memory.append(
                action_t0=pending_action,
                observation_t1=f"User Say: {turn['content']}",
            )
            pending_action = None
        memory.close()
        return memory

    def build_graph(self, session: PlugMemSession, sample_dir: Path) -> PlugMemGraphMemory:
        sample_dir = Path(sample_dir).resolve()
        sample_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("episodic_memory", "semantic_memory", "procedural_memory", "tag", "subgoal"):
            (sample_dir / subdir).mkdir(parents=True, exist_ok=True)

        graph = self._new_memory_graph(log_file=sample_dir / "plugmem.log")
        with plugmem_env(self.env_overrides, sample_dir=sample_dir):
            source_sessions = session.metadata.get("plugmem_source_sessions")
            if isinstance(source_sessions, Sequence) and source_sessions:
                for source_session in source_sessions:
                    if not isinstance(source_session, Mapping):
                        continue
                    memory = self._build_memory_from_chat_session(
                        source_session.get("turns", []),
                        time_value=str(source_session.get("time") or "").strip() or None,
                        goal=str(session.goal or "PlugMem"),
                    )
                    if memory is not None:
                        graph.insert(self._normalize_memory(memory))
            elif session.steps:
                graph.insert(self.build_memory(session))
        return PlugMemGraphMemory(graph=graph, session=session, sample_dir=sample_dir)

    def build_from_trajectory_text(
        self,
        traj_text: Any,
        *,
        task: str = "",
        sample_dir: Optional[Path] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PlugMemGraphMemory:
        raw_steps = list(parse_trajectory_steps(traj_text))

        def _keep(step: Any) -> bool:
            action = getattr(step, "action", None)
            observation = getattr(step, "observation", None)

            if isinstance(step, dict):
                action = step.get("action", action)
                observation = step.get("observation", observation)
            return bool((action and str(action).strip()) or (observation and str(observation).strip()))

        filtered = [s for s in raw_steps if _keep(s)]
        _MAX_TURNS = int(os.environ.get("PLUGMEM_MAX_TURNS", "0") or "0")
        if _MAX_TURNS > 0 and len(filtered) > _MAX_TURNS:
            filtered = filtered[-_MAX_TURNS:]
        session = PlugMemSession(
            session_id=f"plugmem-{self._episode_counter}",
            goal=str(task or "Solve the task."),
            steps=filtered,
            metadata=dict(metadata or {"benchmark": "generic"}),
        )
        if sample_dir is None:
            root = Path(tempfile.mkdtemp(prefix="plugmem_upstream_"))
            sample_dir = root / "episode"
        self._episode_counter += 1
        return self.build_graph(session, sample_dir=sample_dir)

    def retrieve(
        self,
        memory: PlugMemGraphMemory,
        question_text: str,
        *,
        question_meta: Optional[Mapping[str, Any]] = None,
        modes: str | Sequence[str] | None = None,
    ) -> PlugMemRetrievalResult:
        selected_modes = normalize_plugmem_modes(modes) if modes is not None else self.memory_modes
        question_meta = dict(question_meta or {})
        time_value = str(question_meta.get("timestamp") or question_meta.get("time") or "")
        task_type = str(memory.session.metadata.get("task_type") or memory.session.goal or "")
        result = PlugMemRetrievalResult(question=question_text, contexts={})

        with plugmem_env(self.env_overrides, sample_dir=memory.sample_dir):
            for mode in selected_modes:
                try:
                    messages, variables, sel_type = memory.graph.retrieve_memory(
                        goal=str(memory.session.goal or ""),
                        observation=question_text,
                        time=time_value,
                        task_type=task_type,
                        mode=mode,
                    )
                    result.prompts[mode] = list(messages or [])
                    result.variables[mode] = dict(variables or {})
                    result.contexts[mode] = str((variables or {}).get(sel_type, "") or "").strip()
                except Exception as exc:
                    result.errors[mode] = str(exc)
        return result

    def retrieve_native(
        self,
        memory: PlugMemGraphMemory,
        question_text: str,
        *,
        question_meta: Optional[Mapping[str, Any]] = None,
        task_type: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        question_meta = dict(question_meta or {})
        resolved_task_type = str(
            task_type
            or question_meta.get("task_type")
            or memory.session.metadata.get("task_type")
            or memory.session.goal
            or ""
        )
        time_value = str(question_meta.get("timestamp") or question_meta.get("time") or question_meta.get("question_date") or "")
        kwargs: dict[str, Any] = {
            "goal": str(memory.session.goal or ""),
            "observation": question_text,
            "time": time_value,
            "task_type": resolved_task_type,
        }
        if mode is not None:
            kwargs["mode"] = mode

        with plugmem_env(self.env_overrides, sample_dir=memory.sample_dir):
            messages, variables, sel_type = memory.graph.retrieve_memory(**kwargs)
        return list(messages or []), dict(variables or {}), str(sel_type or "")

    def answer_question(
        self,
        memory: PlugMemGraphMemory,
        question_text: str,
        *,
        question_meta: Optional[Mapping[str, Any]] = None,
        modes: str | Sequence[str] | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        retrieval = self.retrieve(memory, question_text, question_meta=question_meta, modes=modes)
        combined_context = retrieval.combined_context()
        if self._provider is None:
            return {"answer": combined_context, "context": combined_context, "contexts": retrieval.contexts}

        user_prompt = "\n\n".join(
            [
                "You are answering a question using PlugMem retrieval output.",
                "Use the retrieved semantic facts, episodic traces, and procedural experiences below.",
                "If the evidence is insufficient, answer conservatively and do not invent details.",
                f"Question: {question_text}",
                f"Retrieved Memory:\n{combined_context or 'No relevant memory found.'}",
                "Answer:",
            ]
        )
        response = self._provider.chat(
            messages=[Message(role="user", content=user_prompt)],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return {
            "answer": str(response.content or "").strip(),
            "context": combined_context,
            "contexts": dict(retrieval.contexts),
            "errors": dict(retrieval.errors),
        }

def parse_trajectory_steps(traj_text: Any) -> Iterable[PlugMemStep]:
    if isinstance(traj_text, Sequence) and not isinstance(traj_text, (str, bytes)):
        for step_idx, item in enumerate(traj_text):
            if isinstance(item, Mapping):
                role = str(item.get("role") or item.get("speaker") or "agent").strip() or "agent"
                content = str(
                    item.get("content")
                    or item.get("observation")
                    or item.get("text")
                    or ""
                ).strip()
                action = str(item.get("action") or "").strip()
                if item.get("role") is not None or item.get("content") is not None:
                    action = "dialogue_turn"
                yield PlugMemStep(
                    index=step_idx,
                    speaker=role,
                    action=action,
                    observation=content,
                    timestamp=str(item.get("timestamp")) if item.get("timestamp") is not None else None,
                    metadata={"source_turn": dict(item)},
                )
            else:
                yield PlugMemStep(
                    index=step_idx,
                    speaker="agent",
                    action="dialogue_turn",
                    observation=str(item or "").strip(),
                )
        return

    import re as _re
    action = ""
    observation = ""
    field = None
    step_idx = 0
    pending_idx: int | None = None
    emitted = False
    for line in str(traj_text or "").splitlines():
        stripped = line.strip()
        m = _re.match(r"^(Turn|Step)\s+(\d+)\s*:?\s*$", stripped)
        if m:
            if action or observation:
                emitted = True
                yield PlugMemStep(
                    index=pending_idx if pending_idx is not None else step_idx,
                    speaker="agent",
                    action=action.strip(),
                    observation=observation.strip(),
                )
                step_idx += 1
                action = ""
                observation = ""
            pending_idx = int(m.group(2))
            field = None
            continue
        if stripped.startswith("Action:"):
            action = stripped[len("Action:") :].strip()
            field = "action"
            continue
        if stripped.startswith("Observation:"):
            observation = stripped[len("Observation:") :].strip()
            field = "observation"
            continue
        if field == "action":
            action = f"{action} {stripped}".strip()
        elif field == "observation":
            observation = f"{observation} {stripped}".strip()

    if action or observation:
        emitted = True
        yield PlugMemStep(
            index=pending_idx if pending_idx is not None else step_idx,
            speaker="agent",
            action=action.strip(),
            observation=observation.strip(),
        )
    if not emitted:
        for idx, chunk in enumerate(_raw_memory_chunks(traj_text)):
            yield PlugMemStep(
                index=idx,
                speaker="environment",
                action="read_memory_chunk",
                observation=chunk,
            )

def _raw_memory_chunks(text: Any, max_chars: int | None = None) -> list[str]:
    if max_chars is None:
        max_chars = int(os.environ.get("PLUGMEM_RAW_CHUNK_CHARS", "4000"))
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
