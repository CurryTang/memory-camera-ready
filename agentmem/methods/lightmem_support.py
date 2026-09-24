"""Shared LightMem helpers for benchmark-specific adapters.

These helpers keep the LightMem ports consistent across procedural trajectories
and long-document QA. The main responsibilities are:
- build runtime LightMem configs from OpenAI-compatible endpoints
- shape benchmark inputs into timestamped messages LightMem can ingest
- provide task-aware extraction prompts
"""

from __future__ import annotations

import inspect
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from openai import OpenAI

DEFAULT_RQ1_LLM_URL = "http://localhost:8000/v1"
DEFAULT_RQ1_EMB_URL = "http://localhost:8001/v1"
DEFAULT_RQ1_API_KEY = "EMPTY"

_OPENAI_MANAGER_COMPAT_ATTRS: dict[str, Any] = {
    "openrouter_base_url": None,
}

PROCEDURAL_LIGHTMEM_PROMPTS: dict[str, str] = {
    "factual": """
You are a task-memory extractor for embodied and web agents.
Your input is a topic-segmented trajectory with messages formatted like:

--- Topic X ---
[timestamp, weekday] <source_id>.<SpeakerName>: <message>

Extract every task-relevant memory item that could help answer downstream
questions or replay the task correctly.

Rules:
1. Process messages strictly in ascending source_id order.
2. Extract concrete facts about:
   - task goals and subgoals
   - actions taken and their outcomes
   - object, tool, room, page, or receptacle names
   - state changes such as picked up, opened, cleaned, heated, cooled, moved,
     toggled, visited, searched, failed, blocked, or missing
   - negative evidence and failed attempts
   - counts, ordering constraints, and temporal updates
3. Preserve exact entity names, counts, and qualifiers. Do not replace them
   with vague paraphrases.
4. When an observation changes world state, write the fact so the new state is
   explicit.
5. Keep each item self-contained and grounded. Do not invent facts.
6. Output JSON only:
{
  "data": [
    {"source_id": <source_id>, "fact": "<standalone memory fact>"}
  ]
}
""".strip(),
    "relational": """
You are a causal-procedural memory extractor for agent trajectories.
Another extractor records factual snippets. Your job is to extract reusable
dependencies, ordering constraints, and cause-effect relations.

Rules:
1. Process messages strictly in ascending source_id order.
2. Extract relations such as:
   - prerequisites ("the mug must be cleaned before placing it")
   - causal outcomes ("opening the fridge revealed the apple")
   - exploration cues ("the drawer was already searched and empty")
   - failure explanations ("action failed because the agent was not holding the object")
   - temporal chains ("after heating the soup, the agent placed it on the table")
3. Keep exact object/location names and state predicates.
4. Make each relation reusable for future planning or QA.
5. Output JSON only:
{
  "data": [
    {"source_id": <source_id>, "relation": "<causal or procedural relation>"}
  ]
}
""".strip(),
}

LONGDOC_LIGHTMEM_PROMPTS: dict[str, str] = {
    "factual": """
You are a long-document memory extractor for multiple-choice QA.
The input is a topic-segmented document represented as messages:

--- Topic X ---
[timestamp, weekday] <source_id>.Document: <chunk text>

Extract every self-contained fact that could help answer a later question.

Rules:
1. Process chunks strictly in ascending source_id order.
2. Extract precise statements about definitions, claims, methods, numbers,
   entities, dates, comparisons, evidence, and conclusions.
3. Preserve exact names, terminology, citations, quantities, and option-level
   distinctions whenever they appear.
4. Prefer faithful compression, not reinterpretation. Do not invent links not
   supported by the text.
5. Output JSON only:
{
  "data": [
    {"source_id": <source_id>, "fact": "<standalone fact>"}
  ]
}
""".strip(),
}

def sanitize_collection_name(text: str, *, max_len: int = 64) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text or "").strip())
    cleaned = cleaned.strip("._-") or "lightmem"
    return cleaned[:max_len]

def resolve_rq1_lightmem_endpoints(
    *,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    embedding_model: str | None = None,
    embedding_base_url: str | None = None,
    embedding_api_key: str | None = None,
    require_models: bool = True,
) -> dict[str, str]:
    """Resolve OpenAI-compatible LightMem endpoints from args and RQ1 env vars."""
    resolved = {
        "llm_model": llm_model or os.environ.get("RQ1_LLM_MODEL", ""),
        "llm_base_url": llm_base_url or os.environ.get("RQ1_LLM_URL", DEFAULT_RQ1_LLM_URL),
        "llm_api_key": llm_api_key or os.environ.get("OPENAI_API_KEY", DEFAULT_RQ1_API_KEY),
        "embedding_model": embedding_model or os.environ.get("RQ1_EMB_MODEL", ""),
        "embedding_base_url": embedding_base_url or os.environ.get("RQ1_EMB_URL", DEFAULT_RQ1_EMB_URL),
        "embedding_api_key": embedding_api_key or os.environ.get("OPENAI_API_KEY", DEFAULT_RQ1_API_KEY),
    }
    if require_models:
        missing = [
            name
            for name in ("llm_model", "embedding_model")
            if not str(resolved.get(name) or "").strip()
        ]
        if missing:
            raise ValueError(
                "LightMem requires model names via explicit kwargs or env vars: "
                + ", ".join("RQ1_" + name.upper().replace("EMBEDDING", "EMB") for name in missing)
            )
    return resolved

@lru_cache(maxsize=16)
def infer_embedding_dims(
    model: str,
    base_url: str,
    api_key: str,
) -> int:
    client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
    response = client.embeddings.create(
        model=model,
        input=["dimension probe"],
    )
    embedding = response.data[0].embedding
    return int(len(embedding))

def guess_embedding_dims(model: str) -> int:
    del model
    configured = os.environ.get("RQ1_EMB_DIMS") or os.environ.get("LIGHTMEM_EMB_DIMS")
    if configured:
        return int(configured)
    return 1536

def ensure_embedding_dims(
    *,
    model: str,
    base_url: str,
    api_key: str,
    configured_dims: int | None = None,
) -> int:
    if configured_dims:
        return int(configured_dims)
    try:
        return infer_embedding_dims(model, base_url, api_key)
    except Exception:
        return guess_embedding_dims(model)

def ensure_lightmem_openai_config_compat() -> None:
    """Patch LightMem's OpenAI config/embedder behavior for compatible APIs."""
    try:
        from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
    except Exception:
        BaseMemoryManagerConfig = None

    if BaseMemoryManagerConfig is not None and not getattr(
        BaseMemoryManagerConfig,
        "_agentmem_openai_compat",
        False,
    ):
        original_init = BaseMemoryManagerConfig.__init__
        try:
            parameters = inspect.signature(original_init).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            compat_values: dict[str, Any] = {}
            for attr, default in _OPENAI_MANAGER_COMPAT_ATTRS.items():
                if accepts_kwargs or attr in parameters:
                    compat_values[attr] = kwargs.get(attr, default)
                else:
                    compat_values[attr] = kwargs.pop(attr, default)
            original_init(self, *args, **kwargs)
            for attr, value in compat_values.items():
                if not hasattr(self, attr):
                    setattr(self, attr, value)

        BaseMemoryManagerConfig.__init__ = __init__
        BaseMemoryManagerConfig._agentmem_openai_compat = True

    _patch_lightmem_openai_embedder()

def _patch_lightmem_openai_embedder() -> None:
    """Do not send OpenAI ``dimensions`` to embedding APIs that reject it."""
    try:
        from lightmem.factory.text_embedder.openai import TextEmbedderOpenAI
    except Exception:
        return

    if getattr(TextEmbedderOpenAI, "_agentmem_embed_compat", False):
        return

    def embed(self: Any, text: Any) -> Any:
        max_tokens = int(os.environ.get("LIGHTMEM_EMBED_MAX_TOKENS", "7000"))

        def preprocess(value: Any) -> str:
            rendered = str(value).replace("\n", " ")
            if max_tokens <= 0:
                return rendered
            try:
                import tiktoken

                enc = tiktoken.get_encoding("cl100k_base")
                tokens = enc.encode(rendered)
                if len(tokens) > max_tokens:
                    rendered = enc.decode(tokens[:max_tokens])
            except Exception:

                max_chars = int(os.environ.get("LIGHTMEM_EMBED_MAX_CHARS", "22000"))
                if max_chars > 0 and len(rendered) > max_chars:
                    rendered = rendered[:max_chars]
            return rendered

        model = getattr(self.config, "model", None)
        api_params: dict[str, Any] = {"model": model}
        dims = getattr(self.config, "embedding_dims", None)
        if dims and "text-embedding-3" in str(model or ""):
            api_params["dimensions"] = dims

        if isinstance(text, list):
            if not text:
                return []
            resp = self.client.embeddings.create(
                input=[preprocess(item) for item in text],
                **api_params,
            )
            self.total_calls += 1
            self.total_tokens += getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
            return [item.embedding for item in resp.data]

        resp = self.client.embeddings.create(input=[preprocess(text)], **api_params)
        self.total_calls += 1
        self.total_tokens += getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
        return resp.data[0].embedding

    TextEmbedderOpenAI.embed = embed
    TextEmbedderOpenAI._agentmem_embed_compat = True

def make_lightmem_config(
    *,
    collection_name: str,
    root_dir: str | os.PathLike[str] | None,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    embedding_model: str,
    embedding_base_url: str,
    embedding_api_key: str,
    embedding_dims: int,
    pre_compress: bool = True,
    topic_segment: bool = True,
    precomp_topic_shared: bool = True,
    messages_use: str = "hybrid",
    metadata_generate: bool = True,
    text_summary: bool = True,
    extract_threshold: float = 0.1,
    extraction_mode: str = "flat",
    llm_max_tokens: int = 4096,
    topic_segmenter: dict[str, Any] | None = None,
    pre_compressor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_lightmem_openai_config_compat()
    base_dir = Path(root_dir or tempfile.mkdtemp(prefix="lightmem_store_"))
    base_dir.mkdir(parents=True, exist_ok=True)
    collection = sanitize_collection_name(collection_name)
    config = {
        "pre_compress": bool(pre_compress),
        "topic_segment": bool(topic_segment),
        "precomp_topic_shared": bool(precomp_topic_shared),
        "messages_use": str(messages_use),
        "metadata_generate": bool(metadata_generate),
        "text_summary": bool(text_summary),
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": llm_model,
                "api_key": llm_api_key or "EMPTY",
                "max_tokens": int(llm_max_tokens),
                "openai_base_url": llm_base_url,
                "openrouter_base_url": llm_base_url,
            },
        },
        "extract_threshold": float(extract_threshold),
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "openai",
            "configs": {
                "model": embedding_model,
                "api_key": embedding_api_key or "EMPTY",
                "openai_base_url": embedding_base_url,
                "embedding_dims": int(embedding_dims),
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection,
                "embedding_model_dims": int(embedding_dims),
                "path": str(base_dir / collection),
                "on_disk": True,
            },
        },
        "summary_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": f"{collection}_summary",
                "embedding_model_dims": int(embedding_dims),
                "path": str(base_dir / f"{collection}_summary"),
                "on_disk": True,
            },
        },
        "update": "offline",
        "extraction_mode": extraction_mode,
    }
    if pre_compress and pre_compressor:
        config["pre_compressor"] = pre_compressor
    if topic_segment and topic_segmenter:
        config["topic_segmenter"] = topic_segmenter
    return config

def format_lightmem_retrieval(
    results: Any,
    *,
    header: str | None = None,
    item_prefix: str = "Memory",
) -> str:
    if isinstance(results, str):
        body = str(results).strip()
        if not body:
            return ""
        return f"{header}\n{body}".strip() if header else body

    if not isinstance(results, Sequence):
        text = str(results or "").strip()
        if not text:
            return ""
        return f"{header}\n{text}".strip() if header else text

    items = [str(item).strip() for item in results if str(item).strip()]
    if not items:
        return ""
    lines: list[str] = []
    if header:
        lines.append(header.strip())
    for idx, item in enumerate(items, start=1):
        lines.append(f"{item_prefix} {idx}: {item}")
    return "\n\n".join(lines)

def procedural_turns_from_text(traj_text: str, *, task: str = "") -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    if task:
        turns.append(("user", f"Task: {task}"))

    current_obs = ""
    max_chars = int(os.environ.get("LIGHTMEM_MAX_MESSAGE_CHARS", "8000") or 0)
    max_tokens = int(os.environ.get("LIGHTMEM_MAX_INPUT_TOKENS", "0") or 0)
    if max_tokens > 0:
        max_chars = min(max_chars, max_tokens * 4) if max_chars > 0 else max_tokens * 4

    def add_turn(role: str, content: str) -> None:
        text = str(content or "").strip()
        if not text:
            return
        if max_chars <= 0 or len(text) <= max_chars:
            turns.append((role, text))
            return
        overlap = max(0, int(os.environ.get("LIGHTMEM_MESSAGE_OVERLAP_CHARS", "400") or 0))
        step = max(1, max_chars - overlap)
        label = "continued"
        for idx, start in enumerate(range(0, len(text), step), start=1):
            chunk = text[start : start + max_chars].strip()
            if chunk:
                turns.append((role, f"{label} chunk {idx}: {chunk}"))
            if start + max_chars >= len(text):
                break

    for raw_line in str(traj_text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("Action:"):
            if current_obs:
                add_turn("assistant", f"Observation: {current_obs.strip()}")
                current_obs = ""
            action = line[len("Action:") :].strip()
            if action:
                add_turn("user", f"Action: {action}")
        elif line.startswith("Observation:"):
            current_obs = line[len("Observation:") :].strip()
        elif line.startswith(("Turn ", "Step ")):
            if current_obs:
                add_turn("assistant", f"Observation: {current_obs.strip()}")
                current_obs = ""
        elif current_obs:
            current_obs = f"{current_obs} {line}".strip()

    if current_obs:
        add_turn("assistant", f"Observation: {current_obs.strip()}")
    if not turns:
        add_turn("user", str(traj_text or "").strip())
    max_messages = int(os.environ.get("LIGHTMEM_MAX_MESSAGES", "0") or 0)
    if max_messages > 0 and len(turns) > max_messages:
        turns = turns[-max_messages:]
    return turns

def timestamped_messages_from_turns(
    turns: Iterable[tuple[str, str]],
    *,
    start_minute: int = 0,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for idx, (role, content) in enumerate(turns, start=start_minute):

        day = (idx // (60 * 24)) % 28 + 1
        hour = (idx // 60) % 24
        minute = idx % 60
        messages.append(
            {
                "role": role,
                "content": str(content).strip(),
                "time_stamp": f"2026/01/{day:02d} (Thu) {hour:02d}:{minute:02d}",
            }
        )
    return messages

def chunk_text_for_lightmem(
    text: str,
    *,
    chunk_chars: int = 2800,
    overlap_chars: int = 250,
) -> list[str]:
    clean = str(text or "").strip()
    if not clean:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
    if not paragraphs:
        return [clean]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > chunk_chars:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks

def document_messages_from_text(
    text: str,
    *,
    domain: str = "",
    sub_domain: str = "",
    chunk_chars: int = 2800,
    overlap_chars: int = 250,
) -> list[dict[str, str]]:
    chunks = chunk_text_for_lightmem(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    turns: list[tuple[str, str]] = []
    meta = " | ".join(part for part in [domain, sub_domain] if part)
    if meta:
        turns.append(("user", f"Document metadata: {meta}"))
    for idx, chunk in enumerate(chunks, start=1):
        turns.append(("user", f"Document chunk {idx}:\n{chunk}"))
    return timestamped_messages_from_turns(turns)
