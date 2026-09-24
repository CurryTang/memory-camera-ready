"""Compatibility imports for zjunlp LightMem installs.

Some environments use the PyPI/site-packages copy of LightMem while older local
environments may still have an editable ``vendor/lightmem`` entry. Keep the
adapter stable by resolving the same ``LightMemory`` API from whichever install
actually exposes it.
"""

from __future__ import annotations

import importlib
import inspect
import os
import site
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_LIGHTMEM_CLASS_CANDIDATES = (
    ("lightmem.memory.lightmem", "LightMemory"),
    ("lightmem.memory", "LightMemory"),
    ("lightmem", "LightMemory"),
    ("lightmem.lightmem", "LightMemory"),
    ("lightmem", "LightMem"),
    ("lightmem.lightmem", "LightMem"),
    ("lightmem", "Memory"),
)

_ADD_METHOD_CANDIDATES = (
    "add_memory",
    "add_messages",
    "add_message",
    "ingest",
    "store",
    "add",
    "remember",
)

_RETRIEVE_METHOD_CANDIDATES = (
    "retrieve",
    "search",
    "query",
    "recall_relevant",
    "recall",
)

def _is_candidate_module_miss(exc: ImportError, module_name: str) -> bool:
    missing = getattr(exc, "name", "") or ""
    if not missing:
        return True
    return module_name == missing or module_name.startswith(f"{missing}.")

def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    repo_vendor_src = Path(__file__).resolve().parents[2] / "vendor" / "lightmem" / "src"
    if repo_vendor_src.exists():
        roots.append(repo_vendor_src.resolve())
    for raw in list(sys.path) + list(getattr(site, "getsitepackages", lambda: [])()):
        if not raw:
            continue
        try:
            root = Path(raw).resolve()
        except OSError:
            continue
        if root not in roots:
            roots.append(root)
    return roots

def _purge_lightmem_modules() -> None:
    for name in list(sys.modules):
        if name == "lightmem" or name.startswith("lightmem."):
            del sys.modules[name]

def _import_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)

def _call_with_supported_kwargs(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return fn(*args, **kwargs)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in parameters
    }
    return fn(*args, **accepted)

def _maybe_await(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    if loop.is_running():
        raise RuntimeError("LightMem async API cannot be driven from a running event loop")
    return loop.run_until_complete(value)

def _first_method(obj: Any, names: Sequence[str]) -> Callable[..., Any] | None:
    for name in names:
        candidate = getattr(obj, name, None)
        if callable(candidate):
            return candidate
    return None

def _looks_like_native_lightmemory(cls: Any) -> bool:
    return callable(getattr(cls, "from_config", None)) and callable(getattr(cls, "retrieve", None))

def _extract_endpoint_config(config: Mapping[str, Any]) -> dict[str, Any]:
    llm_cfg = ((config.get("memory_manager") or {}).get("configs") or {})
    emb_cfg = ((config.get("text_embedder") or {}).get("configs") or {})
    retriever_cfg = ((config.get("embedding_retriever") or {}).get("configs") or {})
    return {
        "config": dict(config),
        "config_dict": dict(config),
        "llm_model": llm_cfg.get("model"),
        "model": llm_cfg.get("model"),
        "model_name": llm_cfg.get("model"),
        "llm_base_url": llm_cfg.get("openai_base_url") or llm_cfg.get("base_url"),
        "base_url": llm_cfg.get("openai_base_url") or llm_cfg.get("base_url"),
        "api_base_url": llm_cfg.get("openai_base_url") or llm_cfg.get("base_url"),
        "openai_base_url": llm_cfg.get("openai_base_url") or llm_cfg.get("base_url"),
        "llm_api_key": llm_cfg.get("api_key"),
        "api_key": llm_cfg.get("api_key"),
        "embedding_model": emb_cfg.get("model"),
        "embed_model": emb_cfg.get("model"),
        "embedding_base_url": emb_cfg.get("openai_base_url") or emb_cfg.get("base_url"),
        "embedding_api_key": emb_cfg.get("api_key"),
        "embedding_dims": emb_cfg.get("embedding_dims") or retriever_cfg.get("embedding_model_dims"),
        "embedding_dim": emb_cfg.get("embedding_dims") or retriever_cfg.get("embedding_model_dims"),
        "storage_path": retriever_cfg.get("path"),
        "path": retriever_cfg.get("path"),
        "persist_dir": retriever_cfg.get("path"),
        "collection_name": retriever_cfg.get("collection_name"),
    }

def _set_endpoint_env(config: Mapping[str, Any]) -> None:
    endpoint = _extract_endpoint_config(config)
    for env_name, key in (
        ("OPENAI_API_KEY", "api_key"),
        ("OPENAI_BASE_URL", "base_url"),
        ("LIGHTMEM_LLM_MODEL", "llm_model"),
        ("LIGHTMEM_EMBEDDING_MODEL", "embedding_model"),
        ("LIGHTMEM_EMBEDDING_BASE_URL", "embedding_base_url"),
        ("LIGHTMEM_EMBEDDING_API_KEY", "embedding_api_key"),
    ):
        value = endpoint.get(key)
        if value and not os.environ.get(env_name):
            os.environ[env_name] = str(value)

def _messages_to_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, Mapping):
        return str(messages.get("content") or messages)
    if isinstance(messages, Sequence):
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, Mapping):
                role = str(msg.get("role") or "user")
                content = str(msg.get("content") or "").strip()
                if content:
                    parts.append(f"{role}: {content}")
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(messages or "")

def _normalize_results(results: Any) -> list[str]:
    results = _maybe_await(results)
    if results is None:
        return []
    if isinstance(results, str):
        return [results]
    if isinstance(results, Mapping):
        for key in ("memories", "results", "items", "data"):
            value = results.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return _normalize_results(value)
        for key in ("text", "memory", "content", "value"):
            value = results.get(key)
            if value:
                return [str(value)]
        return [str(results)]
    if isinstance(results, Sequence):
        normalized: list[str] = []
        for item in results:
            if isinstance(item, Mapping):
                text = (
                    item.get("memory")
                    or item.get("text")
                    or item.get("content")
                    or item.get("value")
                    or item.get("Memory")
                )
                normalized.append(str(text if text is not None else item))
            else:
                normalized.append(str(item))
        return [item for item in normalized if item.strip()]
    return [str(results)]

class _PyPILightMemoryAdapter:
    """Expose the repo's LightMemory surface over lighter PyPI APIs."""

    _target_cls: Any = None

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self._impl = self._construct_impl(dict(config))

    @classmethod
    def bind(cls, target_cls: Any) -> type["_PyPILightMemoryAdapter"]:
        return type(
            f"{getattr(target_cls, '__name__', 'PyPILightMem')}Compat",
            (cls,),
            {"_target_cls": target_cls},
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "_PyPILightMemoryAdapter":
        return cls(config)

    def _construct_impl(self, config: Mapping[str, Any]) -> Any:
        target_cls = self._target_cls
        _set_endpoint_env(config)
        endpoint = _extract_endpoint_config(config)
        attempts = (
            lambda: target_cls.from_config(config),
            lambda: target_cls(config),
            lambda: _call_with_supported_kwargs(target_cls, **endpoint),
            lambda: target_cls(),
        )
        last_error: BaseException | None = None
        for attempt in attempts:
            try:
                return attempt()
            except (AttributeError, TypeError, ValueError, KeyError) as exc:
                last_error = exc
        raise TypeError(f"Could not instantiate PyPI LightMem API {target_cls!r}") from last_error

    def add_memory(
        self,
        messages: Any,
        METADATA_GENERATE_PROMPT: Any = None,
        *,
        force_segment: bool = False,
        force_extract: bool = False,
    ) -> Any:
        method = _first_method(self._impl, _ADD_METHOD_CANDIDATES)
        if method is None:
            raise AttributeError(
                f"{type(self._impl).__name__} exposes no compatible add/ingest method"
            )
        kwargs = {
            "messages": messages,
            "message": messages,
            "data": _messages_to_text(messages),
            "text": _messages_to_text(messages),
            "content": _messages_to_text(messages),
            "METADATA_GENERATE_PROMPT": METADATA_GENERATE_PROMPT,
            "force_segment": force_segment,
            "force_extract": force_extract,
        }
        try:
            return _maybe_await(_call_with_supported_kwargs(method, **kwargs))
        except TypeError:
            return _maybe_await(method(_messages_to_text(messages)))

    def retrieve(self, query: str, limit: int = 10, filters: Mapping[str, Any] | None = None) -> list[str]:
        method = _first_method(self._impl, _RETRIEVE_METHOD_CANDIDATES)
        if method is None:
            raise AttributeError(
                f"{type(self._impl).__name__} exposes no compatible retrieve/search method"
            )
        kwargs = {
            "query": query,
            "text": query,
            "question": query,
            "limit": limit,
            "k": limit,
            "top_k": limit,
            "filters": filters,
        }
        try:
            return _normalize_results(_call_with_supported_kwargs(method, **kwargs))
        except TypeError:
            return _normalize_results(method(query))

    def offline_update(self, memory_list: list[Any], *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._impl, "offline_update", None)
        if callable(method):
            return method(memory_list, *args, **kwargs)
        return None

    def construct_update_queue_all_entries(self, *args: Any, **kwargs: Any) -> None:
        method = getattr(self._impl, "construct_update_queue_all_entries", None)
        if callable(method):
            method(*args, **kwargs)

    def offline_update_all_entries(self, *args: Any, **kwargs: Any) -> None:
        method = getattr(self._impl, "offline_update_all_entries", None)
        if callable(method):
            method(*args, **kwargs)

def _compatible_lightmemory_cls(cls: Any) -> Any:
    if _looks_like_native_lightmemory(cls):
        return cls
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        signature = None
    classish = inspect.isclass(cls) or signature is not None
    if not classish:
        return cls
    return _PyPILightMemoryAdapter.bind(cls)

def _load_from_current_path() -> Any:
    errors: list[BaseException] = []
    for module_name, attr_name in _LIGHTMEM_CLASS_CANDIDATES:
        try:
            return _compatible_lightmemory_cls(_import_attr(module_name, attr_name))
        except ImportError as exc:
            errors.append(exc)
        except AttributeError as exc:
            errors.append(exc)
    if errors:
        raise errors[-1]
    raise ImportError("No LightMem import candidates configured")

def _load_from_package_root(root: Path) -> Any:
    sys.path.insert(0, str(root))
    try:
        _purge_lightmem_modules()
        return _load_from_current_path()
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass

def load_lightmemory_cls() -> Any:
    """Return the upstream ``LightMemory`` class from vendor or PyPI."""
    first_error: BaseException | None = None
    try:
        return _load_from_current_path()
    except (ImportError, AttributeError) as exc:
        first_error = exc

    for root in _candidate_roots():
        package_dir = root / "lightmem"
        if not package_dir.exists():
            continue
        try:
            return _load_from_package_root(root)
        except (ImportError, AttributeError) as exc:
            first_error = exc

    raise ImportError(
        "No compatible LightMemory class was found in lightmem.memory.lightmem, "
        "lightmem.memory, lightmem, or lightmem.lightmem."
    ) from first_error

def load_memory_entry_cls() -> Any:
    """Return LightMem's ``MemoryEntry`` helper for direct-entry insertion."""
    try:
        return _import_attr("lightmem.memory.utils", "MemoryEntry")
    except (ImportError, AttributeError):
        load_lightmemory_cls()
        return _import_attr("lightmem.memory.utils", "MemoryEntry")
