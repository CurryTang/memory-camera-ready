from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

class RequestHistoryRecorder:
    def __init__(
        self,
        *,
        base_dir: Path | str,
        run_tag: str = "default",
        enabled: bool = False,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._run_tag = str(run_tag)
        self._enabled = bool(enabled)
        self._time_fn = time_fn
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._base_dir / self._run_tag / "calls.jsonl"

    def record(self, row: dict[str, Any]) -> Optional[Path]:
        if not self._enabled:
            return None
        payload = dict(row)
        payload.setdefault("recorded_at", self._time_fn())
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        return path

def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)

class LLMCallCache:
    """Persistent cache for LLM API calls keyed by (model, messages, params) hash.

    Stores cached responses in a JSONL file so that identical calls on restart
    return the cached response without hitting the API.

    Cache key: SHA-256 of canonical JSON of (model, messages, temperature,
    max_tokens, tools).  Only the first 24 hex chars are used as the key.

    Thread-safe for concurrent reads; writes are serialised by a lock.
    """

    def __init__(
        self,
        *,
        cache_dir: Path | str,
        run_tag: str = "default",
        enabled: bool = False,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._run_tag = str(run_tag)
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._store: dict[str, dict[str, Any]] = {}
        if self._enabled:
            self._load()

    @property
    def path(self) -> Path:
        return self._cache_dir / self._run_tag / "llm_cache.jsonl"

    def _load(self) -> None:
        p = self.path
        if not p.exists():
            return
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = entry.get("cache_key")
                    if key:
                        self._store[key] = entry
                except Exception:
                    pass

    @staticmethod
    def make_key(
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[list[dict[str, Any]]],
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools or [],
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:24]

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Return cached response dict or None on miss."""
        if not self._enabled:
            return None
        return self._store.get(key)

    def put(self, key: str, response: dict[str, Any]) -> None:
        """Store a response and append to the cache file."""
        if not self._enabled:
            return
        entry = {"cache_key": key, **response}
        with self._lock:
            self._store[key] = entry
            p = self.path
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")

    @property
    def size(self) -> int:
        return len(self._store)
