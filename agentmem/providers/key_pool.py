"""
APIKeyPool — round-robin + rate-limit-aware key rotation for OpenAI-compatible APIs.

Loads all OPENAI_API_KEY, OPENAI_API_KEY2, OPENAI_API_KEY3, ... and
OPENROUTER_API_KEY, OPENROUTER_API_KEY2, ... from environment.

When a key hits a rate limit, the pool marks it as cooling down and
rotates to the next available key. Thread-safe.

Usage:
    pool = APIKeyPool.from_env()                      # auto-discover keys
    pool = APIKeyPool(keys=["sk-...", "sk-..."])       # explicit keys
    key = pool.acquire()                               # get best available key
    pool.report_rate_limit(key, retry_after=60.0)      # mark key as cooling
    pool.report_success(key)                            # mark key as healthy
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

@dataclass
class _KeyState:
    key: str
    label: str                                                 
    cooldown_until: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0

class APIKeyPool:
    """Thread-safe pool of API keys with rate-limit-aware rotation."""

    def __init__(
        self,
        keys: list[str],
        labels: Optional[list[str]] = None,
        time_fn: Callable[[], float] = time.time,
        default_cooldown_sec: float = 30.0,
    ) -> None:
        if not keys:
            raise ValueError("APIKeyPool requires at least one key")
        labels = labels or [f"key_{i}" for i in range(len(keys))]
        self._states = [_KeyState(key=k, label=l) for k, l in zip(keys, labels)]
        self._lock = threading.Lock()
        self._next_idx = 0
        self._time_fn = time_fn
        self._default_cooldown = default_cooldown_sec

    @classmethod
    def from_env(
        cls,
        prefixes: tuple[str, ...] = ("OPENAI_API_KEY", "OPENROUTER_API_KEY"),
        base_url: Optional[str] = None,
        time_fn: Callable[[], float] = time.time,
        default_cooldown_sec: float = 30.0,
    ) -> "APIKeyPool":
        """Auto-discover API keys from environment variables.

        Scans OPENAI_API_KEY, OPENAI_API_KEY2, OPENAI_API_KEY3, ...
        and OPENROUTER_API_KEY, OPENROUTER_API_KEY2, ... etc.

        If base_url contains 'openrouter', only OPENROUTER_ keys are used.
        If base_url contains 'openai.com', only OPENAI_ keys are used.
        Otherwise all keys are collected.
        """
        keys: list[str] = []
        labels: list[str] = []

        active_prefixes = list(prefixes)
        if base_url:
            url_lower = base_url.lower()
            if "openrouter" in url_lower:
                active_prefixes = [p for p in prefixes if "OPENROUTER" in p]
            elif "openai.com" in url_lower:
                active_prefixes = [p for p in prefixes if "OPENAI" in p]

        for prefix in active_prefixes:

            val = os.environ.get(prefix)
            if val and val.strip():
                keys.append(val.strip())
                labels.append(prefix)

            for i in range(2, 20):
                env_name = f"{prefix}{i}"
                val = os.environ.get(env_name)
                if val and val.strip():
                    keys.append(val.strip())
                    labels.append(env_name)

        if not keys:

            for fallback in ("OPENAI_KEY", "OPENAI_API_KEY"):
                val = os.environ.get(fallback)
                if val and val.strip():
                    keys.append(val.strip())
                    labels.append(fallback)
                    break

        if not keys:
            raise ValueError(
                f"No API keys found in environment. "
                f"Set one of: {', '.join(prefixes)}"
            )

        logger.info("APIKeyPool: loaded %d keys: %s", len(keys), ", ".join(labels))
        return cls(keys=keys, labels=labels, time_fn=time_fn,
                   default_cooldown_sec=default_cooldown_sec)

    @property
    def size(self) -> int:
        return len(self._states)

    def acquire(self) -> str:
        """Get the best available API key.

        Tries round-robin, skipping keys that are in cooldown.
        If all keys are in cooldown, waits for the soonest one to become available.
        """
        with self._lock:
            now = self._time_fn()
            n = len(self._states)

            for offset in range(n):
                idx = (self._next_idx + offset) % n
                state = self._states[idx]
                if state.cooldown_until <= now:
                    self._next_idx = (idx + 1) % n
                    state.total_requests += 1
                    return state.key

            soonest = min(self._states, key=lambda s: s.cooldown_until)
            wait_time = soonest.cooldown_until - now
            if wait_time > 0:
                logger.info(
                    "APIKeyPool: all %d keys in cooldown, waiting %.1fs for %s",
                    n, wait_time, soonest.label,
                )

        if wait_time > 0:
            time.sleep(wait_time)

        with self._lock:
            soonest.total_requests += 1
            soonest.consecutive_errors = 0
            return soonest.key

    def report_rate_limit(self, key: str, retry_after: Optional[float] = None) -> None:
        """Mark a key as rate-limited. It will be skipped until cooldown expires."""
        with self._lock:
            state = self._find_state(key)
            if state is None:
                return
            cooldown = retry_after if retry_after and retry_after > 0 else self._default_cooldown

            state.consecutive_errors += 1
            if state.consecutive_errors > 1:
                cooldown = min(cooldown * (1.5 ** (state.consecutive_errors - 1)), 120.0)
            state.cooldown_until = self._time_fn() + cooldown
            state.total_errors += 1
            logger.debug(
                "APIKeyPool: %s rate-limited, cooldown %.1fs (consecutive=%d)",
                state.label, cooldown, state.consecutive_errors,
            )

    def report_success(self, key: str) -> None:
        """Mark a key as healthy after a successful request."""
        with self._lock:
            state = self._find_state(key)
            if state is None:
                return
            state.consecutive_errors = 0

    def get_stats(self) -> list[dict]:
        """Return stats for all keys (for debugging)."""
        with self._lock:
            now = self._time_fn()
            return [
                {
                    "label": s.label,
                    "total_requests": s.total_requests,
                    "total_errors": s.total_errors,
                    "cooling": s.cooldown_until > now,
                    "cooldown_remaining": max(0, s.cooldown_until - now),
                }
                for s in self._states
            ]

    def _find_state(self, key: str) -> Optional[_KeyState]:
        for state in self._states:
            if state.key == key:
                return state
        return None
