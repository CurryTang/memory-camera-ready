from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

def parse_reset_seconds(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    total = 0.0
    matches = re.findall(r"([0-9]*\.?[0-9]+)\s*(ms|s|m|h)", text)
    if not matches:
        return None
    for amount, unit in matches:
        scale = {
            "ms": 0.001,
            "s": 1.0,
            "m": 60.0,
            "h": 3600.0,
        }[unit]
        total += float(amount) * scale
    return total

@dataclass
class RequestLease:
    _governor: "RateLimitGovernor"
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._governor.release()

    def __enter__(self) -> "RequestLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

class RateLimitGovernor:
    def __init__(
        self,
        *,
        max_concurrent_requests: int = 8,
        min_remaining_requests_threshold: int = 1,
        min_remaining_tokens_threshold: int = 1024,
        reset_safety_seconds: float = 0.5,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._max_concurrent_requests = max(1, int(max_concurrent_requests))
        self._min_remaining_requests_threshold = max(0, int(min_remaining_requests_threshold))
        self._min_remaining_tokens_threshold = max(0, int(min_remaining_tokens_threshold))
        self._reset_safety_seconds = max(0.0, float(reset_safety_seconds))
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._lock = threading.Lock()
        self._inflight = 0
        self._cooldown_until = 0.0
        self._remaining_requests: Optional[int] = None
        self._remaining_tokens: Optional[int] = None
        self._reset_requests_at: Optional[float] = None
        self._reset_tokens_at: Optional[float] = None

    def before_request(self, estimated_tokens: Optional[int] = None) -> RequestLease:
        while True:
            wait_seconds = 0.0
            with self._lock:
                now = self._time_fn()
                wait_seconds = self._wait_seconds_locked(now, estimated_tokens)
                if wait_seconds <= 0.0 and self._inflight < self._max_concurrent_requests:
                    self._inflight += 1
                    return RequestLease(self)
                if wait_seconds <= 0.0:
                    wait_seconds = 0.01
            self._sleep_fn(wait_seconds)

    def release(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    def update_from_headers(
        self,
        headers: Optional[dict[str, object]],
        *,
        usage: Optional[dict[str, int]] = None,
    ) -> None:
        if not headers:
            return
        now = self._time_fn()
        normalized = {str(k).lower(): v for k, v in headers.items()}
        with self._lock:
            self._remaining_requests = _to_int(normalized.get("x-ratelimit-remaining-requests"), self._remaining_requests)
            self._remaining_tokens = _to_int(normalized.get("x-ratelimit-remaining-tokens"), self._remaining_tokens)
            reset_requests = parse_reset_seconds(normalized.get("x-ratelimit-reset-requests"))
            reset_tokens = parse_reset_seconds(normalized.get("x-ratelimit-reset-tokens"))
            if reset_requests is not None:
                self._reset_requests_at = now + reset_requests
            if reset_tokens is not None:
                self._reset_tokens_at = now + reset_tokens
            if usage and self._remaining_tokens is None:
                total = usage.get("total_tokens")
                if isinstance(total, int):
                    self._remaining_tokens = max(0, total)

    def handle_rate_limit_error(
        self,
        *,
        retry_after: Optional[float] = None,
        headers: Optional[dict[str, object]] = None,
    ) -> None:
        delay = parse_reset_seconds(retry_after)
        if delay is None and headers:
            normalized = {str(k).lower(): v for k, v in headers.items()}
            delay = parse_reset_seconds(normalized.get("retry-after"))
            if delay is None:
                req_delay = parse_reset_seconds(normalized.get("x-ratelimit-reset-requests"))
                tok_delay = parse_reset_seconds(normalized.get("x-ratelimit-reset-tokens"))
                delay = max(x for x in (req_delay, tok_delay, 0.0))
        if delay is None:
            delay = 1.0
        with self._lock:
            self._cooldown_until = max(
                self._cooldown_until,
                self._time_fn() + float(delay) + self._reset_safety_seconds,
            )

    def _wait_seconds_locked(self, now: float, estimated_tokens: Optional[int]) -> float:
        waits: list[float] = []
        if self._cooldown_until > now:
            waits.append(self._cooldown_until - now)
        if (
            self._remaining_requests is not None
            and self._remaining_requests <= self._min_remaining_requests_threshold
            and self._reset_requests_at is not None
            and self._reset_requests_at > now
        ):
            waits.append((self._reset_requests_at - now) + self._reset_safety_seconds)
        token_budget = self._remaining_tokens
        if estimated_tokens is not None and token_budget is not None:
            token_budget = token_budget - max(0, int(estimated_tokens))
        if (
            token_budget is not None
            and token_budget <= self._min_remaining_tokens_threshold
            and self._reset_tokens_at is not None
            and self._reset_tokens_at > now
        ):
            waits.append((self._reset_tokens_at - now) + self._reset_safety_seconds)
        return max(waits) if waits else 0.0

_GOVERNOR_REGISTRY: dict[tuple[str, str, str], RateLimitGovernor] = {}
_GOVERNOR_REGISTRY_LOCK = threading.Lock()

def get_shared_rate_limit_governor(
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_concurrent_requests: int = 8,
    min_remaining_requests_threshold: int = 1,
    min_remaining_tokens_threshold: int = 1024,
    reset_safety_seconds: float = 0.5,
    time_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> RateLimitGovernor:
    key = (str(base_url or ""), str(api_key or ""), str(model or ""))
    with _GOVERNOR_REGISTRY_LOCK:
        governor = _GOVERNOR_REGISTRY.get(key)
        if governor is None:
            governor = RateLimitGovernor(
                max_concurrent_requests=max_concurrent_requests,
                min_remaining_requests_threshold=min_remaining_requests_threshold,
                min_remaining_tokens_threshold=min_remaining_tokens_threshold,
                reset_safety_seconds=reset_safety_seconds,
                time_fn=time_fn,
                sleep_fn=sleep_fn,
            )
            _GOVERNOR_REGISTRY[key] = governor
        return governor

def _to_int(value: object, fallback: Optional[int]) -> Optional[int]:
    if value is None:
        return fallback
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return fallback
