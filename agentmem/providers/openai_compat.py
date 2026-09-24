"""
OpenAI-compatible provider.

Works with OpenAI, local vLLM endpoints, and any service that implements
the OpenAI chat completions API (including Gemini via their compat endpoint).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from agentmem.providers.base import BaseLLMProvider, LLMResponse, Message
from agentmem.providers.key_pool import APIKeyPool
from agentmem.providers.rate_limit import get_shared_rate_limit_governor, parse_reset_seconds
from agentmem.providers.recording import LLMCallCache, RequestHistoryRecorder

import logging as _logging

_logger = _logging.getLogger(__name__)

def lc_variance_active() -> bool:
    """True when running an LC (longcontext) variance sweep, in which case the
    LLM call cache must NOT be populated (so repeated identical prompts re-sample).
    Defaults to False so normal runs keep caching. This symbol is referenced on the
    chat() success path (cache-write guard); it was previously undefined, which
    raised NameError on every completion and silently emptied predictions on code
    paths that route the final answer through this provider (e.g. HotpotQA). Set
    AGENTMEM_LC_VARIANCE=1 to skip cache-writes during variance sweeps.
    """
    import os as _os
    return _os.environ.get("AGENTMEM_LC_VARIANCE", "").strip().lower() in {"1", "true", "yes", "on"}

class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Provider wrapping the OpenAI Python SDK.

    Set base_url to point at a different endpoint (e.g. local vLLM).

    Key rotation: if ``api_key="auto"`` (or omitted when env keys exist),
    all OPENAI_API_KEY* / OPENROUTER_API_KEY* keys are pooled and the provider
    rotates to the next key on rate-limit errors.  You can also pass an
    explicit ``APIKeyPool`` via the ``key_pool`` parameter.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        default_logprobs: bool = False,
        default_top_logprobs: Optional[int] = None,
        default_max_tokens: Optional[int] = None,
        disable_thinking: bool = True,
        max_retries: int = 5,
        retry_backoff_sec: float = 1.0,
        max_concurrent_requests: int = 8,
        min_remaining_requests_threshold: int = 1,
        min_remaining_tokens_threshold: int = 1024,
        reset_safety_seconds: float = 0.5,
        history_dir: Optional[Path | str] = None,
        run_tag: str = "default",
        enable_history: bool = False,
        history_metadata: Optional[dict[str, Any]] = None,
        enable_call_cache: bool = False,
        call_cache_dir: Optional[Path | str] = None,
        time_fn: Any = time.time,
        sleep_fn: Any = time.sleep,
        key_pool: Optional[APIKeyPool] = None,
    ) -> None:
        from openai import OpenAI

        self._key_pool: Optional[APIKeyPool] = key_pool
        if self._key_pool is None and api_key in ("auto", "AUTO", "pool"):
            try:
                self._key_pool = APIKeyPool.from_env(base_url=base_url, time_fn=time_fn)
            except ValueError:
                pass                                                

        if self._key_pool is not None:

            api_key = self._key_pool.acquire()

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self._api_key = api_key
        self._model = model
        self._default_logprobs = bool(default_logprobs)
        self._default_top_logprobs = default_top_logprobs
        self._default_max_tokens = (
            int(default_max_tokens) if default_max_tokens is not None else None
        )
        resolved_base_url = str(getattr(self._client, "base_url", "") or "")
        env_base_url = str(os.getenv("OPENAI_BASE_URL", "") or "")
        base_for_policy = (resolved_base_url or env_base_url).lower()
        self._disable_thinking = bool(disable_thinking)

        self._apply_thinking_control = bool(base_for_policy and "api.openai.com" not in base_for_policy)

        if os.getenv("AGENTMEM_DISABLE_THINKING_KWARGS", "").lower() in {"1", "true", "yes"}:
            self._apply_thinking_control = False
        self._max_retries = max(1, int(max_retries))
        self._retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._base_url = resolved_base_url or str(base_url or env_base_url or "")
        self._governor = get_shared_rate_limit_governor(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model,
            max_concurrent_requests=max_concurrent_requests,
            min_remaining_requests_threshold=min_remaining_requests_threshold,
            min_remaining_tokens_threshold=min_remaining_tokens_threshold,
            reset_safety_seconds=reset_safety_seconds,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
        self._recorder = RequestHistoryRecorder(
            base_dir=history_dir or Path("results/llm_history"),
            run_tag=run_tag,
            enabled=enable_history,
            time_fn=time_fn,
        )
        self._history_metadata = dict(history_metadata or {})
        self._call_cache = LLMCallCache(
            cache_dir=call_cache_dir or Path("results/llm_cache"),
            run_tag=run_tag,
            enabled=enable_call_cache,
        )

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(messages),
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        use_max_tokens = (
            self._default_max_tokens if max_tokens is None else int(max_tokens)
        )
        if use_max_tokens is not None:

            _m = self._model.lower()
            if any(_m.startswith(p) for p in ("gpt-5", "o1", "o2", "o3", "o4")):
                payload["max_completion_tokens"] = int(use_max_tokens)
            else:
                payload["max_tokens"] = int(use_max_tokens)
        if self._disable_thinking and self._apply_thinking_control:

            if "openrouter.ai" not in self._base_url.lower():
                extra_body = payload.get("extra_body")
                if not isinstance(extra_body, dict):
                    extra_body = {}
                template_kwargs = extra_body.get("chat_template_kwargs")
                if not isinstance(template_kwargs, dict):
                    template_kwargs = {}
                template_kwargs.setdefault("enable_thinking", False)
                extra_body["chat_template_kwargs"] = template_kwargs
                payload["extra_body"] = extra_body
        use_logprobs = self._default_logprobs if logprobs is None else bool(logprobs)
        if use_logprobs:
            payload["logprobs"] = True
            use_top = self._default_top_logprobs if top_logprobs is None else top_logprobs
            if use_top is not None:
                payload["top_logprobs"] = int(use_top)
        if kwargs:
            payload.update(kwargs)

        temperature = payload.get("temperature", temperature)

        _cache_key = LLMCallCache.make_key(
            model=self._model,
            messages=payload["messages"],
            temperature=temperature,
            max_tokens=use_max_tokens,
            tools=payload.get("tools"),
        )
        _cached = self._call_cache.get(_cache_key)
        if _cached is not None:
            _resp = _cached.get("response", {})
            _tool_calls = _resp.get("tool_calls") or []
            return LLMResponse(
                content=_resp.get("content"),
                tool_calls=_tool_calls,
                raw=None,
                usage=_cached.get("usage") or {},
            )

        raw = None
        last_error: Optional[Exception] = None
        current_key = self._api_key
        for attempt in range(self._max_retries):
            lease = self._governor.before_request(estimated_tokens=use_max_tokens)
            start = self._time_fn()
            try:
                raw, headers = self._create_chat_completion(payload)
                usage = self._extract_usage(raw)
                self._governor.update_from_headers(headers, usage=usage)
                if self._key_pool is not None:
                    self._key_pool.report_success(current_key)
                self._record_event(
                    event="response",
                    payload=payload,
                    attempt=attempt,
                    raw=raw,
                    headers=headers,
                    usage=usage,
                    error=None,
                    latency_seconds=self._time_fn() - start,
                )
                break
            except Exception as exc:
                last_error = exc
                headers = self._extract_headers_from_error(exc)
                retry_after = self._extract_retry_after_seconds(exc, headers)
                is_rate_limit = self._is_rate_limit_error(exc)
                if self._is_retryable_error(exc):
                    self._governor.handle_rate_limit_error(retry_after=retry_after, headers=headers)
                self._record_event(
                    event="error",
                    payload=payload,
                    attempt=attempt,
                    raw=None,
                    headers=headers,
                    usage=None,
                    error=exc,
                    latency_seconds=self._time_fn() - start,
                )
                if attempt >= self._max_retries - 1 or not self._is_retryable_error(exc):
                    raise

                if is_rate_limit and self._key_pool is not None and self._key_pool.size > 1:
                    self._key_pool.report_rate_limit(current_key, retry_after=retry_after)
                    new_key = self._key_pool.acquire()
                    if new_key != current_key:
                        current_key = new_key
                        self._swap_client_key(new_key)
                        _logger.info("APIKeyPool: rotated to next key after rate limit")
                        continue                                            
                if retry_after is None:
                    self._sleep_fn(self._retry_backoff_sec * (2**attempt))
                else:
                    self._sleep_fn(min(retry_after, 60.0))
            finally:
                lease.release()
        if raw is None:

            raise RuntimeError(f"Chat completion failed after retries: {last_error}")
        choice = raw.choices[0]
        msg = choice.message
        usage = self._extract_usage(raw)
        tool_calls = self._parse_tool_calls(getattr(msg, "tool_calls", None))
        if not tool_calls:
            tool_calls = self._parse_tool_calls_from_content(getattr(msg, "content", None))

        if not lc_variance_active():
            self._call_cache.put(_cache_key, {
                "model": self._model,
                "usage": usage,
                "response": {
                    "content": msg.content,
                    "tool_calls": tool_calls,
                },
            })

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            raw=raw,
            usage=usage,
        )

    def _swap_client_key(self, new_key: str) -> None:
        """Hot-swap the OpenAI client's API key without reconstructing."""
        from openai import OpenAI
        self._api_key = new_key
        client_kwargs: dict[str, Any] = {"api_key": new_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        self._client = OpenAI(**client_kwargs)

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Check if an exception is specifically a rate limit error (429)."""
        if exc.__class__.__name__ == "RateLimitError":
            return True
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        return "rate limit" in str(exc).lower()

    def _create_chat_completion(self, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        completions = self._client.chat.completions
        raw_create = getattr(getattr(completions, "with_raw_response", None), "create", None)
        if callable(raw_create):
            raw_response = raw_create(**payload)
            headers = dict(getattr(raw_response, "headers", {}) or {})
            parsed = raw_response.parse() if hasattr(raw_response, "parse") else raw_response
            return parsed, headers
        parsed = completions.create(**payload)
        return parsed, {}

    def _record_event(
        self,
        *,
        event: str,
        payload: dict[str, Any],
        attempt: int,
        raw: Any,
        headers: dict[str, Any],
        usage: Optional[dict[str, int]],
        error: Optional[Exception],
        latency_seconds: float,
    ) -> None:
        row = {
            "event": event,
            "attempt": int(attempt) + 1,
            "model": self._model,
            "base_url": self._base_url,
            "request": payload,
            "headers": dict(headers or {}),
            "usage": dict(usage or {}),
            "latency_seconds": float(latency_seconds),
            "metadata": dict(self._history_metadata),
        }
        if raw is not None:
            choice = raw.choices[0] if getattr(raw, "choices", None) else None
            message = getattr(choice, "message", None)
            row["response"] = {
                "content": getattr(message, "content", None),
                "tool_calls": self._parse_tool_calls(getattr(message, "tool_calls", None)),
            }
        if error is not None:
            row["error"] = {
                "type": error.__class__.__name__,
                "message": str(error),
                "status_code": getattr(error, "status_code", None),
            }
        self._recorder.record(row)

    @staticmethod
    def _extract_usage(raw: Any) -> dict[str, int]:
        usage = {}
        if getattr(raw, "usage", None) is not None:
            usage = {
                "prompt_tokens": int(getattr(raw.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(raw.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(raw.usage, "total_tokens", 0) or 0),
            }
        return usage

    @staticmethod
    def _extract_headers_from_error(exc: Exception) -> dict[str, Any]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return {}
        return dict(headers)

    @staticmethod
    def _extract_retry_after_seconds(exc: Exception, headers: dict[str, Any]) -> Optional[float]:
        retry_after = parse_reset_seconds(headers.get("retry-after"))
        if retry_after is not None:
            return retry_after
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            retry_after = parse_reset_seconds(body.get("retry_after"))
            if retry_after is not None:
                return retry_after
        return None

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert provider-agnostic Message objects to OpenAI API dicts."""
        out: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "content": message.content or "",
                    "tool_call_id": message.tool_call_id,
                }
                if message.name:
                    tool_msg["name"] = message.name
                out.append(tool_msg)
                continue

            base: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.role == "assistant" and message.tool_calls:
                normalized_calls: list[dict[str, Any]] = []
                for call in message.tool_calls:
                    name = None
                    args: Any = {}
                    if isinstance(call, dict):
                        if "function" in call and isinstance(call["function"], dict):
                            name = call["function"].get("name")
                            args = call["function"].get("arguments", {})
                        else:
                            name = call.get("name")
                            args = call.get("arguments", {})
                    if isinstance(args, str):
                        args_json = args
                    else:
                        args_json = json.dumps(args, ensure_ascii=False)
                    normalized_calls.append(
                        {
                            "id": call.get("id") if isinstance(call, dict) else None,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args_json,
                            },
                        }
                    )
                base["tool_calls"] = normalized_calls
            out.append(base)
        return out

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
        """Parse OpenAI tool_calls into a list of {name, arguments, id} dicts."""
        if not raw_tool_calls:
            return []
        parsed: list[dict[str, Any]] = []
        for call in raw_tool_calls:
            if hasattr(call, "model_dump"):
                call_dict = call.model_dump()
            elif isinstance(call, dict):
                call_dict = call
            else:
                call_dict = {}
            function = call_dict.get("function", {}) or {}
            raw_args = function.get("arguments", {})
            args: Any
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            parsed.append(
                {
                    "id": call_dict.get("id"),
                    "name": function.get("name"),
                    "arguments": args,
                }
            )
        return parsed

    @staticmethod
    def _parse_tool_calls_from_content(content: Any) -> list[dict[str, Any]]:
        """
        Fallback parser for models that emit textual tool calls:
        <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        """
        if not isinstance(content, str) or "<tool_call>" not in content:
            return []
        matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
        parsed: list[dict[str, Any]] = []
        for idx, block in enumerate(matches):
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                continue
            args = payload.get("arguments") or {}
            if not isinstance(args, dict):
                args = {"_raw": str(args)}
            name = payload.get("name")
            if not name:
                continue
            parsed.append(
                {
                    "id": payload.get("id") or f"text_tool_call_{idx}",
                    "name": str(name),
                    "arguments": args,
                }
            )
        return parsed

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        name = exc.__class__.__name__
        if name in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
            "ServiceUnavailableError",
        }:
            return True
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "connection error",
                "timed out",
                "timeout",
                "temporarily unavailable",
                "rate limit",
                "server error",
                "502",
                "503",
                "504",
            )
        )
