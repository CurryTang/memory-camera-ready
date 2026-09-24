"""Thin OpenAI-compatible model client for AMAbench evaluation.

Supports any OpenAI-compatible endpoint (sglang, vllm, OpenRouter, etc.)
via base_url + api_key.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from model output (Qwen3 thinking mode).

    Handles both closed blocks and unclosed blocks (model hit max_tokens mid-thought).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()

from openai import OpenAI

class ModelClient:
    """Unified client that wraps an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        extra_body: Optional[dict] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens

        if extra_body is None and base_url and "api.openai.com" not in (base_url or ""):
            if "openrouter" in (base_url or "").lower():
                extra_body = {"reasoning": {"exclude": True}}
            else:

                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.extra_body = extra_body or {}

        api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

        self.last_usage: dict = {}

    def query(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 5,
    ) -> str:
        max_tokens = max_tokens or self.max_tokens
        debug_timing = os.getenv("AMABENCH_DEBUG_TIMING") == "1"
        for attempt in range(max_retries):
            try:
                if debug_timing:
                    print(
                        f"[amabench-debug] llm start model={self.model} "
                        f"prompt_chars={len(prompt)} max_tokens={max_tokens} "
                        f"temperature={temperature}",
                        flush=True,
                    )
                t0 = time.perf_counter()
                kwargs: dict = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content = message.content

                if not content:
                    content = getattr(message, "reasoning", None) or ""
                content = _strip_thinking(content or "")
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.last_usage = {
                        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    }
                else:
                    self.last_usage = {}
                if debug_timing:
                    print(
                        f"[amabench-debug] llm done model={self.model} "
                        f"latency_sec={time.perf_counter() - t0:.2f} "
                        f"response_chars={len(content)}",
                        flush=True,
                    )
                return content
            except Exception as exc:
                if debug_timing:
                    print(
                        f"[amabench-debug] llm error model={self.model} "
                        f"attempt={attempt + 1}/{max_retries} exc={exc!r}",
                        flush=True,
                    )
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt, 30)
                    print(f"  [retry {attempt+1}/{max_retries}] {exc!r} — waiting {wait}s")
                    time.sleep(wait)
                else:
                    raise
        return ""               
