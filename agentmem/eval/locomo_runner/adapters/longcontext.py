"""Long-context baseline adapter — feeds entire conversation to the LLM."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from agentmem.providers.openai_compat import OpenAICompatibleProvider
from agentmem.providers.base import Message
from agentmem.eval.locomo_runner.prompts import _build_locomo_answer_prompt

class LongContextLoCoMoAdapter:
    """Long-context baseline: feed the entire conversation to the LLM.

    Uses the same ``_build_locomo_answer_prompt`` as C1-C9 for consistency.
    No retrieval, no system prompt — just the user message with full context.
    """

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        max_context_tokens: int = 128000,
        provider_kwargs: Optional[dict[str, Any]] = None,
        **_extra: Any,
    ) -> None:
        self._answer_provider = OpenAICompatibleProvider(
            api_key=(answer_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"),
            model=answer_model,
            base_url=answer_base_url,
            **dict(provider_kwargs or {}),
        )
        self._max_context_tokens = max_context_tokens
        self._turns: list[str] = []

    def reset(self) -> None:
        self._turns = []

    def observe(self, content: str, timestamp: Optional[str] = None) -> None:
        self._turns.append(content)

    def finalize(self) -> None:
        pass

    def ask(self, question: str, category: Optional[int] = None) -> str:
        import time as _time
        retrieve_start = _time.perf_counter()
        context = "\n".join(self._turns)
        chars_per_token = 4
        max_chars = self._max_context_tokens * chars_per_token
        if len(context) > max_chars:
            half = max_chars // 2
            context = (
                context[:half]
                + "\n\n... [middle section truncated] ...\n\n"
                + context[-half:]
            )
        retrieve_seconds = _time.perf_counter() - retrieve_start
        prompt = _build_locomo_answer_prompt(
            question=question, context=context, category=category
        )
        msgs = [Message(role="user", content=prompt)]
        llm_start = _time.perf_counter()
        resp = self._answer_provider.chat(msgs)
        llm_seconds = _time.perf_counter() - llm_start
        usage = dict(getattr(resp, "usage", None) or {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
        self._last_resource_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)),
            "retrieved_context_tokens": prompt_tokens,                                                 
            "retrieval_calls": 0,
            "llm_calls": 1,
            "retrieve_seconds": float(retrieve_seconds),
            "llm_seconds": float(llm_seconds),
            "latency_seconds": float(retrieve_seconds + llm_seconds),
        }
        if usage:
            self._last_resource_usage["usage"] = usage
        return resp.content or ""

    def last_trajectory(self) -> Optional[Dict[str, Any]]:
        return {"method": "longcontext", "num_turns": len(self._turns)}

    def canonical_efficiency(self) -> dict:
        from agentmem.eval.locomo_runner.adapters.base import canonical_efficiency_from_usage
        return canonical_efficiency_from_usage(
            getattr(self, "_build_resource_usage", None),
            getattr(self, "_last_resource_usage", None),
        )

    def shutdown(self) -> None:
        self._turns = []
