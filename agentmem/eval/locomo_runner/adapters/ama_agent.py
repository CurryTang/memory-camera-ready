"""AMA-Agent adapter for LoCoMo evaluation.

Uses AMA-Agent's full pipeline (causal graph construction + self-eval
escalation retrieval) but answers questions via the shared
``_build_locomo_answer_prompt`` for fair comparison with other methods.

AMA-Agent was designed for agent trajectories with action/observation pairs.
For LoCoMo dialogue, this adapter converts each dialogue turn into a
pseudo-trajectory turn:
  - Action: "dialogue_turn" (analogous to PlugMem's PlugMemStep conversion)
  - Observation: the actual dialogue content

AMA-Agent's memory construction, causal graph building, and multi-stage
retrieval (embedding → sufficiency check → tool search) are unchanged —
only the final context→answer step is standardized.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from agentmem.providers.openai_compat import OpenAICompatibleProvider
from agentmem.providers.base import Message
from agentmem.eval.locomo_runner.prompts import _build_locomo_answer_prompt

class AMAAgentLoCoMoAdapter:
    """AMA-Agent adapter using the shared LoCoMo answer prompt.

    Follows the same observe/finalize/ask pattern as other adapters.
    AMA-Agent builds its causal graph (env/object state nodes with
    causal/temporal edges) from dialogue turns converted to trajectory
    format, retrieves via embedding + self-eval escalation, then passes
    the retrieved context through ``_build_locomo_answer_prompt`` for
    consistent evaluation.
    """

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        retrieval_mode: str = "embed",
        top_k: int = 5,
        neighbor_radius: int = 0,
        chunk_size: int = 8192,
        causal: bool = True,
        enable_tools: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        provider_kwargs: Optional[dict[str, Any]] = None,
        **_extra: Any,
    ) -> None:

        self._answer_provider = OpenAICompatibleProvider(
            api_key=(answer_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"),
            model=answer_model,
            base_url=answer_base_url,
            **dict(provider_kwargs or {}),
        )

        from agentmem.eval.amabench_runner.methods.ama_agent import AMAAgentMethod

        self._ama_agent = AMAAgentMethod(
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            neighbor_radius=neighbor_radius,
            chunk_size=chunk_size,
            causal=causal,
            enable_tools=enable_tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._turns: list[dict[str, Any]] = []
        self._memory: Optional[dict[str, Any]] = None
        self._last_trajectory: Optional[dict[str, Any]] = None
        self._sample_counter = 0

    def reset(self) -> None:
        self._turns = []
        self._memory = None
        self._last_trajectory = None
        self._sample_counter += 1

    def observe(self, content: str, timestamp: Optional[str] = None) -> None:
        self._turns.append({"content": content, "timestamp": timestamp})

    def _dialogue_to_trajectory_text(self) -> str:
        """Convert dialogue turns to trajectory text for AMA-Agent.

        AMA-Agent's ``_parse_trajectory_text`` expects lines like::

            Turn 0:
            Action: ...
            Observation: ...

        For dialogue, each turn becomes:
          - Action: ``dialogue_turn`` (with speaker if parseable)
          - Observation: the dialogue content
        """
        lines: list[str] = []
        for i, turn in enumerate(self._turns):
            content = turn["content"]

            speaker = "Unknown"
            text = content
            if ": " in content:
                maybe_speaker, _, maybe_text = content.partition(": ")
                if len(maybe_speaker) < 40:
                    speaker = maybe_speaker.strip()
                    text = maybe_text.strip()

            timestamp_str = ""
            if turn.get("timestamp"):
                timestamp_str = f" [{turn['timestamp']}]"

            lines.append(f"Turn {i}:")
            lines.append(f"Action: dialogue_turn by {speaker}{timestamp_str}")
            lines.append(f"Observation: {text}")
            lines.append("")                        

        return "\n".join(lines)

    def finalize(self) -> None:
        """Convert dialogue to trajectory format and build AMA-Agent's causal graph."""
        trajectory_text = self._dialogue_to_trajectory_text()
        self._memory = self._ama_agent.memory_construction(
            traj_text=trajectory_text,
            task="Multi-session conversational memory",
        )

    def ask(self, question: str, category: Optional[int] = None) -> str:
        import time as _time
        from agentmem.eval.resource_metrics import estimate_text_tokens
        if self._memory is None:
            self.finalize()

        retrieve_start = _time.perf_counter()
        context = self._ama_agent.memory_retrieve(
            memory=self._memory,
            question=question,
        )
        retrieve_seconds = _time.perf_counter() - retrieve_start
        self._last_trajectory = self._ama_agent.last_trajectory()

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
            "retrieved_context_tokens": estimate_text_tokens(context or "", model=self._answer_provider.model),
            "retrieval_calls": 1,
            "llm_calls": 1,
            "retrieve_seconds": float(retrieve_seconds),
            "llm_seconds": float(llm_seconds),
            "latency_seconds": float(retrieve_seconds + llm_seconds),
        }
        if usage:
            self._last_resource_usage["usage"] = usage
        return resp.content or ""

    def last_trajectory(self) -> Optional[Dict[str, Any]]:
        return self._last_trajectory

    def canonical_efficiency(self) -> dict:
        from agentmem.eval.locomo_runner.adapters.base import canonical_efficiency_from_usage
        return canonical_efficiency_from_usage(
            getattr(self, "_build_resource_usage", None),
            getattr(self, "_last_resource_usage", None),
        )

    def shutdown(self) -> None:
        self._turns = []
        self._memory = None
        self._last_trajectory = None
