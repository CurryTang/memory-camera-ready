"""Generic MemoryArena adapters for repo-owned unified memory methods.

MemoryArena is a streaming benchmark: each subtask/session may only retrieve
state written by earlier sessions, then update memory once the current session
finishes. Most repo methods expose the older build/answer interface over a
fixed trajectory. This module bridges the two protocols by treating completed
session records as the trajectory, rebuilding the method memory after updates,
and using ``method.answer(memory, query)`` as the retrieved memory context.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agentmem.methods import build_method
from agentmem.methods.multi_session import MultiSessionMemory, SessionFeedback

class UnifiedMethodSessionAdapter(MultiSessionMemory):
    """Session-level wrapper around ``agentmem.methods.build_method``.

    The adapter is deliberately conservative:
    - retrieve once at session start;
    - update once at session end;
    - never receives or stores gold answers;
    - fail loudly if the underlying method dependency is unavailable.

    Rebuilding after each update is slower than an incremental native adapter,
    but it preserves a uniform protocol across baselines and is acceptable for
    the initial MemoryArena appendix runs.
    """

    def __init__(
        self,
        method_name: str,
        *,
        method_kwargs: dict[str, Any] | None = None,
        max_chars: int | None = 24000,
    ) -> None:
        self.method_name = method_name
        self.method_kwargs = dict(method_kwargs or {})
        self.max_chars = max_chars
        self.task_id: str | None = None
        self.schema_hint: str | None = None
        self._method: Any = None
        self._records: list[str] = []
        self._memory: Any = None
        self._dirty = True

    def reset(self, task_id: str, *, schema_hint: str | None = None) -> None:
        self.task_id = str(task_id)
        self.schema_hint = schema_hint
        self._method = self._build_method()
        self._records = []
        self._memory = None
        self._dirty = True

    def retrieve(self, query: str, *, session_id: int, k: int = 5) -> str:
        if not self._records:
            return "No prior memory."
        self._ensure_memory()
        context = str(self._method.answer(self._memory, query) or "")
        if not context.strip():
            return "No relevant prior memory retrieved."
        return self._truncate(context)

    def update(self, fb: SessionFeedback) -> None:
        self._records.append(self._format_record(fb))
        self._dirty = True

    def _ensure_memory(self) -> None:
        if not self._dirty and self._memory is not None:
            return
        if self._method is None:
            self._method = self._build_method()
        trajectory = "\n\n".join(self._records)
        task = (
            f"MemoryArena task_id={self.task_id or ''}; "
            f"config={self.schema_hint or 'unknown'}"
        )
        self._memory = self._method.build(trajectory, task=task)
        self._dirty = False

    def _build_method(self) -> Any:
        kwargs = dict(self.method_kwargs)
        while True:
            try:
                return build_method(self.method_name, **kwargs)
            except TypeError as exc:
                match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
                if not match:
                    raise
                bad_key = match.group(1)
                if bad_key not in kwargs:
                    raise
                kwargs.pop(bad_key)

    def _format_record(self, fb: SessionFeedback) -> str:
        signal = {
            k: v
            for k, v in (fb.judge_signal or {}).items()
            if k not in {"gold", "gold_answer", "gold_exact", "gold_asin", "gold_excerpt"}
        }
        lines = [
            f"[Session {fb.session_id}]",
            f"Question: {fb.question}",
            f"Prediction: {fb.prediction}",
            f"Correct: {fb.correct}",
        ]
        if signal:
            lines.append(f"Feedback signal: {signal}")
        if fb.observations:
            lines.append("Observations:")
            lines.extend(f"- {obs}" for obs in fb.observations)
        if fb.trace:
            lines.append("Trace:")
            lines.append(str(fb.trace))
        return "\n".join(lines)

    def _truncate(self, text: str) -> str:
        if self.max_chars is None or len(text) <= self.max_chars:
            return text
        return text[-self.max_chars :]

class SimpleMemAdapter(UnifiedMethodSessionAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("simplemem", **kwargs)

class LightMemAdapter(UnifiedMethodSessionAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("lightmem", **kwargs)

class HippoRAGAdapter(UnifiedMethodSessionAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("hipporag", **kwargs)

    def _format_record(self, fb: SessionFeedback) -> str:
        signal = {
            k: v
            for k, v in (fb.judge_signal or {}).items()
            if k not in {"gold", "gold_answer", "gold_exact", "gold_asin", "gold_excerpt"}
        }
        lines = [
            f"Turn {fb.session_id}",
            f"Question: {fb.question}",
            f"Answer: {fb.prediction}",
            f"Outcome correct: {fb.correct}",
        ]
        if signal:
            lines.append(f"Feedback signal: {signal}")
        if fb.observations:
            lines.append("Observations:")
            lines.extend(f"- {obs}" for obs in fb.observations[:5])
        return "\n".join(lines)

class HippoRAGv2Adapter(HippoRAGAdapter):
    """Alias used by paper/status docs."""

class PlugMemAdapter(UnifiedMethodSessionAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("plugmem", **kwargs)

    def reset(self, task_id: str, *, schema_hint: str | None = None) -> None:
        self.task_id = str(task_id)
        self.schema_hint = schema_hint
        self._method = self._build_method()
        self._records = []
        self._dirty = False

        from agentmem.plugmem.session import PlugMemSession
        from agentmem.plugmem.upstream import PlugMemGraphMemory

        inner = getattr(self._method, "_inner", self._method)
        adapter = inner.adapter
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.task_id or "unknown")
        sample_dir = Path(inner.save_dir) / f"memoryarena_{safe_task_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("episodic_memory", "semantic_memory", "procedural_memory", "tag", "subgoal"):
            (sample_dir / subdir).mkdir(parents=True, exist_ok=True)
        graph = adapter._new_memory_graph(log_file=sample_dir / "plugmem.log")
        session = PlugMemSession(
            session_id=f"memoryarena_{safe_task_id}",
            goal=(
                f"MemoryArena task_id={self.task_id or ''}; "
                f"config={self.schema_hint or 'unknown'}"
            ),
            steps=[],
            metadata={"benchmark": "memoryarena", "config": self.schema_hint or "unknown"},
        )
        self._memory = PlugMemGraphMemory(graph=graph, session=session, sample_dir=sample_dir)

    def update(self, fb: SessionFeedback) -> None:
        if self._method is None or self._memory is None:
            self.reset(self.task_id or "unknown", schema_hint=self.schema_hint)

        from agentmem.plugmem.session import PlugMemSession
        from agentmem.plugmem.upstream import parse_trajectory_steps, plugmem_env

        record = self._format_record(fb)
        self._records.append(record)

        inner = getattr(self._method, "_inner", self._method)
        adapter = inner.adapter
        raw_steps = list(parse_trajectory_steps(record))

        def _keep(step: Any) -> bool:
            action = getattr(step, "action", None)
            observation = getattr(step, "observation", None)
            if isinstance(step, dict):
                action = step.get("action", action)
                observation = step.get("observation", observation)
            return bool((action and str(action).strip()) or (observation and str(observation).strip()))

        steps = [step for step in raw_steps if _keep(step)]
        if not steps:
            return

        session = PlugMemSession(
            session_id=f"memoryarena_{self.task_id}_{fb.session_id}",
            goal=str(self._memory.session.goal or "MemoryArena"),
            steps=steps,
            metadata={"benchmark": "memoryarena", "config": self.schema_hint or "unknown"},
        )
        with plugmem_env(adapter.env_overrides, sample_dir=self._memory.sample_dir):
            self._memory.graph.insert(adapter.build_memory(session))
        self._dirty = False

class AMAAgentAdapter(UnifiedMethodSessionAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("ama_agent", **kwargs)

class MemTAdapter(UnifiedMethodSessionAdapter):
    def __init__(self, **kwargs: Any) -> None:
        method_kwargs = kwargs.pop("method_kwargs", None)
        max_chars = kwargs.pop("max_chars", 24000)
        super().__init__("memt", method_kwargs=method_kwargs, max_chars=max_chars)
        self._engine: Any = None
        self._sample_id: str = ""
        self._memt_inner: Any = None

    def reset(self, task_id: str, *, schema_hint: str | None = None) -> None:
        self.task_id = str(task_id)
        self.schema_hint = schema_hint
        self._method = self._build_method()
        self._records = []
        self._memory = None
        self._dirty = False
        self._sample_id = f"memoryarena_{re.sub(r'[^A-Za-z0-9_]+', '_', self.task_id)}"
        self._memt_inner = getattr(self._method, "_inner", self._method)
        self._engine = self._memt_inner._make_engine()

    def retrieve(self, query: str, *, session_id: int, k: int = 5) -> str:
        if not self._records or self._engine is None:
            return "No prior memory."
        from agentmem.eval.amabench_runner.methods.memt import MEMT_ANSWER_PREFIX, _memt_env

        with _memt_env(
            api_key=self._memt_inner.llm_api_key,
            base_url=self._memt_inner.llm_base_url,
            embedding_model=self._memt_inner.embedding_model,
            embedding_base_url=self._memt_inner.embedding_base_url,
            embedding_api_key=self._memt_inner.embedding_api_key,
        ):
            result = self._engine.retrieve_and_answer(
                query,
                sample_id=self._sample_id,
                category="",
            )
        if isinstance(result, dict):
            answer = result.get("answer", "")
            traces = result.get("traces") or []
        else:
            answer = getattr(result, "answer", "")
            traces = getattr(result, "traces", []) or []
        self._memt_inner.last_trace = list(traces) if isinstance(traces, list) else []
        context = f"{MEMT_ANSWER_PREFIX}{answer}" if answer else ""
        return self._truncate(context or "No relevant prior memory retrieved.")

    def update(self, fb: SessionFeedback) -> None:
        if self._method is None or self._engine is None:
            self.reset(self.task_id or "unknown", schema_hint=self.schema_hint)
        from agentmem.eval.amabench_runner.methods.memt import _memt_env, _traj_text_to_sample

        record = self._format_record(fb)
        self._records.append(record)
        sample = _traj_text_to_sample(
            record,
            (
                f"MemoryArena task_id={self.task_id or ''}; "
                f"config={self.schema_hint or 'unknown'}"
            ),
            self._sample_id,
        )
        with _memt_env(
            api_key=self._memt_inner.llm_api_key,
            base_url=self._memt_inner.llm_base_url,
            embedding_model=self._memt_inner.embedding_model,
            embedding_base_url=self._memt_inner.embedding_base_url,
            embedding_api_key=self._memt_inner.embedding_api_key,
        ):
            self._engine.build_from_sample(sample)

class MemRLAdapter(UnifiedMethodSessionAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("memrl", **kwargs)

class Mem0Adapter(UnifiedMethodSessionAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("mem0", **kwargs)

class MemoryOSAdapter(UnifiedMethodSessionAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("memoryos", **kwargs)

class AMemAdapter(UnifiedMethodSessionAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("amem", **kwargs)

class AutoHarnessAdapter(MultiSessionMemory):
    """Native MemoryArena adapter for the auto-managed memory controller."""

    def __init__(
        self,
        *,
        method_kwargs: dict[str, Any] | None = None,
        max_chars: int | None = 24000,
    ) -> None:
        from agentmem.methods.autoharness import AutoHarnessMemory

        self.method_kwargs = dict(method_kwargs or {})
        self.max_chars = max_chars
        self.task_id: str | None = None
        self.schema_hint: str | None = None
        self._method: Any = None
        self._memory: AutoHarnessMemory | None = None

    def reset(self, task_id: str, *, schema_hint: str | None = None) -> None:
        self.task_id = str(task_id)
        self.schema_hint = schema_hint
        self._method = build_method("autoharness", **self.method_kwargs)
        self._memory = self._method.build(
            "",
            task=(
                f"MemoryArena task_id={self.task_id}; "
                f"config={self.schema_hint or 'unknown'}"
            ),
        )

    def retrieve(self, query: str, *, session_id: int, k: int = 5) -> str:
        if self._method is None or self._memory is None:
            self.reset(self.task_id or "unknown", schema_hint=self.schema_hint)
        context = str(self._method.answer(self._memory, query) or "")
        if self.max_chars is not None and len(context) > self.max_chars:
            context = context[-self.max_chars :]
        return context or "No prior memory."

    def update(self, fb: SessionFeedback) -> None:
        if self._method is None or self._memory is None:
            self.reset(self.task_id or "unknown", schema_hint=self.schema_hint)
        signal = {
            k: v
            for k, v in (fb.judge_signal or {}).items()
            if k not in {"gold", "gold_answer", "gold_exact", "gold_asin", "gold_excerpt"}
        }
        if fb.correct is not None:
            signal.setdefault("correct", fb.correct)
        self._method.update_from_feedback(
            self._memory,
            session_id=fb.session_id,
            question=fb.question,
            prediction=fb.prediction,
            feedback=signal,
            trace=fb.trace,
            observations=fb.observations,
        )

class DCILiteAdapter(MultiSessionMemory):
    """Native MemoryArena adapter for DCI-Agent-Lite style session search."""

    method_name = "dci_lite"

    def __init__(
        self,
        *,
        method_kwargs: dict[str, Any] | None = None,
        max_chars: int | None = 24000,
    ) -> None:
        from agentmem.methods.dci_lite import DCIMemory

        self.method_kwargs = dict(method_kwargs or {})
        self.max_chars = max_chars
        self.task_id: str | None = None
        self.schema_hint: str | None = None
        self._method: Any = None
        self._memory: DCIMemory | None = None

    def reset(self, task_id: str, *, schema_hint: str | None = None) -> None:
        self.task_id = str(task_id)
        self.schema_hint = schema_hint
        self._method = build_method(self.method_name, **self.method_kwargs)
        self._memory = self._method.build(
            "",
            task=(
                f"MemoryArena task_id={self.task_id}; "
                f"config={self.schema_hint or 'unknown'}"
            ),
        )

    def retrieve(self, query: str, *, session_id: int, k: int = 5) -> str:
        if self._method is None or self._memory is None:
            self.reset(self.task_id or "unknown", schema_hint=self.schema_hint)
        context = str(self._method.answer(self._memory, query) or "")
        if self.max_chars is not None and len(context) > self.max_chars:
            context = context[: self.max_chars // 2] + "\n... [truncated] ...\n" + context[-self.max_chars // 2 :]
        return context or "No prior memory."

    def update(self, fb: SessionFeedback) -> None:
        if self._method is None or self._memory is None:
            self.reset(self.task_id or "unknown", schema_hint=self.schema_hint)
        signal = {
            k: v
            for k, v in (fb.judge_signal or {}).items()
            if k not in {"gold", "gold_answer", "gold_exact", "gold_asin", "gold_excerpt"}
        }
        if fb.correct is not None:
            signal.setdefault("correct", fb.correct)
        self._method.update_from_feedback(
            self._memory,
            session_id=fb.session_id,
            question=fb.question,
            prediction=fb.prediction,
            feedback=signal,
            trace=fb.trace,
            observations=fb.observations,
        )

class DCILiteSummarizeAdapter(DCILiteAdapter):
    """MemoryArena adapter for DCI Lite level4 summarizing context management."""

    method_name = "dci_lite_sum"

class AutoMemAdapter(DCILiteAdapter):
    """MemoryArena adapter for AutoMem; inherits the DCI-Lite grep contract."""

    method_name = "automem"
