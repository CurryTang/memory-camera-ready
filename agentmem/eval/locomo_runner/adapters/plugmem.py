"""PlugMem adapter for LoCoMo evaluation.

Uses PlugMem's heterogeneous KG and retrieval modes, with a LoCoMo-specific
conversation construction path and the shared ``_build_locomo_answer_prompt``
for fair answer generation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

from agentmem.providers.openai_compat import OpenAICompatibleProvider
from agentmem.providers.base import Message
from agentmem.eval.locomo_runner.prompts import _build_locomo_answer_prompt, _parse_prefixed_dialogue

class PlugMemLoCoMoAdapter:
    """PlugMem adapter using the shared LoCoMo answer prompt.

    Follows the same observe/finalize/ask pattern as other adapters.
    PlugMem builds a heterogeneous knowledge graph (semantic, episodic,
    procedural, tag, subgoal nodes) from session-grouped dialogue turns,
    retrieves via all modes, then passes the combined retrieved context through
    ``_build_locomo_answer_prompt`` for consistent evaluation.
    """

    def __init__(
        self,
        answer_model: str = "Qwen/Qwen3-32B",
        answer_api_key: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        plugmem_source_root: Optional[str] = None,
        retrieval_topk: int = 15,
        memory_modes: Optional[Sequence[str]] = None,
        env_overrides: Optional[Mapping[str, Optional[str]]] = None,
        save_dir: Optional[str] = None,
        provider_kwargs: Optional[dict[str, Any]] = None,
        **_extra: Any,
    ) -> None:
        self._answer_provider = OpenAICompatibleProvider(
            api_key=(answer_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"),
            model=answer_model,
            base_url=answer_base_url,
            **dict(provider_kwargs or {}),
        )
        self._plugmem_source_root = plugmem_source_root
        self._retrieval_topk = max(1, retrieval_topk)
        self._memory_modes = list(memory_modes) if memory_modes else None
        self._env_overrides = dict(env_overrides or {})
        if "QWEN_MODEL_NAME" not in self._env_overrides:
            from agentmem.eval.amabench_runner.methods.plugmem import _plugmem_qwen_model_name

            self._env_overrides["QWEN_MODEL_NAME"] = _plugmem_qwen_model_name(answer_model)
        self._save_dir = save_dir or os.path.join(tempfile.gettempdir(), "plugmem_locomo")
        self._turns: list[dict[str, Any]] = []
        self._adapter: Any = None
        self._memory: Any = None
        self._sample_counter = 0

    def reset(self) -> None:
        self._turns = []
        self._adapter = None
        self._memory = None
        self._sample_counter += 1

    def observe(self, content: str, timestamp: Optional[str] = None) -> None:
        self._turns.append({"content": content, "timestamp": timestamp})

    @staticmethod
    def _parse_observed_turn(turn: Mapping[str, Any]) -> dict[str, Any]:
        content = str(turn.get("content") or "").strip()
        observed_timestamp = turn.get("timestamp")
        speaker, parsed_timestamp, text = _parse_prefixed_dialogue(content)
        timestamp = str(observed_timestamp or parsed_timestamp or "").strip() or None

        if speaker == "Unknown" and ": " in text:
            maybe_speaker, _, maybe_text = text.partition(": ")
            if 0 < len(maybe_speaker.strip()) < 40:
                speaker = maybe_speaker.strip()
                text = maybe_text.strip()

        return {
            "speaker": speaker or "Unknown",
            "text": text,
            "timestamp": timestamp,
        }

    def _group_turns_by_session(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        active_timestamp: Optional[str] = None
        active: Optional[dict[str, Any]] = None

        for raw_turn in self._turns:
            parsed = self._parse_observed_turn(raw_turn)
            timestamp = parsed.get("timestamp")
            if active is None or (timestamp and timestamp != active_timestamp):
                active_timestamp = timestamp
                active = {
                    "session_key": f"session_{len(sessions)}",
                    "timestamp": timestamp,
                    "turns": [],
                }
                sessions.append(active)

            parsed["turn_num"] = len(active["turns"])
            active["turns"].append(parsed)

        return sessions

    @staticmethod
    def _dialogue_text(turns: Sequence[Mapping[str, Any]]) -> str:
        lines = []
        for idx, turn in enumerate(turns):
            speaker = str(turn.get("speaker") or "Unknown").strip() or "Unknown"
            text = str(turn.get("text") or "").strip()
            timestamp = str(turn.get("timestamp") or "").strip()
            time_part = f" [time={timestamp}]" if timestamp else ""
            lines.append(f"Turn {turn.get('turn_num', idx)}{time_part}: {speaker}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _windowed(turns: Sequence[Mapping[str, Any]], size: int = 12) -> list[list[Mapping[str, Any]]]:
        size = max(1, int(size))
        return [list(turns[start:start + size]) for start in range(0, len(turns), size)]

    def _ensure_adapter(self) -> Any:
        if self._adapter is not None:
            return self._adapter
        from agentmem.plugmem.upstream import PlugMemUpstreamAdapter, default_plugmem_source_root
        from pathlib import Path

        source_root = Path(self._plugmem_source_root) if self._plugmem_source_root else default_plugmem_source_root()
        kwargs: dict[str, Any] = {
            "source_root": source_root,
            "retrieval_topk": self._retrieval_topk,
            "env_overrides": self._env_overrides,
        }
        if self._memory_modes is not None:
            kwargs["memory_modes"] = self._memory_modes
        self._adapter = PlugMemUpstreamAdapter(**kwargs)
        return self._adapter

    def finalize(self) -> None:

        from agentmem.plugmem.session import PlugMemSession, PlugMemStep
        from agentmem.plugmem.upstream import PlugMemGraphMemory, plugmem_env

        adapter = self._ensure_adapter()

        steps: list[PlugMemStep] = []
        sessions = self._group_turns_by_session()
        for session in sessions:
            session_key = session["session_key"]
            for turn in session["turns"]:
                global_index = len(steps)
                steps.append(
                    PlugMemStep(
                        index=global_index,
                        speaker=str(turn["speaker"]),
                        action="dialogue_turn",
                        observation=str(turn["text"]),
                        timestamp=str(turn["timestamp"]) if turn.get("timestamp") else None,
                        metadata={"session_key": session_key, "turn_index": turn["turn_num"]},
                    )
                )

        memory_payload: dict[str, Any] = {
            "goal": "Multi-session conversation memory",
            "episodic": [],
            "semantic": [],
            "procedural": [],
        }
        memory_embeddings: dict[str, Any] = {"semantic": [], "procedural": []}
        previous_summaries: list[str] = []

        sample_dir_path = Path(os.path.join(
            self._save_dir, f"sample_{self._sample_counter:04d}"
        )).resolve()
        sample_dir_path.mkdir(parents=True, exist_ok=True)
        for subdir in ("episodic_memory", "semantic_memory", "procedural_memory", "tag", "subgoal"):
            (sample_dir_path / subdir).mkdir(exist_ok=True)

        import sys as _sys
        _vendor_src = str(Path(self._plugmem_source_root or "vendor/plugmem/src").resolve())
        if _vendor_src not in _sys.path or _sys.path[0] != _vendor_src:

            _sys.path = [p for p in _sys.path if p != _vendor_src]
            _sys.path.insert(0, _vendor_src)

        for _modname in ("utils", "memory_structuring", "memory_structuring.structuring_inference"):
            _sys.modules.pop(_modname, None)

        with plugmem_env(adapter.env_overrides, sample_dir=sample_dir_path):
            from memory_structuring.structuring_inference import (
                get_conversation_semantic,
                get_session_summary,
                get_speaker_stance,
                get_topic,
            )

            for session_index, grouped_session in enumerate(sessions):
                session_key = grouped_session["session_key"]
                session_turns = grouped_session["turns"]
                timestamp = grouped_session.get("timestamp") or session_index
                dialogue = self._dialogue_text(session_turns)
                previous_summary_text = "\n\n".join(previous_summaries) or "No previous sessions."

                topic = get_topic(session_turns, previous_summary=previous_summary_text, mode="conversation")
                summary = get_session_summary(previous_summary_text, session_turns, mode="conversation")

                window_stances: list[dict[str, Any]] = []
                for window in self._windowed(session_turns, size=12):
                    first_turn_num = int(window[0].get("turn_num", 0)) if window else 0
                    window_stances.extend(
                        get_speaker_stance(
                            window,
                            trajectory_num=session_index,
                            turn_num=first_turn_num,
                            time=timestamp,
                            mode="conversation",
                        )
                    )

                semantic_items = get_conversation_semantic(
                    session_turns,
                    trajectory_num=session_index,
                    time=timestamp,
                    session_context=f"Topic: {topic}\nPrevious summaries:\n{previous_summary_text}",
                    window_size=12,
                )

                for stance in window_stances:
                    stance_text = str(stance.get("speaker_stance") or "").strip()
                    if not stance_text:
                        continue
                    semantic_items.append(
                        {
                            "semantic_memory": stance_text,
                            "tags": list(stance.get("tags") or []),
                            "trajectory_num": session_index,
                            "turn_num": int(stance.get("turn_num", 0) or 0),
                            "time": timestamp,
                            "st_ed": "mid",
                        }
                    )

                stance_text = "; ".join(
                    str(item.get("speaker_stance") or "").strip()
                    for item in window_stances[:4]
                    if str(item.get("speaker_stance") or "").strip()
                )
                trajectory = []
                for turn in session_turns:
                    speaker = str(turn.get("speaker") or "Unknown").strip() or "Unknown"
                    text = str(turn.get("text") or "").strip()
                    trajectory.append(
                        {
                            "subgoal": topic,
                            "state": summary,
                            "observation": f"{speaker}: {text}",
                            "action": "dialogue_turn",
                            "reward": stance_text or "Conversation turn contributes factual or social context.",
                            "similarity_subgoal": -1,
                            "time": timestamp,
                        }
                    )

                memory_payload["episodic"].append(trajectory)
                memory_payload["semantic"].extend(semantic_items)
                memory_payload["procedural"].append(
                    {
                        "subgoal": topic,
                        "procedural_memory": (
                            f"Session {session_key}"
                            f"{' at ' + str(timestamp) if timestamp else ''}.\n"
                            f"Topic: {topic}\n"
                            f"Summary: {summary}\n"
                            f"Transcript:\n{dialogue}"
                        ),
                        "trajectory_num": session_index,
                        "time": timestamp,
                        "return": 0,
                    }
                )
                previous_summaries.append(f"{session_key}: {summary}")

        session = PlugMemSession(
            session_id=f"locomo_sample_{self._sample_counter:04d}",
            goal="Multi-session conversation memory",
            steps=steps,
            questions=[],
            metadata={"benchmark": "locomo"},
        )
        memory_obj = SimpleNamespace(
            time=0,
            observation_t0="Multi-session conversation begins.",
            memory=memory_payload,
            memory_embedding=memory_embeddings,
        )

        graph = adapter._new_memory_graph(log_file=sample_dir_path / "plugmem.log")
        with plugmem_env(adapter.env_overrides, sample_dir=sample_dir_path):
            graph.insert(adapter._normalize_memory(memory_obj))
        self._memory = PlugMemGraphMemory(graph=graph, session=session, sample_dir=sample_dir_path)

    def ask(self, question: str, category: Optional[int] = None) -> str:
        import time as _time
        from agentmem.eval.resource_metrics import estimate_text_tokens
        from agentmem.plugmem.upstream import plugmem_env
        if self._memory is None:
            self.finalize()

        adapter = self._ensure_adapter()
        sample_path = getattr(self._memory, "sample_dir", None)
        retrieve_start = _time.perf_counter()
        with plugmem_env(adapter.env_overrides, sample_dir=sample_path):
            retrieval = adapter.retrieve(
                self._memory,
                question,
                question_meta={"category": category},
            )
        retrieve_seconds = _time.perf_counter() - retrieve_start
        context = retrieval.combined_context()

        self._last_trajectory = {
            "contexts": dict(retrieval.contexts),
            "errors": dict(retrieval.errors),
            "modes": list(retrieval.contexts.keys()),
        }

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
        return getattr(self, "_last_trajectory", None)

    def canonical_efficiency(self) -> dict:
        from agentmem.eval.locomo_runner.adapters.base import canonical_efficiency_from_usage
        return canonical_efficiency_from_usage(
            getattr(self, "_build_resource_usage", None),
            getattr(self, "_last_resource_usage", None),
        )

    def shutdown(self) -> None:
        self._turns = []
        self._adapter = None
        self._memory = None
