"""Auto-managed memory/query method.

``AutoHarnessMethod`` is an experimental memory controller for the paper's
next-step hypothesis: memory layout and retrieval queries should be managed by
the model itself, rather than by a hand-written domain schema. It keeps a
generic store of session records, asks an optional OpenAI-compatible controller
LLM to compress and retrieve from that store, and uses reward-like feedback to
evolve a compact policy note over time.

The implementation intentionally avoids benchmark-specific prompts. The
controller sees only generic session text, observable feedback, and a bounded
operation schema: summarize, select records, rewrite query, update policy.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agentmem.methods.base import BaseMethod, MeteredMethod, MethodKind

@dataclass
class AutoHarnessRecord:
    """One memory item managed by the recursive controller."""

    record_id: str
    session_id: int | None
    content: str
    summary: str = ""
    feedback_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class AutoHarnessMemory:
    """Opaque memory handle returned by :meth:`AutoHarnessMethod.build`."""

    task: str
    records: list[AutoHarnessRecord] = field(default_factory=list)
    global_summary: str = ""
    policy_note: str = (
        "Prefer compact factual state that was useful for previous questions; "
        "ignore unsupported speculation and irrelevant traces."
    )
    feedback_window: list[dict[str, Any]] = field(default_factory=list)

class AutoHarnessMethod(MeteredMethod, BaseMethod):
    """Recursive LLM-managed memory controller.

    The method is usable without a controller endpoint: it falls back to a
    deterministic lexical retriever and extractive summaries, which keeps unit
    tests and offline debugging cheap. With ``llm_base_url`` configured, the
    same API asks the model to recursively compress records and plan retrieval.
    """

    kind = MethodKind.STRUCTURED
    name = "autoharness"

    def __init__(
        self,
        *,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_records: int = 48,
        top_k: int = 8,
        recency_k: int = 3,
        max_context_chars: int = 24000,
        recursive_depth: int = 2,
        evolve_every: int = 3,
        controller_max_tokens: int = 256,
        temperature: float = 0.0,
        request_timeout: float = 60.0,
        save_dir: str | None = None,
        token_counter: Callable[[str], int] | None = None,
        **_kw: Any,
    ) -> None:
        super().__init__()
        self.llm_model = llm_model or model
        self.llm_base_url = (llm_base_url or base_url or "").rstrip("/")
        self.llm_api_key = llm_api_key or api_key or "EMPTY"
        self.max_records = int(max_records)
        self.top_k = int(top_k)
        self.recency_k = max(0, int(recency_k))
        self.max_context_chars = int(max_context_chars)
        self.recursive_depth = max(0, int(recursive_depth))
        self.evolve_every = max(1, int(evolve_every))
        self.controller_max_tokens = max(64, int(controller_max_tokens))
        self.temperature = float(temperature)
        self.request_timeout = float(request_timeout)
        self.save_dir = Path(save_dir) if save_dir else None
        self._token_counter = token_counter

    def build(self, traj_text: str, *, task: str = "") -> AutoHarnessMemory:
        with self._counters.time_block("build_wallclock_seconds"):
            records = self._records_from_text(traj_text)
            memory = AutoHarnessMemory(task=str(task or ""), records=records)
            self._refresh_global_summary(memory, phase="build")
            self._trim(memory)
        self._persist(memory)
        return memory

    def memory_construction(self, traj_text: str, task: str = "") -> AutoHarnessMemory:
        return self.build(traj_text, task=task)

    def answer(self, memory: AutoHarnessMemory, question: str) -> str:
        start = time.perf_counter()
        if not memory.records and not memory.global_summary:
            return "No prior memory."
        plan = self._plan_retrieval(memory, question)
        selected = self._select_records(memory, question, plan)
        context = self._render_context(memory, question, selected, plan)
        elapsed = time.perf_counter() - start
        self._counters.add_seconds("w_tool", elapsed)
        self._counters.record_retrieval(
            candidates_scored=len(memory.records),
            evidence_injected=len(selected),
            context_tokens=self._count_tokens(context),
        )
        return context

    def memory_retrieve(self, memory: AutoHarnessMemory, question: str) -> str:
        return self.answer(memory, question)

    def update_from_feedback(
        self,
        memory: AutoHarnessMemory,
        *,
        session_id: int,
        question: str,
        prediction: str,
        feedback: dict[str, Any] | None = None,
        trace: str | None = None,
        observations: list[str] | None = None,
    ) -> None:
        """Append an observable session and let feedback adjust the policy note."""

        signal = self._sanitize_feedback(feedback or {})
        record = AutoHarnessRecord(
            record_id=f"s{session_id}",
            session_id=session_id,
            content=self._format_session(
                session_id=session_id,
                question=question,
                prediction=prediction,
                feedback=signal,
                trace=trace,
                observations=observations or [],
            ),
            summary="",
            feedback_score=self._feedback_score(signal),
            metadata={"feedback": signal},
        )
        memory.records.append(record)
        memory.feedback_window.append(
            {
                "session_id": session_id,
                "question": question,
                "correct": signal.get("correct"),
                "score": record.feedback_score,
            }
        )
        self._trim(memory)
        self._refresh_global_summary(memory, phase="build")
        if len(memory.feedback_window) % self.evolve_every == 0:
            self._evolve_policy(memory)
        self._persist(memory)

    def persistent_store_bytes(self, memory: AutoHarnessMemory) -> int:
        if not self.save_dir or not self.save_dir.exists():
            return 0
        return sum(p.stat().st_size for p in self.save_dir.rglob("*") if p.is_file())

    def _records_from_text(self, text: str) -> list[AutoHarnessRecord]:
        chunks = self._split_sessions(text)
        records: list[AutoHarnessRecord] = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            records.append(
                AutoHarnessRecord(
                    record_id=f"r{i}",
                    session_id=self._extract_session_id(chunk),
                    content=chunk.strip(),
                    summary=self._cheap_summary(chunk),
                )
            )
        return records[-self.max_records :]

    def _split_sessions(self, text: str) -> list[str]:
        text = str(text or "").strip()
        if not text:
            return []
        parts = re.split(r"(?=\n?\[(?:Session|Turn)\s+\d+\])", text, flags=re.I)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return parts
        chunk_chars = max(1200, self.max_context_chars // 8)
        return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]

    def _extract_session_id(self, text: str) -> int | None:
        match = re.search(r"(?:Session|Turn)\s+(\d+)", text, flags=re.I)
        return int(match.group(1)) if match else None

    def _refresh_global_summary(self, memory: AutoHarnessMemory, *, phase: str) -> None:
        if not memory.records:
            memory.global_summary = ""
            return
        summaries = [r.summary or self._cheap_summary(r.content) for r in memory.records]
        memory.global_summary = self._recursive_compress(summaries, memory.task, phase=phase)

    def _recursive_compress(self, items: list[str], task: str, *, phase: str) -> str:
        items = [i.strip() for i in items if i and i.strip()]
        if not items:
            return ""
        current = items
        depth = 0
        while len(current) > 1 and depth <= self.recursive_depth:
            grouped = [
                "\n\n".join(current[i : i + 6])
                for i in range(0, len(current), 6)
            ]
            next_level = []
            for group in grouped:
                prompt = (
                    "You are a memory controller. Compress the records into a "
                    "domain-agnostic state summary for future questions.\n"
                    f"Task label/context:\n{task or 'unknown'}\n\n"
                    f"Records:\n{group}\n\n"
                    "Return concise facts, constraints, decisions, and unresolved state. "
                    "Do not invent details."
                )
                next_level.append(self._llm_text(prompt, phase=phase) or self._cheap_summary(group))
            current = next_level
            depth += 1
        return self._truncate(current[0] if current else "")

    def _plan_retrieval(
        self,
        memory: AutoHarnessMemory,
        question: str,
    ) -> dict[str, Any]:
        catalog = [
            {
                "id": r.record_id,
                "session_id": r.session_id,
                "summary": r.summary or self._cheap_summary(r.content),
                "feedback_score": r.feedback_score,
            }
            for r in memory.records[-self.max_records :]
        ]
        prompt = (
            "You are an automatic memory/query controller. Choose which prior "
            "records are useful for the current question and optionally rewrite "
            "the retrieval query. Use only generic memory-management criteria.\n\n"
            f"Policy note:\n{memory.policy_note}\n\n"
            f"Global memory summary:\n{memory.global_summary or 'None'}\n\n"
            f"Current question:\n{question}\n\n"
            f"Record catalog JSON:\n{json.dumps(catalog, ensure_ascii=False)}\n\n"
            "Return only JSON with keys: query, focus, selected_ids, rationale."
        )
        raw = self._llm_text(prompt, phase="query")
        plan = self._parse_json(raw)
        if isinstance(plan, dict):
            return plan
        return {
            "query": question,
            "focus": "lexical fallback",
            "selected_ids": [],
            "rationale": "controller unavailable",
        }

    def _select_records(
        self,
        memory: AutoHarnessMemory,
        question: str,
        plan: dict[str, Any],
    ) -> list[AutoHarnessRecord]:
        by_id = {r.record_id: r for r in memory.records}
        selected: list[AutoHarnessRecord] = []
        for rid in plan.get("selected_ids") or []:
            rec = by_id.get(str(rid))
            if rec and rec not in selected:
                selected.append(rec)
        for rec in memory.records[-self.recency_k :]:
            if rec not in selected:
                selected.append(rec)
        if len(selected) >= self.top_k:
            return selected[: self.top_k]
        query = str(plan.get("query") or question)
        ranked = sorted(
            memory.records,
            key=lambda r: (
                self._lexical_score(query, r.content + "\n" + (r.summary or "")),
                r.feedback_score if r.feedback_score is not None else 0.0,
                r.session_id if r.session_id is not None else -1,
            ),
            reverse=True,
        )
        for rec in ranked:
            if rec not in selected:
                selected.append(rec)
            if len(selected) >= self.top_k:
                break
        return selected[: self.top_k]

    def _render_context(
        self,
        memory: AutoHarnessMemory,
        question: str,
        selected: list[AutoHarnessRecord],
        plan: dict[str, Any],
    ) -> str:
        lines = [
            "Auto-managed recursive memory:",
            f"Policy: {memory.policy_note}",
            f"Global state: {memory.global_summary or 'None'}",
            f"Controller query: {plan.get('query') or question}",
            f"Controller focus: {plan.get('focus') or 'None'}",
            "Selected records:",
        ]
        for rec in selected:
            lines.append(
                f"[{rec.record_id} | session={rec.session_id} | reward={rec.feedback_score}]\n"
                f"{rec.summary or self._cheap_summary(rec.content)}\n"
                f"{rec.content}"
            )
        return self._truncate("\n\n".join(lines))

    def _evolve_policy(self, memory: AutoHarnessMemory) -> None:
        recent = memory.feedback_window[-self.evolve_every :]
        prompt = (
            "You are improving a generic memory policy from rewarded feedback. "
            "Update the policy note for future retrieval and write-back. "
            "Do not specialize to a named benchmark or leak answer patterns.\n\n"
            f"Current policy:\n{memory.policy_note}\n\n"
            f"Recent feedback JSON:\n{json.dumps(recent, ensure_ascii=False)}\n\n"
            "Return one concise policy note."
        )
        evolved = self._llm_text(prompt, phase="build")
        if evolved:
            memory.policy_note = self._truncate(evolved, limit=1200)
        else:
            successes = [x for x in recent if x.get("correct") is True or float(x.get("score") or 0) > 0.5]
            failures = [x for x in recent if x.get("correct") is False or float(x.get("score") or 0) <= 0.5]
            memory.policy_note = (
                "Prefer records with direct overlap to the current question and "
                "with positive feedback; when recent feedback is weak, include "
                "the latest unresolved constraints and avoid over-compression."
                f" Recent positives={len(successes)}, weak={len(failures)}."
            )

    def _format_session(
        self,
        *,
        session_id: int,
        question: str,
        prediction: str,
        feedback: dict[str, Any],
        trace: str | None,
        observations: list[str],
    ) -> str:
        parts = [
            f"[Session {session_id}]",
            f"Question: {question}",
            f"Prediction: {prediction}",
            f"Feedback: {json.dumps(feedback, ensure_ascii=False, sort_keys=True)}",
        ]
        if observations:
            parts.append("Observations:\n" + "\n".join(f"- {o}" for o in observations[:8]))
        if trace:
            parts.append("Trace:\n" + self._truncate(str(trace), limit=4000))
        return "\n".join(parts)

    def _sanitize_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        blocked = {"gold", "gold_answer", "gold_exact", "gold_asin", "gold_excerpt", "answer"}
        out = {k: v for k, v in feedback.items() if k not in blocked}
        if "correct" not in out and "score" not in out:
            for key in ("progress", "soft_progress", "soft_progress_score"):
                if key in out:
                    out["score"] = out[key]
                    break
        return out

    def _feedback_score(self, feedback: dict[str, Any]) -> float | None:
        if "score" in feedback:
            try:
                return float(feedback["score"])
            except (TypeError, ValueError):
                return None
        if "correct" in feedback:
            return 1.0 if bool(feedback["correct"]) else 0.0
        return None

    def _trim(self, memory: AutoHarnessMemory) -> None:
        if len(memory.records) > self.max_records:
            memory.records = memory.records[-self.max_records :]

    def _cheap_summary(self, text: str, *, limit: int = 700) -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(clean) <= limit:
            return clean
        return clean[: limit - 3].rstrip() + "..."

    def _lexical_score(self, query: str, text: str) -> float:
        q_terms = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
        t_terms = set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
        if not q_terms or not t_terms:
            return 0.0
        return len(q_terms & t_terms) / max(1, len(q_terms))

    def _llm_text(self, prompt: str, *, phase: str) -> str:
        if not self.llm_base_url or not self.llm_model:
            return ""
        url = self.llm_base_url
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": "/no_think\n" + prompt}],
            "temperature": self.temperature,
            "max_tokens": self.controller_max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}",
            },
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return ""
        elapsed = time.perf_counter() - start
        if phase == "build":
            self._counters.add_seconds("w_llm_build", elapsed)
        else:
            self._counters.add_seconds("w_llm", elapsed)
        usage = body.get("usage") or {}
        self._counters.record_llm_call(
            prompt_tokens=int(usage.get("prompt_tokens") or self._count_tokens(prompt)),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            phase="build" if phase == "build" else "query",
        )
        choices = body.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "").strip()

    def _parse_json(self, text: str) -> Any:
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def _truncate(self, text: str, *, limit: int | None = None) -> str:
        max_len = int(limit or self.max_context_chars)
        text = str(text or "")
        if len(text) <= max_len:
            return text
        return text[: max_len // 2] + "\n\n... [auto-memory truncated] ...\n\n" + text[-max_len // 2 :]

    def _count_tokens(self, text: str) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0
        return len(str(text or "").split())

    def _persist(self, memory: AutoHarnessMemory) -> None:
        if not self.save_dir:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": memory.task,
            "global_summary": memory.global_summary,
            "policy_note": memory.policy_note,
            "feedback_window": memory.feedback_window,
            "records": [
                {
                    "record_id": r.record_id,
                    "session_id": r.session_id,
                    "content": r.content,
                    "summary": r.summary,
                    "feedback_score": r.feedback_score,
                    "metadata": r.metadata,
                }
                for r in memory.records
            ],
        }
        (self.save_dir / "autoharness_state.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
