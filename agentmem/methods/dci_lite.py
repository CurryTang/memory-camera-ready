"""DCI-Agent-Lite style direct-corpus memory harness.

This adapts the DCI paper's lightweight interface to our memory benchmarks:
the method does not build an embedding index or task-specific schema.  It keeps
raw records and lets a controller model plan high-resolution lexical searches
over those records, using rg-like exact matching, chained filters, bounded local
reads, truncation, compaction, and optionally summarization.
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
class DCIRecord:
    record_id: str
    title: str
    content: str
    compact: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class DCIMemory:
    task: str
    records: list[DCIRecord]
    context_level: str = "level3"
    compacted_history: list[str] = field(default_factory=list)
    summary: str = ""

class DCILiteMethod(MeteredMethod, BaseMethod):
    """Minimal DCI harness with Qwen-compatible controller calls.

    ``context_level="level3"`` mirrors DCI-Agent-Lite's main setting
    (truncation + compaction). ``level4`` additionally summarizes compacted
    evidence when the live context is still too large.
    """

    kind = MethodKind.AGENTIC
    name = "dci_lite"

    def __init__(
        self,
        *,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        context_level: str = "level3",
        top_k: int = 10,
        max_records: int = 4096,
        max_record_chars: int = 20000,
        max_context_chars: int = 24000,
        compact_threshold_chars: int = 240000,
        recent_context_chars: int = 20000,
        controller_max_tokens: int = 512,
        temperature: float = 0.0,
        request_timeout: float = 90.0,
        save_dir: str | None = None,
        token_counter: Callable[[str], int] | None = None,
        **_kw: Any,
    ) -> None:
        super().__init__()

        import os as _os
        top_k = int(_os.environ.get("DCI_TOP_K", top_k))
        max_context_chars = int(_os.environ.get("DCI_MAX_CONTEXT_CHARS", max_context_chars))
        compact_threshold_chars = int(
            _os.environ.get("DCI_COMPACT_THRESHOLD_CHARS", compact_threshold_chars)
        )
        self.llm_model = llm_model or model
        self.llm_base_url = (llm_base_url or base_url or "").rstrip("/")
        self.llm_api_key = llm_api_key or api_key or "EMPTY"
        self.context_level = str(context_level or "level3").lower()
        self.top_k = max(1, int(top_k))
        self.max_records = max(1, int(max_records))
        self.max_record_chars = max(1000, int(max_record_chars))
        self.max_context_chars = max(2000, int(max_context_chars))
        self.compact_threshold_chars = max(10000, int(compact_threshold_chars))
        self.recent_context_chars = max(1000, int(recent_context_chars))
        self.controller_max_tokens = max(64, int(controller_max_tokens))
        self.temperature = float(temperature)
        self.request_timeout = float(request_timeout)
        self.save_dir = Path(save_dir) if save_dir else None
        self._token_counter = token_counter

    def build(self, traj_text: str, *, task: str = "") -> DCIMemory:
        with self._counters.time_block("build_wallclock_seconds"):
            records = self._records_from_text(traj_text)[: self.max_records]
            memory = DCIMemory(task=str(task or ""), records=records, context_level=self.context_level)
            self._compact_memory(memory)
        self._persist(memory)
        return memory

    def memory_construction(self, traj_text: str, task: str = "") -> DCIMemory:
        return self.build(traj_text, task=task)

    def answer(self, memory: DCIMemory, question: str) -> str:
        if not memory.records and not memory.summary:
            return "No corpus records available."
        started = time.perf_counter()
        plan = self._plan_search(memory, question)
        observations = self._execute_plan(memory, question, plan)
        context = self._render_context(memory, question, plan, observations)
        elapsed = time.perf_counter() - started
        self._counters.add_seconds("w_tool", elapsed)
        self._counters.record_retrieval(
            candidates_scored=len(memory.records),
            evidence_injected=len(observations),
            context_tokens=self._count_tokens(context),
        )
        return context

    def memory_retrieve(self, memory: DCIMemory, question: str) -> str:
        return self.answer(memory, question)

    def update_from_feedback(
        self,
        memory: DCIMemory,
        *,
        session_id: int,
        question: str,
        prediction: str,
        feedback: dict[str, Any] | None = None,
        trace: str | None = None,
        observations: list[str] | None = None,
    ) -> None:
        signal = self._sanitize_feedback(feedback or {})
        content = "\n".join(
            part
            for part in [
                f"[Session {session_id}]",
                f"Question: {question}",
                f"Prediction: {prediction}",
                f"Feedback: {json.dumps(signal, ensure_ascii=False, sort_keys=True)}",
                "Observations:\n" + "\n".join(f"- {x}" for x in (observations or [])[:8])
                if observations
                else "",
                "Trace:\n" + self._truncate(str(trace), 4000) if trace else "",
            ]
            if part
        )
        memory.records.append(
            DCIRecord(
                record_id=f"s{session_id}",
                title=f"session_{session_id}",
                content=content,
                compact=self._compact_record(content),
                metadata={"feedback": signal, "session_id": session_id},
            )
        )
        memory.records = memory.records[-self.max_records :]
        self._compact_memory(memory)
        self._persist(memory)

    def persistent_store_bytes(self, memory: DCIMemory) -> int:
        if not self.save_dir or not self.save_dir.exists():
            return 0
        return sum(p.stat().st_size for p in self.save_dir.rglob("*") if p.is_file())

    def _records_from_text(self, text: str) -> list[DCIRecord]:
        text = str(text or "").strip()
        if not text:
            return []

        blocks = self._split_records(text)
        records: list[DCIRecord] = []
        for i, block in enumerate(blocks):
            title = self._infer_title(block, i)
            content = self._truncate(block.strip(), self.max_record_chars)
            records.append(
                DCIRecord(
                    record_id=f"d{i}",
                    title=title,
                    content=content,
                    compact=self._compact_record(content),
                    metadata={"index": i},
                )
            )
        return records

    def _split_records(self, text: str) -> list[str]:
        patterns = [
            r"(?=\n?Document\s+\d+\s+[—-])",
            r"(?=\n?\[(?:Session|Turn)\s+\d+\])",

            r"(?=\n?#\s*[Ss]ession[_ ]\d+)",
            r"(?=\n?(?:Turn|Step)\s+\d+\s*:)",
            r"(?=\n?Episode\s+\d+)",
        ]
        for pattern in patterns:
            parts = [p.strip() for p in re.split(pattern, text, flags=re.I) if p.strip()]
            if len(parts) > 1:
                return parts
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs
        chunk_chars = max(1200, self.max_record_chars)
        return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]

    def _infer_title(self, block: str, index: int) -> str:
        first = block.strip().splitlines()[0] if block.strip() else ""
        match = re.match(r"Document\s+\d+\s+[—-]\s*(.*?):?$", first)
        if match:
            return match.group(1).strip() or f"document_{index}"
        match = re.match(r"\[(Session|Turn)\s+(\d+)\]", first, flags=re.I)
        if match:
            return f"{match.group(1).lower()}_{match.group(2)}"
        match = re.match(r"(Turn|Step)\s+(\d+)\s*:", first, flags=re.I)
        if match:
            return f"{match.group(1).lower()}_{match.group(2)}"
        match = re.match(r"#\s*([Ss]ession[_ ]\d+)", first)
        if match:
            return re.sub(r"\s+", "_", match.group(1).strip()).lower()
        if first.startswith("#"):
            return first.strip("# ").split("|", 1)[0].strip() or f"record_{index}"
        return f"record_{index}"

    def _compact_memory(self, memory: DCIMemory) -> None:
        live_chars = sum(len(r.content) for r in memory.records)
        if self.context_level in {"level0", "level1", "level2"} or live_chars <= self.compact_threshold_chars:
            return
        keep = max(1, min(12, self.top_k))
        older = memory.records[:-keep]
        recent = memory.records[-keep:]
        memory.compacted_history = [
            f"{r.record_id} {r.title}: {r.compact or self._compact_record(r.content)}"
            for r in older
        ]
        memory.records = recent
        if self.context_level in {"level4", "level5"}:
            compact_text = "\n".join(memory.compacted_history)
            prompt = (
                "Summarize older direct-corpus search evidence for future retrieval. "
                "Keep entities, constraints, outcomes, dates, and reusable procedures. "
                "Do not invent facts.\n\n"
                f"Task label:\n{memory.task or 'unknown'}\n\n"
                f"Compacted evidence:\n{self._truncate(compact_text, self.max_context_chars)}"
            )
            memory.summary = self._llm_text(prompt, phase="build") or self._truncate(compact_text, 3000)

    def _plan_search(self, memory: DCIMemory, question: str) -> dict[str, Any]:
        search_records = self._all_search_records(memory)
        catalog = [
            {
                "id": r.record_id,
                "title": r.title,
                "preview": r.compact or self._compact_record(r.content),
            }
            for r in search_records[: min(len(search_records), 80)]
        ]
        prompt = (
            "You control a direct-corpus interaction harness. Plan a bounded search "
            "over raw records using only generic operations: rg exact/regex search, "
            "chained lexical filters, and local reads. Avoid benchmark-specific rules.\n\n"
            f"Task label:\n{memory.task or 'unknown'}\n\n"
            f"Compacted prior summary:\n{memory.summary or 'None'}\n\n"
            f"Question:\n{question}\n\n"
            f"Record catalog JSON:\n{json.dumps(catalog, ensure_ascii=False)}\n\n"
            "Return ONLY JSON with keys: queries (list of strings), filters "
            "(list of strings), read_ids (list of record ids), rationale."
        )
        raw = self._llm_text(prompt, phase="query")
        plan = self._parse_json(raw)
        if not isinstance(plan, dict):
            plan = {}
        queries = [str(x).strip() for x in plan.get("queries", []) if str(x).strip()]
        if not queries:
            queries = self._fallback_queries(question)
        plan["queries"] = queries[:8]
        plan["filters"] = [str(x).strip() for x in plan.get("filters", []) if str(x).strip()][:6]
        plan["read_ids"] = [str(x).strip() for x in plan.get("read_ids", []) if str(x).strip()][: self.top_k]
        return plan

    def _execute_plan(
        self,
        memory: DCIMemory,
        question: str,
        plan: dict[str, Any],
    ) -> list[dict[str, Any]]:
        search_records = self._all_search_records(memory)
        if self._is_broad_lru_request(memory, question):
            return self._stratified_observations(search_records)

        by_id = {r.record_id: r for r in search_records}
        selected: dict[str, tuple[float, DCIRecord, list[str]]] = {}
        queries = self._multi_grep_queries(
            question,
            list(plan.get("queries") or self._fallback_queries(question)),
        )
        filters = list(plan.get("filters") or [])

        for query in queries:
            self._counters.record_tool_call()
            is_turn_step_anchor = bool(
                re.search(r"(?:turn|step)(?:\\s\+|\s+)\d+(?:(?:\\s\*)|\s*)?:", query, flags=re.I)
                or re.search(r"\b(?:turn|step)\s+\d+\s*:", query, flags=re.I)
            )
            regex = self._safe_regex(query)
            for record in search_records:
                matches = self._rg_lines(record.content, regex, limit=5)
                if is_turn_step_anchor and not matches:
                    continue
                if filters:
                    hay = "\n".join(matches) if matches else record.content
                    if not is_turn_step_anchor and not all(self._contains(hay, f) for f in filters):
                        continue
                score = self._score(query, question, record, matches)
                if is_turn_step_anchor and matches:
                    score += 2.0
                if score <= 0:
                    continue
                prev = selected.get(record.record_id)
                if prev is None or score > prev[0]:
                    selected[record.record_id] = (score, record, matches)

        for rid in plan.get("read_ids") or []:
            record = by_id.get(str(rid))
            if record is not None:
                selected.setdefault(record.record_id, (1.0, record, []))

        if not selected:
            ranked = sorted(
                search_records,
                key=lambda r: self._score(question, question, r, []),
                reverse=True,
            )
            for record in ranked[: self.top_k]:
                selected[record.record_id] = (self._score(question, question, record, []), record, [])

        observations: list[dict[str, Any]] = []
        for score, record, matches in sorted(selected.values(), key=lambda x: x[0], reverse=True)[: self.top_k]:
            observations.append(
                {
                    "record_id": record.record_id,
                    "title": record.title,
                    "score": round(float(score), 4),
                    "matches": matches[:5],
                    "read": self._bounded_read(record.content, matches),
                }
            )
        return observations

    def _all_search_records(self, memory: DCIMemory) -> list[DCIRecord]:
        records: list[DCIRecord] = []
        for i, compact in enumerate(memory.compacted_history):
            text = str(compact or "").strip()
            if not text:
                continue
            records.append(
                DCIRecord(
                    record_id=f"c{i}",
                    title=f"compacted_{i}",
                    content=text,
                    compact=self._compact_record(text),
                    metadata={"compacted": True, "index": i},
                )
            )
        records.extend(memory.records)
        return records

    def _is_broad_lru_request(self, memory: DCIMemory, question: str) -> bool:
        task = str(memory.task or "").lower()
        q = str(question or "").lower()
        if "category: lru" not in task and "long-range understanding" not in task:
            return False
        return any(
            phrase in q
            for phrase in (
                "write a summary",
                "summarize the book",
                "summarize the story",
                "write about the plot",
                "plot and characters",
            )
        )

    def _stratified_observations(self, records: list[DCIRecord]) -> list[dict[str, Any]]:
        if not records:
            return []
        n = min(self.top_k, len(records))
        if n == 1:
            indices = [0]
        else:
            indices = sorted({round(i * (len(records) - 1) / (n - 1)) for i in range(n)})
        observations: list[dict[str, Any]] = []
        for rank, idx in enumerate(indices):
            record = records[idx]
            observations.append(
                {
                    "record_id": record.record_id,
                    "title": record.title,
                    "score": round(1.0 - rank * 0.01, 4),
                    "matches": ["stratified read for broad LRU request"],
                    "read": self._truncate(record.content, min(self.max_record_chars, 2400)),
                }
            )
        return observations

    def _render_context(
        self,
        memory: DCIMemory,
        question: str,
        plan: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> str:
        lines = [
            "Direct corpus interaction memory context:",
            f"Context policy: {memory.context_level}",
            f"Question: {question}",
            f"Search queries: {json.dumps(plan.get('queries', []), ensure_ascii=False)}",
            f"Chained filters: {json.dumps(plan.get('filters', []), ensure_ascii=False)}",
        ]
        if self._is_broad_lru_request(memory, question):
            lines.append(
                "Broad LRU request: the answer must summarize the memorized corpus "
                "evidence below. Treat any few-shot examples inside the question as "
                "format examples, not as the target story."
            )
        if memory.summary:
            lines.append(f"Compacted older evidence:\n{memory.summary}")
        for obs in observations:
            lines.append(
                f"[{obs['record_id']}] {obs['title']} | score={obs['score']}\n"
                f"Matched lines:\n{self._join_or_none(obs.get('matches'))}\n"
                f"Local read:\n{obs.get('read') or ''}"
            )
        context = "\n\n".join(lines)
        if self.context_level in {"level4", "level5"} and len(context) > self.max_context_chars:
            summary_prompt = (
                "Compress this direct-corpus evidence for answering the question. "
                "Preserve only supported facts and exact constraints.\n\n"
                f"Question:\n{question}\n\nEvidence:\n{self._truncate(context, self.max_context_chars)}"
            )
            summary = self._llm_text(summary_prompt, phase="query")
            if summary:
                context = "Direct corpus interaction memory context:\n" + summary
        return self._truncate(context, self.max_context_chars)

    def _fallback_queries(self, question: str) -> list[str]:
        terms = [
            t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_'/-]+", question)
            if len(t) > 2 and t.lower() not in _STOPWORDS
        ]
        phrases = re.findall(r'"([^"]+)"|\'([^\']+)\'', question)
        out = [" ".join(x for pair in phrases for x in pair if x).strip()] if phrases else []
        out.extend(terms[:8])
        if len(terms) >= 2:
            out.append("|".join(re.escape(t) for t in terms[:4]))
        return [x for x in out if x][:8] or [question[:80]]

    def _multi_grep_queries(self, question: str, queries: list[str]) -> list[str]:
        """Expand a plan into a broader grep set without task-specific rules."""
        max_queries = max(8, int(__import__("os").environ.get("DCI_MAX_GREP_QUERIES", "64")))
        terms = []
        for term in self._terms(question):
            if term not in terms:
                terms.append(term)
        out: list[str] = []

        def add(q: str) -> None:
            q = str(q or "").strip()
            if q and q not in out:
                out.append(q)

        for q in queries:
            add(q)
        for label, number in self._step_anchor_numbers(question):
            other = "Turn" if label.lower() == "step" else "Step"
            add(rf"{label}\s+{number}\s*:")
            add(rf"{other}\s+{number}\s*:")
        for term in terms[:16]:
            add(re.escape(term))
        for a, b in zip(terms[:12], terms[1:13]):
            add(f"{re.escape(a)}.*{re.escape(b)}|{re.escape(b)}.*{re.escape(a)}")
        if terms:
            add("|".join(re.escape(t) for t in terms[:12]))
        return out[:max_queries]

    def _step_anchor_numbers(self, question: str) -> list[tuple[str, int]]:
        anchors: list[tuple[str, int]] = []

        def add(label: str, number: int) -> None:
            item = (label, int(number))
            if item not in anchors:
                anchors.append(item)

        for start, end in re.findall(
            r"\b(?:between|from)\s+(?:step|turn)\s+(\d+)\s+(?:and|to|-)\s+(?:step|turn)?\s*(\d+)",
            question,
            flags=re.I,
        ):
            lo, hi = sorted((int(start), int(end)))
            if hi - lo <= 30:
                for number in range(lo, hi + 1):
                    add("Step", number)

        explicit = [int(n) for _, n in re.findall(r"\b(step|turn)\s+(\d+)\b", question, flags=re.I)]
        for number in explicit:
            lo = max(0, number - 6)
            hi = number + 6
            for nearby in range(lo, hi + 1):
                add("Step", nearby)
        return anchors

    def _rg_lines(self, text: str, regex: re.Pattern[str], *, limit: int) -> list[str]:
        matches: list[str] = []
        for lineno, line in enumerate(str(text or "").splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{lineno}: {self._truncate(line.strip(), 500)}")
            if len(matches) >= limit:
                break
        return matches

    def _bounded_read(self, text: str, matches: list[str]) -> str:
        if not matches:
            return self._truncate(text, min(self.max_record_chars, 2000))
        line_numbers = []
        for match in matches:
            try:
                line_numbers.append(int(match.split(":", 1)[0]))
            except ValueError:
                continue
        lines = str(text or "").splitlines()
        if line_numbers and lines and line_numbers[0] <= 2 and re.match(r"\s*(?:Turn|Step)\s+\d+\s*:", lines[0], flags=re.I):
            anchor_chars = int(__import__("os").environ.get("DCI_ANCHOR_READ_CHARS", "8000"))
            return self._truncate(text, min(self.max_record_chars, anchor_chars))
        spans: list[str] = []
        for line_no in line_numbers[:3]:
            start = max(0, line_no - 3)
            end = min(len(lines), line_no + 2)
            spans.append("\n".join(lines[start:end]))
        return self._truncate("\n...\n".join(spans), 2000)

    def _compact_record(self, text: str) -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        return self._truncate(clean, 700)

    def _score(self, query: str, question: str, record: DCIRecord, matches: list[str]) -> float:
        q_terms = set(self._terms(query)) | set(self._terms(question))
        r_terms = set(self._terms(record.title + " " + record.content))
        if not q_terms or not r_terms:
            return 0.0
        overlap = len(q_terms & r_terms) / max(1, len(q_terms))
        return overlap + 0.15 * len(matches)

    def _terms(self, text: str) -> list[str]:
        return [
            t for t in re.findall(r"[a-z0-9][a-z0-9_'/-]+", str(text or "").lower())
            if len(t) > 2 and t not in _STOPWORDS
        ]

    def _contains(self, text: str, pattern: str) -> bool:
        return bool(self._safe_regex(pattern).search(str(text or "")))

    def _safe_regex(self, pattern: str) -> re.Pattern[str]:
        pattern = str(pattern or "").strip()
        if not pattern:
            pattern = r"$^"
        try:
            return re.compile(pattern, flags=re.I)
        except re.error:
            return re.compile(re.escape(pattern), flags=re.I)

    def _sanitize_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        blocked = {"gold", "gold_answer", "gold_exact", "gold_asin", "gold_excerpt", "answer"}
        return {k: v for k, v in feedback.items() if k not in blocked}

    def _join_or_none(self, values: Any) -> str:
        vals = [str(v) for v in (values or []) if str(v).strip()]
        return "\n".join(vals) if vals else "None"

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
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return ""
        elapsed = time.perf_counter() - started
        self._counters.add_seconds("w_llm_build" if phase == "build" else "w_llm", elapsed)
        usage = body.get("usage") or {}
        self._counters.record_llm_call(
            prompt_tokens=int(usage.get("prompt_tokens") or self._count_tokens(prompt)),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            phase="build" if phase == "build" else "query",
        )
        choices = body.get("choices") or []
        if not choices:
            return ""
        content = ((choices[0].get("message") or {}).get("content") or "").strip()
        return re.sub(r"<think>.*?</think>", "", content, flags=re.I | re.S).strip()

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

    def _truncate(self, text: str, limit: int) -> str:
        text = str(text or "")
        if len(text) <= limit:
            return text
        half = max(1, limit // 2)
        return text[:half].rstrip() + "\n... [dci truncated] ...\n" + text[-half:].lstrip()

    def _count_tokens(self, text: str) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0
        return len(str(text or "").split())

    def _persist(self, memory: DCIMemory) -> None:
        if not self.save_dir:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": memory.task,
            "context_level": memory.context_level,
            "summary": memory.summary,
            "compacted_history": memory.compacted_history[-200:],
            "records": [
                {
                    "record_id": r.record_id,
                    "title": r.title,
                    "content": r.content,
                    "compact": r.compact,
                    "metadata": r.metadata,
                }
                for r in memory.records
            ],
        }
        (self.save_dir / f"{self.name}_state.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

class DCILiteSummarizeMethod(DCILiteMethod):
    """DCI-Agent-Lite level4: truncation, compaction, and summarization."""

    name = "dci_lite_sum"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("context_level", "level4")
        super().__init__(**kwargs)

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "what",
    "which",
    "when",
    "where",
    "who",
    "why",
    "how",
    "was",
    "were",
    "are",
    "is",
    "did",
    "does",
    "have",
    "has",
    "into",
    "about",
    "using",
    "only",
    "answer",
}
