"""AutoMem: agent-controlled retrieval over raw text and an optional graph.

The planner chooses among dump, grep, read, graph query, and aggregate.
A judge can request follow-up retrieval rounds (up to ``max_agentic_iters``,
default 2). Grep and read always run on the raw records; graph queries run
only when graph querying is enabled and a graph backend is reachable.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from agentmem.methods.base import MethodKind
from agentmem.methods.dci_lite import (
    DCILiteMethod,
    DCIMemory,
    DCIRecord,
    _STOPWORDS,
)

@dataclass
class _CorpusCache:
    """Per-corpus artefacts keyed by sha1 of the trajectory text."""

    corpus_id: str
    record_token_counts: dict[str, int] = field(default_factory=dict)
    regex_cache: dict[str, re.Pattern[str]] = field(default_factory=dict)
    entity_table: dict[str, list[str]] | None = None                             
    entity_table_built: bool = False
    dump_answer_cache: dict[str, str] = field(default_factory=dict)
    regex_lru_order: list[str] = field(default_factory=list)

    def get_regex(self, pattern: str) -> re.Pattern[str] | None:
        return self.regex_cache.get(pattern)

    def put_regex(self, pattern: str, compiled: re.Pattern[str], *, max_entries: int = 2048) -> None:
        if pattern in self.regex_cache:
            return
        self.regex_cache[pattern] = compiled
        self.regex_lru_order.append(pattern)
        if len(self.regex_lru_order) > max_entries:
            old = self.regex_lru_order.pop(0)
            self.regex_cache.pop(old, None)

_AGG_OP_NAMES = {"intersect", "union", "temporal_sort", "count"}
_DATE_RE = re.compile(
    r"(?:date_time\s*=\s*)?([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|\d{4}-\d{2}-\d{2})",
    re.I,
)

class AutoMemMethod(DCILiteMethod):
    """DCI-Lite + three targeted memory-task upgrades."""

    kind = MethodKind.AGENTIC
    name = "automem"

    def __init__(
        self,
        *,

        enable_memory_primitives: bool = True,
        enable_entity_grep: bool = True,
        enable_aggregate: bool = True,
        enable_evidence_commit: bool = True,
        entity_table_max_aliases: int = 5,
        entity_table_min_freq: int = 2,

        enable_adaptive_dump: bool = True,
        dump_threshold_chars: int = 60000,                                  

        dump_overhead_ratio: float = 1.0,

        enable_corpus_cache: bool = True,
        cache_regex_max_entries: int = 2048,
        cache_dump_answers: bool = False,                                   

        enable_graph_query: bool | None = None,
        graph_uri: str | None = None,                                        
        graph_namespace_fn: Any = None,                                                
        graph_schema_hint: str = "",                                          
        graph_max_rows: int = 50,                                          

        max_agentic_iters: int = 2,

        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("top_k", 16)
        kwargs.setdefault("max_context_chars", 32000)
        super().__init__(**kwargs)
        self.enable_memory_primitives = bool(enable_memory_primitives)
        self.enable_entity_grep = bool(enable_entity_grep) and self.enable_memory_primitives
        self.enable_aggregate = bool(enable_aggregate) and self.enable_memory_primitives
        self.enable_evidence_commit = bool(enable_evidence_commit) and self.enable_memory_primitives
        self.entity_table_max_aliases = max(1, int(entity_table_max_aliases))
        self.entity_table_min_freq = max(1, int(entity_table_min_freq))
        self.enable_adaptive_dump = bool(enable_adaptive_dump)
        self.dump_threshold_chars = int(dump_threshold_chars)
        self.dump_overhead_ratio = float(dump_overhead_ratio)
        self.enable_corpus_cache = bool(enable_corpus_cache)
        self.cache_regex_max_entries = max(64, int(cache_regex_max_entries))
        self.cache_dump_answers = bool(cache_dump_answers)

        import os as _os
        env_graph_uri = _os.environ.get("AUTOMEM_GRAPH_URI") or ""
        if enable_graph_query is None:
            enable_graph_query = bool(env_graph_uri) or bool(graph_uri)
        self.enable_graph_query = bool(enable_graph_query)
        self.graph_uri = graph_uri
        self.graph_namespace_fn = graph_namespace_fn
        self.graph_schema_hint = str(graph_schema_hint or "")
        self.graph_max_rows = max(1, int(graph_max_rows))

        max_agentic_iters = int(_os.environ.get("AUTOMEM_MAX_AGENTIC_ITERS", max_agentic_iters))
        self.max_agentic_iters = max(0, int(max_agentic_iters))
        self._graph_driver = None                                  
        self._corpus_caches: dict[str, _CorpusCache] = {}

        self._fs_counters: dict[str, int] = {
            "u2_dump_fired": 0,
            "u2_dump_skipped": 0,
            "u2_escalated_to_u1": 0,
            "u3_regex_cache_hit": 0,
            "u3_regex_cache_miss": 0,
            "u1_entity_table_built": 0,
            "u1_aggregate_calls": 0,
            "u1_evidence_retry": 0,
            "u4_graph_query_calls": 0,
            "u4_graph_rows_returned": 0,
            "u4_graph_query_errors": 0,
            "judge_calls": 0,
            "judge_continue": 0,
        }

    def build(self, traj_text: str, *, task: str = "") -> DCIMemory:
        memory = super().build(traj_text, task=task)
        if self.enable_corpus_cache:
            corpus_id = hashlib.sha1(str(traj_text or "").encode("utf-8")).hexdigest()[:16]
            cache = self._corpus_caches.setdefault(corpus_id, _CorpusCache(corpus_id=corpus_id))
            for rec in self._all_search_records(memory):
                cache.record_token_counts[rec.record_id] = self._count_tokens(rec.content)

            memory.compacted_history = list(memory.compacted_history)
            setattr(memory, "_corpus_id", corpus_id)
        return memory

    def answer(self, memory: DCIMemory, question: str) -> str:
        if not memory.records and not memory.summary:
            return "No corpus records available."
        self._active_memory = memory                                           
        try:
            return self._answer_inner(memory, question)
        finally:
            self._active_memory = None

    def _answer_inner(self, memory: DCIMemory, question: str) -> str:
        """Unified agentic loop. No hand-coded U1 vs U2 dispatch, no
        bench-specific branches. The LLM-planner sees the corpus catalog
        plus a tool menu (dump / rg / read / graph_query / aggregate) and
        picks which tools to fire per question. The LLM-judge decides if
        the evidence is sufficient or another plan round is needed.

        Efficiency: dump is one tool option among many — fired only when
        the planner asks for it. Graph is auto-detected from the corpus
        cache, no precomputed namespace map required. The harness pays
        only for the tools the agent actually invokes."""
        started = time.perf_counter()
        plan = self._plan_search(memory, question)
        observations: list[dict[str, Any]] = []
        observations.extend(self._dispatch_actions(memory, question, plan))
        if self.enable_aggregate:
            agg_observations, used = self._apply_aggregate_ops(memory, plan, observations)
            if used:
                self._fs_counters["u1_aggregate_calls"] += 1
                observations = observations + agg_observations

        max_iters = max(0, int(getattr(self, "max_agentic_iters", 2)))
        for it in range(max_iters):
            verdict = self._judge_sufficient(memory, question, plan, observations)
            if verdict.get("sufficient", True):
                break
            self._fs_counters["u1_evidence_retry"] += 1
            followup = self._plan_search(
                memory, question,
                retry_hint=str(verdict.get("why_more") or "")[:400],
            )
            more = self._execute_plan(memory, question, followup)
            if self.enable_graph_query:
                more = more + self._execute_graph_queries(memory, followup)
            observations = observations + more

            plan["queries"] = (plan.get("queries") or []) + (followup.get("queries") or [])
        context = self._render_context(memory, question, plan, observations)
        elapsed = time.perf_counter() - started
        self._counters.add_seconds("w_tool", elapsed)
        self._counters.record_retrieval(
            candidates_scored=len(self._all_search_records(memory)),
            evidence_injected=len(observations),
            context_tokens=self._count_tokens(context),
        )
        self._publish_fs_counters()
        return context

    def _dispatch_actions(
        self,
        memory: DCIMemory,
        question: str,
        plan: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Single dispatcher for all four typed actions: dump, rg, read,
        graph_query. The plan dict tells us which to fire; the harness
        executes them and returns observations in a single list.
        Order: dump first (if requested) so its answer anchors the
        evidence list, then grep+read, then graph rows.
        """
        obs: list[dict[str, Any]] = []
        if plan.get("use_dump") and self.enable_adaptive_dump:
            self._fs_counters["u2_dump_fired"] += 1
            cache = self._cache_for(memory)
            cached = None
            if cache is not None and self.cache_dump_answers:
                cached = cache.dump_answer_cache.get(question)
            ans = cached if cached is not None else self._dump_and_answer(memory, question)
            if cache is not None and self.cache_dump_answers and ans and not cached:
                cache.dump_answer_cache[question] = ans
            obs.append({
                "record_id": "dump",
                "title": "whole_corpus_dump",
                "score": 1.0,
                "matches": [str(ans)],
                "read": "",
                "kind": "dump",
            })

        obs.extend(self._execute_plan(memory, question, plan))
        if self.enable_graph_query:
            obs.extend(self._execute_graph_queries(memory, plan))
        return obs

    def _dump_tool_hint(self, total_chars: int, n_records: int) -> str:
        """Brief data-derived hint about when dump is cheap; budgets only,
        no benchmark-specific advice."""
        if total_chars == 0:
            return ""
        rough_tokens = total_chars // 4
        return (
            f"Dump cost estimate: ~{rough_tokens} prompt tokens on the dump action; "
            f"if the question is a single-fact recall, dump is usually enough."
        )

    def _publish_fs_counters(self) -> None:
        """Expose internal fs counters under family_specific so the runner persists them per row."""
        self._counters.family_specific.update({
            f"automem/{k}": v for k, v in self._fs_counters.items()
        })

    def _judge_sufficient(
        self,
        memory: DCIMemory,
        question: str,
        plan: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ask the LLM whether the current evidence is sufficient to answer
        the question. No hand-coded heuristics; the model decides.

        Returns ``{"sufficient": bool, "why_more": str}`` (best-effort
        parsed JSON; on parse failure we default to sufficient=True so
        we don't loop forever on a malformed judge).
        """
        self._fs_counters["judge_calls"] += 1
        context = self._render_context(memory, question, plan, observations)

        if len(context) > 12000:
            context = context[:6000] + "\n…[truncated]…\n" + context[-6000:]
        prompt = (
            "You are auditing the evidence a retrieval harness collected "
            "for a question. Decide whether it is ENOUGH to write a "
            "confident, factually-grounded answer.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence so far:\n{context}\n\n"
            "Return ONLY one JSON object on a single line with keys:\n"
            '  "sufficient": true|false,\n'
            '  "why_more": "<if false, one sentence on what is still missing'
            ' — name an entity, a date, a session, or a relation>"\n'
            "Only mark false if a factual answer would require a specific "
            "piece of evidence that does not appear above."
        )
        raw = self._llm_text(prompt, phase="query")
        verdict = self._parse_json(raw) or {}
        if not isinstance(verdict, dict):
            verdict = {}
        sufficient = bool(verdict.get("sufficient", True))
        if not sufficient:
            self._fs_counters["judge_continue"] += 1
        return {
            "sufficient": sufficient,
            "why_more": str(verdict.get("why_more") or ""),
        }

    def _dump_and_answer(self, memory: DCIMemory, question: str) -> str:
        """Single-call answer over the full corpus.

        Returns ``DIRECT_ANSWER_PREFIX``-prefixed string so the benchmark
        runner skips its own answer LLM. The prefix sentinel is borrowed
        from ``agentmem/eval/hotpotqa_mem_methods.py`` to keep parity.
        """
        body_lines: list[str] = []
        for rec in self._all_search_records(memory):
            body_lines.append(f"[{rec.record_id}] {rec.title}\n{rec.content}")
        body = "\n\n".join(body_lines)
        body = self._truncate(body, self.dump_threshold_chars)
        prompt = (
            "Answer the question using only the corpus below. "
            "Cite the record_id(s) of the span you used.\n"
            "If the corpus does not contain the answer, write 'Not mentioned'.\n"
            "Output format: ANSWER: <text> | EVIDENCE: <record_ids comma-separated>\n\n"
            f"Question: {question}\n\nCorpus:\n{body}"
        )
        raw = self._llm_text(prompt, phase="query")
        if not raw:
            return ""
        answer, evidence = self._parse_dump_answer(raw)

        return f"###Answer: {answer}".strip()

    def _parse_dump_answer(self, raw: str) -> tuple[str, list[str]]:
        text = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.I | re.S).strip()
        ans_match = re.search(r"ANSWER\s*:\s*(.*?)(?:\s*\|\s*EVIDENCE\s*:|$)", text, flags=re.I | re.S)
        ev_match = re.search(r"EVIDENCE\s*:\s*(.*)$", text, flags=re.I | re.S)
        answer = (ans_match.group(1).strip() if ans_match else text).strip()
        evidence_raw = ev_match.group(1).strip() if ev_match else ""
        evidence = [
            tok.strip() for tok in re.split(r"[,\s]+", evidence_raw) if tok.strip()
        ]
        return answer, evidence

    def _plan_search(
        self,
        memory: DCIMemory,
        question: str,
        *,
        retry_hint: str | None = None,
    ) -> dict[str, Any]:
        if not self.enable_memory_primitives:
            return super()._plan_search(memory, question)

        search_records = self._all_search_records(memory)
        catalog = [
            {
                "id": r.record_id,
                "title": r.title,
                "preview": r.compact or self._compact_record(r.content),
            }
            for r in search_records[: min(len(search_records), 80)]
        ]
        entity_table_hint = ""
        if self.enable_entity_grep:
            entities = self._get_or_build_entity_table(memory)
            if entities:

                preview = dict(list(entities.items())[:12])
                entity_table_hint = (
                    "\nEntity normalization hints (canonical -> aliases). "
                    "Use the aliases when forming queries to catch paraphrases / "
                    "pronouns; pass them as alternation in a regex query.\n"
                    f"{json.dumps(preview, ensure_ascii=False)}"
                )

        agg_hint = ""
        if self.enable_aggregate:
            agg_hint = (
                "\nYou may optionally request post-retrieval aggregation ops by "
                "setting the 'aggregate_ops' field. Each op is "
                "{op: 'intersect'|'union'|'temporal_sort'|'count', args: [query_a, query_b?]}. "
                "'intersect' / 'union' operate over record_ids whose content matched a previous query. "
                "'temporal_sort' orders matched records by session date_time header. "
                "'count' returns the number of matches as a single observation."
            )

        graph_hint = ""
        if self.enable_graph_query and self._get_graph_driver(memory) is not None:
            schema_block = self.graph_schema_hint or (
                "Graph schema (Memgraph, openCypher):\n"
                "  (:Episodic {eid, observation, action, subgoal, state, time})\n"
                "  (:Semantic {sid, text, time})\n"
                "  (:Tag      {name})\n"
                "  (:Semantic)-[:HAS_TAG]->(:Tag)\n"
                "  (:Semantic)-[:MENTIONS]->(:Episodic)\n"
                "  (:Semantic)-[:SHARES_TAG]->(:Semantic)\n"
                "All nodes carry a 'ns' property scoping them to this trajectory."
            )
            graph_hint = (
                "\nYou may also issue ONE OR MORE typed graph queries against a "
                "preloaded openCypher endpoint by populating 'graph_queries' "
                "(list of Cypher strings). Use them when the question is a "
                "set / counting / 2-hop / temporal-ordering question that "
                "would otherwise require reading many records.\n"
                f"{schema_block}\n"
                "Each query MUST include `{ns: $ns}` filters and MUST be "
                "read-only (MATCH/RETURN; no CREATE/SET/DELETE)."
            )

        total_chars = sum(len(r.content) for r in memory.records) if memory.records else 0
        n_records = len(memory.records) if memory.records else 0
        dump_hint = self._dump_tool_hint(total_chars, n_records) if self.enable_adaptive_dump else ""

        prompt_lines = [
            "You control a direct-corpus interaction harness. You have a menu of "
            "actions and you decide which to invoke per question:",
            "- `queries`/`filters` (rg-style regex over records; cheapest, surfaces spans)",
            "- `read_ids` (read full content of named records when you know the right ones)",
            "- `aggregate_ops` (intersect/union/temporal_sort/count over prior hits)" if self.enable_aggregate else "",
            "- `graph_queries` (typed Cypher against a preloaded subgraph)" if (self.enable_graph_query and self._get_graph_driver(memory) is not None) else "",
            "- `use_dump: true` (single LLM call over the whole corpus; cheap on small corpora, lossy on large ones)" if self.enable_adaptive_dump else "",
            f"Task label:\n{memory.task or 'unknown'}",
            f"Compacted prior summary:\n{memory.summary or 'None'}",
            f"Question:\n{question}",
            f"Corpus stats: {n_records} records, {total_chars} chars total.",
            dump_hint,
            entity_table_hint,
            agg_hint,
            graph_hint,
        ]
        if retry_hint:
            prompt_lines.append(f"Planner hint: {retry_hint}")
        plan_keys = "use_dump (bool), queries (list of strings), filters (list of strings), read_ids (list of record ids), aggregate_ops (list, optional)"
        if self.enable_graph_query and self._get_graph_driver(memory) is not None:
            plan_keys += ", graph_queries (list of Cypher strings, optional)"
        plan_keys += ", rationale"
        prompt_lines.extend([
            f"Record catalog JSON:\n{json.dumps(catalog, ensure_ascii=False)}",
            f"Return ONLY JSON with keys: {plan_keys}.",
        ])
        prompt = "\n\n".join(p for p in prompt_lines if p)
        raw = self._llm_text(prompt, phase="query")
        plan = self._parse_json(raw) or {}
        if not isinstance(plan, dict):
            plan = {}
        queries = [str(x).strip() for x in plan.get("queries", []) if str(x).strip()]
        if not queries:
            queries = self._fallback_queries(question)
        if self.enable_entity_grep:
            queries = self._expand_with_entities(memory, queries)
        plan["queries"] = queries[:8]
        plan["filters"] = [str(x).strip() for x in plan.get("filters", []) if str(x).strip()][:6]
        plan["read_ids"] = [str(x).strip() for x in plan.get("read_ids", []) if str(x).strip()][: self.top_k]
        plan["aggregate_ops"] = self._sanitize_aggregate_ops(plan.get("aggregate_ops"))
        plan["graph_queries"] = self._sanitize_graph_queries(plan.get("graph_queries"))
        plan["use_dump"] = bool(plan.get("use_dump", False))
        return plan

    def _sanitize_aggregate_ops(self, ops: Any) -> list[dict[str, Any]]:
        if not isinstance(ops, list):
            return []
        clean: list[dict[str, Any]] = []
        for op in ops[:4]:
            if not isinstance(op, dict):
                continue
            name = str(op.get("op", "")).strip().lower()
            if name not in _AGG_OP_NAMES:
                continue
            args = op.get("args")
            if not isinstance(args, list):
                continue
            args = [str(a).strip() for a in args if str(a).strip()][:3]
            if not args:
                continue
            clean.append({"op": name, "args": args})
        return clean

    def _expand_with_entities(self, memory: DCIMemory, queries: list[str]) -> list[str]:
        cache = self._cache_for(memory)
        if cache is None or not cache.entity_table:
            return queries
        expanded: list[str] = []
        for q in queries:
            aliases: list[str] = []
            q_lower = q.lower()
            for canonical, alias_list in cache.entity_table.items():
                if canonical.lower() in q_lower:
                    for a in alias_list:
                        if a.lower() not in q_lower and a not in aliases:
                            aliases.append(a)
            if aliases:
                expanded.append(q + " | " + " | ".join(re.escape(a) for a in aliases[: self.entity_table_max_aliases]))
            else:
                expanded.append(q)
        return expanded

    def _get_or_build_entity_table(self, memory: DCIMemory) -> dict[str, list[str]]:
        cache = self._cache_for(memory)
        if cache is None:
            return {}
        if cache.entity_table_built:
            return cache.entity_table or {}

        sketch_parts: list[str] = []
        budget = 6000
        for rec in self._all_search_records(memory):
            if budget <= 0:
                break
            chunk = rec.content[: min(len(rec.content), 800)]
            sketch_parts.append(f"[{rec.record_id}] {rec.title}\n{chunk}")
            budget -= len(chunk)
        sketch = "\n\n".join(sketch_parts)
        prompt = (
            "Extract the canonical entities (people, places, organisations, "
            "topical objects) from the corpus below, along with up to "
            f"{self.entity_table_max_aliases} aliases each (pronouns, "
            "nicknames, paraphrases). Return ONLY JSON like:\n"
            '{"Caroline": ["she", "Carol"], "Project Atlas": ["Atlas"]}.\n'
            "Skip entities you cannot find an alias for. Skip entities that "
            f"appear fewer than {self.entity_table_min_freq} times.\n\n"
            f"Corpus sketch:\n{sketch}"
        )
        raw = self._llm_text(prompt, phase="build")
        cache.entity_table_built = True
        parsed = self._parse_json(raw)
        if not isinstance(parsed, dict):
            cache.entity_table = {}
            return {}
        table: dict[str, list[str]] = {}
        for k, v in parsed.items():
            if not isinstance(k, str) or not isinstance(v, list):
                continue
            aliases = [str(a).strip() for a in v if str(a).strip()][: self.entity_table_max_aliases]
            if aliases:
                table[k.strip()] = aliases
        cache.entity_table = table
        self._fs_counters["u1_entity_table_built"] += 1
        return table

    def _apply_aggregate_ops(
        self,
        memory: DCIMemory,
        plan: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        ops = plan.get("aggregate_ops") or []
        if not ops:
            return [], False
        search_records = self._all_search_records(memory)
        by_id = {r.record_id: r for r in search_records}
        results: list[dict[str, Any]] = []
        used = False
        for op_spec in ops:
            if not isinstance(op_spec, dict):
                continue
            op = op_spec.get("op")
            args = op_spec.get("args") or []
            if not op:
                continue
            if op in {"intersect", "union", "count"} and args:
                hits_per_query: list[set[str]] = []
                for query in args:
                    regex = self._safe_regex_cached(memory, query)
                    matched = {
                        rec.record_id
                        for rec in search_records
                        if regex.search(rec.content)
                    }
                    hits_per_query.append(matched)
                if not hits_per_query:
                    continue
                if op == "intersect":
                    final = set.intersection(*hits_per_query) if hits_per_query else set()
                    payload = sorted(final)
                    results.append({
                        "record_id": "agg",
                        "title": f"aggregate:intersect({', '.join(args)})",
                        "score": 1.0,
                        "matches": [f"intersect_ids={','.join(payload) or 'none'}"],
                        "read": "; ".join(
                            f"[{rid}] {by_id[rid].title}" for rid in payload if rid in by_id
                        ) or "no intersection",
                    })
                elif op == "union":
                    final = set.union(*hits_per_query) if hits_per_query else set()
                    payload = sorted(final)
                    results.append({
                        "record_id": "agg",
                        "title": f"aggregate:union({', '.join(args)})",
                        "score": 1.0,
                        "matches": [f"union_ids={','.join(payload) or 'none'}"],
                        "read": "; ".join(
                            f"[{rid}] {by_id[rid].title}" for rid in payload[:8] if rid in by_id
                        ) or "no union members",
                    })
                elif op == "count":
                    n = len(hits_per_query[0])
                    results.append({
                        "record_id": "agg",
                        "title": f"aggregate:count({args[0]})",
                        "score": 1.0,
                        "matches": [f"count={n}"],
                        "read": f"{n} record(s) matched query '{args[0]}'",
                    })
                used = True
            elif op == "temporal_sort":
                ordered = self._temporal_sort_records(memory, observations)
                if ordered:
                    results.append({
                        "record_id": "agg",
                        "title": "aggregate:temporal_sort",
                        "score": 1.0,
                        "matches": [f"order={','.join(rid for rid, _ in ordered)}"],
                        "read": "\n".join(
                            f"[{rid}] {date_str}" for rid, date_str in ordered
                        ),
                    })
                    used = True
        return results, used

    def _temporal_sort_records(
        self,
        memory: DCIMemory,
        observations: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        by_id = {r.record_id: r for r in self._all_search_records(memory)}
        items: list[tuple[str, str, str]] = []                              
        for obs in observations:
            rid = obs.get("record_id")
            if not rid or rid == "agg" or rid not in by_id:
                continue
            content = by_id[rid].content
            match = _DATE_RE.search(content[:400]) or _DATE_RE.search(by_id[rid].title)
            if not match:
                continue
            items.append((rid, match.group(1), match.group(1)))
        items.sort(key=lambda x: x[1])
        return [(rid, raw) for rid, _, raw in items]

    def _cache_for(self, memory: DCIMemory) -> _CorpusCache | None:
        if not self.enable_corpus_cache:
            return None
        corpus_id = getattr(memory, "_corpus_id", None)
        if corpus_id is None:
            return None
        cache = self._corpus_caches.get(corpus_id)
        if cache is None:
            cache = _CorpusCache(corpus_id=corpus_id)
            self._corpus_caches[corpus_id] = cache
        return cache

    def _safe_regex_cached(self, memory: DCIMemory, pattern: str) -> re.Pattern[str]:
        cache = self._cache_for(memory)
        if cache is None:
            return super()._safe_regex(pattern)
        hit = cache.get_regex(pattern)
        if hit is not None:
            self._fs_counters["u3_regex_cache_hit"] += 1
            return hit
        self._fs_counters["u3_regex_cache_miss"] += 1
        compiled = super()._safe_regex(pattern)
        cache.put_regex(pattern, compiled, max_entries=self.cache_regex_max_entries)
        return compiled

    def _safe_regex(self, pattern: str) -> re.Pattern[str]:                          
        mem = getattr(self, "_active_memory", None)
        if mem is not None and self.enable_corpus_cache:
            return self._safe_regex_cached(mem, pattern)
        return super()._safe_regex(pattern)

    _CYPHER_FORBIDDEN = re.compile(
        r"\b(?:CREATE|DELETE|DETACH|MERGE|SET|REMOVE|DROP|CALL|LOAD\s+CSV)\b",
        re.IGNORECASE,
    )

    def _sanitize_graph_queries(self, raw: Any) -> list[str]:
        if not self.enable_graph_query or not isinstance(raw, list):
            return []
        out: list[str] = []
        for q in raw[: 4]:
            s = str(q or "").strip()
            if not s:
                continue
            if self._CYPHER_FORBIDDEN.search(s):
                continue
            if "MATCH" not in s.upper() and "RETURN" not in s.upper():
                continue
            out.append(s)
        return out

    def _get_graph_driver(self, memory: DCIMemory) -> Any:
        if not self.enable_graph_query or not self.graph_uri:
            return None
        if self._graph_driver is not None:
            return self._graph_driver
        try:
            from neo4j import GraphDatabase                
        except Exception:
            return None
        try:
            self._graph_driver = GraphDatabase.driver(self.graph_uri, auth=("", ""))
        except Exception:
            self._graph_driver = None
        return self._graph_driver

    def _graph_namespace(self, memory: DCIMemory) -> str:
        if callable(self.graph_namespace_fn):
            try:
                return str(self.graph_namespace_fn(getattr(memory, "_corpus_id", None), memory))
            except Exception:
                pass
        return getattr(memory, "_corpus_id", "") or "default"

    def _execute_graph_queries(self, memory: DCIMemory, plan: dict[str, Any]) -> list[dict[str, Any]]:
        cyphers = plan.get("graph_queries") or []
        if not cyphers:
            return []
        driver = self._get_graph_driver(memory)
        if driver is None:
            return []
        ns = self._graph_namespace(memory)
        obs: list[dict[str, Any]] = []
        for idx, cypher in enumerate(cyphers[:4]):
            self._fs_counters["u4_graph_query_calls"] += 1

            try:
                with driver.session() as sess:
                    rows = [r.data() for r in sess.run(cypher, ns=ns)]
                rows = rows[: self.graph_max_rows]
                self._fs_counters["u4_graph_rows_returned"] += len(rows)
                summary_lines = [self._format_graph_row(r) for r in rows[:10]]
                obs.append({
                    "record_id": f"graph_q{idx + 1}",
                    "title": f"graph_query: {cypher[:120].replace(chr(10), ' ')}",
                    "score": 1.0,
                    "matches": summary_lines or ["(empty result)"],
                    "read": "",
                    "kind": "graph",
                    "cypher": cypher,
                    "rows": rows,
                })
            except Exception as exc:
                self._fs_counters["u4_graph_query_errors"] += 1
                obs.append({
                    "record_id": f"graph_q{idx + 1}_err",
                    "title": "graph_query (error)",
                    "score": 0.0,
                    "matches": [f"ERROR: {str(exc)[:200]}"],
                    "read": "",
                    "kind": "graph",
                    "cypher": cypher,
                    "error": str(exc)[:200],
                })
        return obs

    @staticmethod
    def _format_graph_row(row: dict[str, Any]) -> str:
        """Compact one Cypher row into a single line for the LLM context."""
        parts: list[str] = []
        for k, v in row.items():
            s = str(v)
            if len(s) > 160:
                s = s[:157] + "…"
            parts.append(f"{k}={s}")
        return " | ".join(parts)

    def reset_counters(self) -> Any:
        out = super().reset_counters() if hasattr(super(), "reset_counters") else None
        for k in list(self._fs_counters.keys()):
            self._fs_counters[k] = 0
        return out

    def shutdown(self) -> None:                          
        if self._graph_driver is not None:
            try:
                self._graph_driver.close()
            except Exception:
                pass
            self._graph_driver = None
        sup = getattr(super(), "shutdown", None)
        if callable(sup):
            sup()

class AutoMemU1Method(AutoMemMethod):
    """Ablation: only U1 (memory primitives) enabled."""

    name = "automem_u1"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("enable_memory_primitives", True)
        kwargs.setdefault("enable_adaptive_dump", False)
        kwargs.setdefault("enable_corpus_cache", False)
        super().__init__(**kwargs)

class AutoMemU2Method(AutoMemMethod):
    """Ablation: only U2 (adaptive dump) enabled."""

    name = "automem_u2"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("enable_memory_primitives", False)
        kwargs.setdefault("enable_adaptive_dump", True)
        kwargs.setdefault("enable_corpus_cache", False)
        super().__init__(**kwargs)

class AutoMemU3Method(AutoMemMethod):
    """Ablation: only U3 (per-corpus cache) enabled."""

    name = "automem_u3"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("enable_memory_primitives", False)
        kwargs.setdefault("enable_adaptive_dump", False)
        kwargs.setdefault("enable_corpus_cache", True)
        super().__init__(**kwargs)

class AutoMemNoGraphMethod(AutoMemMethod):
    """Ablation: full AutoMEM with the index_query (typed graph query) tool
    forced off, everything else at default.

    This is the clean ``AutoMEM - index_query`` cell: it isolates the indexed /
    graph retrieval component from the grep/read loop + U1 primitives + U2 dump +
    judge-replan. Pair it with ``automem_graph`` (index_query ON, requires a
    populated Memgraph + neo4j driver + AUTOMEM_GRAPH_URI) to read off exactly
    how much the indexed-access tool contributes over the DCI-style loop.

    NOTE: with no AUTOMEM_GRAPH_URI and no neo4j server, plain ``automem`` also
    runs with graph off — but this class pins enable_graph_query=False explicitly
    so the ablation is unambiguous and independent of the environment.
    """

    name = "automem_nograph"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["enable_graph_query"] = False                                
        super().__init__(**kwargs)

class AutoMemSumMethod(AutoMemMethod):
    """AutoMem + summarization (level4). Mirrors DCILiteSummarizeMethod's
    role on HotpotQA where +Sum is the paper's strongest variant (89.0)."""

    name = "automem_sum"

    def __init__(self, **kwargs: Any) -> None:

        kwargs.setdefault("enable_adaptive_dump", False)
        kwargs.setdefault("enable_memory_primitives", True)
        kwargs.setdefault("enable_corpus_cache", True)
        kwargs.setdefault("context_level", "level4")
        super().__init__(**kwargs)

class AutoMemGraphMethod(AutoMemMethod):
    """AutoMem with U4 (typed graph_query against Memgraph) enabled.

    Reads runtime config from env vars to avoid plumbing CLI changes
    through every benchmark runner:

      AUTOMEM_GRAPH_URI       : Bolt URI (default bolt://127.0.0.1:7687)
      AUTOMEM_GRAPH_NS_MAP    : path to JSON {corpus_id: namespace} map
                                (e.g. produced by scripts/locomo_corpus_to_ns.py)
      AUTOMEM_GRAPH_FALLBACK_NS: namespace returned when corpus_id is unknown
                                 (default "default")
    """

    name = "automem_graph"

    def __init__(self, **kwargs: Any) -> None:
        import os as _os
        import json as _json

        kwargs.setdefault("enable_graph_query", True)
        kwargs.setdefault(
            "graph_uri", _os.environ.get("AUTOMEM_GRAPH_URI", "bolt://127.0.0.1:7687"),
        )

        schema_hint = _os.environ.get("AUTOMEM_GRAPH_SCHEMA_HINT", "")
        if schema_hint:
            kwargs.setdefault("graph_schema_hint", schema_hint)
        ns_map_path = _os.environ.get("AUTOMEM_GRAPH_NS_MAP", "")
        ns_map: dict[str, str] = {}
        if ns_map_path:
            try:
                ns_map = _json.loads(open(ns_map_path, "r").read())
            except Exception:
                ns_map = {}
        fallback = _os.environ.get("AUTOMEM_GRAPH_FALLBACK_NS", "default")
        kwargs.setdefault(
            "graph_namespace_fn",
            lambda cid, _mem: ns_map.get(str(cid or ""), fallback),
        )
        super().__init__(**kwargs)

_ = _STOPWORDS
