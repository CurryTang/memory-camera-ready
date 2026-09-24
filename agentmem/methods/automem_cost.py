"""AutoMem-Cost: call-level gated inference for agentic memory.

Subclasses :class:`AutoMemMethod` and adds three deterministic *cost gates*
without changing the retrieval algorithm, memory store, or LLM prompts:

* **CatalogGate (SPC)** — embeds records once at build, then at query time
  builds the planner catalog as `top_K_e_by_embedding ∪ lexical_entity_hits ∪
  top-3-most-recent`, packed into a token budget. The planner LLM is always
  fired; only its *input* is compressed.
* **SufficiencyGate (GS)** — replaces the per-iter LLM judge with a typed
  deterministic predicate stack (entity_coverage, score_margin,
  tool_diversity_if_needed, renderable). LLM judge fires only when
  predicates fail (or on empty observations).
* **DispatchGate** — at query entry, either reuses a cached planner output
  (`(corpus_id, normalize(q))` lookup; tools execute fresh), or fires the
  whole-corpus dump LLM only when its token cost is locked-arithmetic-cheaper
  than the cascade, or runs the full cascade.

A `kvcache_friendly=True` flag rebuilds the planner prompt so corpus-stable
content (catalog + tool menu) comes BEFORE per-query content (question +
retry hint), maximizing prefix-cache hits across queries on the same corpus.

When `enable_cost_gates=False`, every override delegates to the parent so
behaviour is bit-identical to AutoMemMethod.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Literal, NamedTuple, Protocol

from agentmem.methods.automem import AutoMemMethod, _CorpusCache
from agentmem.methods.dci_lite import DCIMemory, DCIRecord

class GateDecision(NamedTuple):
    action: Literal["skip", "fire", "fallback"]
    uncertainty_regime: str
    reasoning: dict[str, Any]

class Gate(Protocol):
    def __call__(
        self,
        call_type: Literal["planner", "judge", "dispatch"],
        query: str,
        memory: DCIMemory,
        observations: list[dict[str, Any]] | None,
    ) -> GateDecision: ...

_ENTITY_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]{2,}")
_AGG_KEYWORDS = {"count", "how many", "all of", "list", "every", "since", "before", "after", "between", "total"}
_REL_KEYWORDS = {"who", "which", "where", "what relation", "how is"}

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))

def _normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").lower().rstrip("? ")).strip()

def _extract_entities(question: str) -> list[str]:
    """Cheap entity extraction: capitalized words + numerics + multi-char tokens."""
    tokens = _ENTITY_TOKEN_RE.findall(question or "")
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        low = t.lower()
        if low in {"the", "and", "what", "which", "who", "when", "where", "why", "how", "did", "does", "was", "were", "is", "are", "for", "from", "with", "this", "that"}:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(t)
    return out

class AutoMemCostMethod(AutoMemMethod):
    """AutoMem + three deterministic cost gates."""

    name = "automem_cost"

    def __init__(
        self,
        *,

        enable_cost_gates: bool = True,

        spc_enabled: bool = True,
        gs_enabled: bool = True,
        dispatch_enabled: bool = True,

        spc_token_budget: int = 5000,
        spc_K_e: int = 20,
        spc_K_l: int = 10,
        spc_recent_top: int = 3,
        spc_preview_chars: int = 200,

        gs_theta_top: float = 0.7,
        gs_theta_margin: float = 0.2,
        gs_min_context_tokens: int = 200,

        dispatch_cache_enabled: bool = True,
        dispatch_dump_arithmetic_enabled: bool = True,
        dispatch_dump_min_chars: int = 0,                                                   
        calibrated_mean_planner_tokens: int = 4200,
        calibrated_mean_judge_tokens: int = 2800,

        kvcache_friendly: bool = False,

        embedding_model: str | None = None,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_batch_size: int = 64,

        spc_disabled_corpora: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        def _envf(name, default):
            v = os.environ.get(name)
            return type(default)(v) if v is not None and v != "" else default
        self.enable_cost_gates = bool(enable_cost_gates)
        self.spc_enabled = bool(spc_enabled)
        self.gs_enabled = bool(gs_enabled)
        self.dispatch_enabled = bool(dispatch_enabled)
        self.spc_token_budget = int(_envf("AUTOMEM_COST_BUDGET", spc_token_budget))
        self.spc_K_e = int(spc_K_e)
        self.spc_K_l = int(spc_K_l)
        self.spc_recent_top = int(spc_recent_top)
        self.spc_preview_chars = int(spc_preview_chars)
        self.gs_theta_top = float(_envf("AUTOMEM_COST_THETA_TOP", gs_theta_top))
        self.gs_theta_margin = float(_envf("AUTOMEM_COST_THETA_MARGIN", gs_theta_margin))
        self.gs_min_context_tokens = int(gs_min_context_tokens)
        self.dispatch_cache_enabled = bool(dispatch_cache_enabled) and bool(int(_envf("AUTOMEM_COST_DISPATCH_CACHE", 1)))
        self.dispatch_dump_arithmetic_enabled = bool(dispatch_dump_arithmetic_enabled) and bool(int(_envf("AUTOMEM_COST_DISPATCH_DUMP", 1)))
        self.dispatch_dump_min_chars = int(_envf("AUTOMEM_COST_DUMP_MIN_CHARS", dispatch_dump_min_chars))
        self.calibrated_mean_planner_tokens = int(calibrated_mean_planner_tokens)
        self.calibrated_mean_judge_tokens = int(calibrated_mean_judge_tokens)
        self.kvcache_friendly = bool(kvcache_friendly)
        self.embedding_model = embedding_model or os.environ.get("EMBEDDING_MODEL") or "Qwen/Qwen3-Embedding-4B"
        self.embedding_base_url = (embedding_base_url or os.environ.get("EMBEDDING_BASE_URL") or "").rstrip("/")
        self.embedding_api_key = embedding_api_key or os.environ.get("EMBEDDING_API_KEY") or "EMPTY"
        self.embedding_batch_size = int(embedding_batch_size)
        self.spc_disabled_corpora = set(spc_disabled_corpora or [])

        self._fs_counters.update({
            "automem_cost/spc_skipped": 0,
            "automem_cost/spc_applied": 0,
            "automem_cost/spc_catalog_records": 0,
            "automem_cost/gs_tier1_pass": 0,
            "automem_cost/gs_tier1_fail_p1": 0,
            "automem_cost/gs_tier1_fail_p2": 0,
            "automem_cost/gs_tier1_fail_p3": 0,
            "automem_cost/gs_tier1_fail_p4": 0,
            "automem_cost/gs_tier2_fired": 0,
            "automem_cost/gs_empty_obs_fallback": 0,
            "automem_cost/dispatch_cache_hit": 0,
            "automem_cost/dispatch_dump_dominated": 0,
            "automem_cost/dispatch_cascade_default": 0,
            "automem_cost/kvcache_friendly_prompts": 0,
        })

    def build(self, traj_text: str, *, task: str = "") -> DCIMemory:
        memory = super().build(traj_text, task=task)
        if self.enable_cost_gates and self.spc_enabled and self.embedding_base_url:
            self._populate_embeddings(memory)
        return memory

    def _populate_embeddings(self, memory: DCIMemory) -> None:
        cache = self._cache_for(memory)
        if cache is None:
            return

        if not hasattr(cache, "embeddings"):
            cache.embeddings = {}                              
        records = self._all_search_records(memory)

        if records and cache.embeddings and len(cache.embeddings) >= len(records):                              
            return
        if not records:
            return
        texts: list[str] = []
        rids: list[str] = []
        for r in records:
            payload = (r.compact or "") + " " + (r.title or "")
            payload = payload.strip()[:1500]
            if not payload:
                payload = (r.content or "")[:1500]
            if not payload:
                continue
            texts.append(payload)
            rids.append(r.record_id)
        if not texts:
            return
        embeddings = self._embed_batch(texts)
        if not embeddings:
            return
        for rid, vec in zip(rids, embeddings):
            cache.embeddings[rid] = vec                              

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI                
        except Exception:
            return []
        if not self.embedding_base_url:
            return []
        try:
            client = OpenAI(api_key=self.embedding_api_key or "EMPTY", base_url=self.embedding_base_url)
        except Exception:
            return []
        out: list[list[float]] = []
        bs = max(1, int(self.embedding_batch_size))
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            try:
                resp = client.embeddings.create(model=self.embedding_model, input=chunk)
            except Exception as exc:

                self._fs_counters["automem_cost/embed_errors"] = self._fs_counters.get("automem_cost/embed_errors", 0) + 1
                _ = exc
                return []
            for item in resp.data:
                out.append(list(item.embedding))

        try:
            self._counters.record_llm_call(
                prompt_tokens=sum(len(t) // 4 for t in texts),
                completion_tokens=0,
                phase="build",
            )
        except Exception:
            pass
        return out

    def _dispatch_gate(self, memory: DCIMemory, question: str) -> GateDecision:
        cache = self._cache_for(memory)

        if cache is not None and self.dispatch_cache_enabled:
            plan_cache: dict[str, dict] = getattr(cache, "plan_cache", {})                            
            key = _normalize_question(question)
            plan = plan_cache.get(key)
            if plan is not None:
                return GateDecision("skip", "plan_cache_hit", {"key": key})

        if self.dispatch_dump_arithmetic_enabled and self.enable_adaptive_dump:
            total_chars = sum(len(r.content) for r in self._all_search_records(memory))
            dump_in = -(-total_chars // 4)                 
            catalog_chars = min(total_chars, self.spc_token_budget * 4)
            answer_estimate = min(2000, catalog_chars // 4 + 400)
            max_iters = max(0, int(getattr(self, "max_agentic_iters", 2)))
            cascade_in = (
                self.calibrated_mean_planner_tokens
                + self.calibrated_mean_judge_tokens * max_iters
                + answer_estimate
            )

            big_enough = total_chars >= int(self.dispatch_dump_min_chars)
            fits_in_dump_window = total_chars <= int(self.dump_threshold_chars)
            if dump_in <= cascade_in and big_enough and fits_in_dump_window:
                return GateDecision(
                    "fire",
                    "dump_cost_dominated",
                    {"dump_in": dump_in, "cascade_in": cascade_in, "total_chars": total_chars},
                )
        return GateDecision("fire", "cascade_default", {})

    def _answer_inner(self, memory: DCIMemory, question: str) -> str:
        if not (self.enable_cost_gates and self.dispatch_enabled):
            return super()._answer_inner(memory, question)
        decision = self._dispatch_gate(memory, question)
        if decision.uncertainty_regime == "plan_cache_hit":
            self._fs_counters["automem_cost/dispatch_cache_hit"] += 1
            cache = self._cache_for(memory)
            assert cache is not None
            plan_cache: dict[str, dict] = getattr(cache, "plan_cache", {})
            plan = plan_cache[_normalize_question(question)]

            observations = self._dispatch_actions(memory, question, plan)
            if self.enable_aggregate:
                agg_obs, used = self._apply_aggregate_ops(memory, plan, observations)
                if used:
                    observations = observations + agg_obs
            context = self._render_context(memory, question, plan, observations)
            self._counters.record_retrieval(
                candidates_scored=len(self._all_search_records(memory)),
                evidence_injected=len(observations),
                context_tokens=self._count_tokens(context),
            )
            self._publish_fs_counters()
            return context
        if decision.uncertainty_regime == "dump_cost_dominated":
            self._fs_counters["automem_cost/dispatch_dump_dominated"] += 1
            self._fs_counters["u2_dump_fired"] += 1
            ans = self._dump_and_answer(memory, question)
            self._publish_fs_counters()
            if ans:
                return ans

        self._fs_counters["automem_cost/dispatch_cascade_default"] += 1
        result = super()._answer_inner(memory, question)

        self._maybe_cache_plan(memory, question)
        return result

    def _maybe_cache_plan(self, memory: DCIMemory, question: str) -> None:
        if not (self.enable_cost_gates and self.dispatch_cache_enabled):
            return
        cache = self._cache_for(memory)
        if cache is None:
            return
        plan_cache: dict[str, dict] = getattr(cache, "plan_cache", None) or {}
        first = getattr(self, "_first_plan", None)
        if first is None:
            return
        key = _normalize_question(question)
        if key not in plan_cache and isinstance(first, dict):
            plan_cache[key] = {k: first.get(k) for k in ("queries", "filters", "read_ids", "aggregate_ops", "graph_queries", "use_dump", "rationale")}
            setattr(cache, "plan_cache", plan_cache)

        self._first_plan = None

    def _spc_filtered_records(self, memory: DCIMemory, question: str) -> list[DCIRecord]:
        """Return a token-budgeted, deduped union of:
          - top-K_e by embedding cosine
          - top-K_l by lexical entity match
          - top-N most recent
        Ranking key: max(embed_score_norm, lex_score_norm, recency_score_norm).
        """
        records = self._all_search_records(memory)
        if not records:
            return []
        cache = self._cache_for(memory)

        corpus_id = getattr(memory, "_corpus_id", None)
        if corpus_id in self.spc_disabled_corpora:
            return None                              

        embed_scores: dict[str, float] = {}
        if cache is not None and getattr(cache, "embeddings", None):
            q_vec_list = self._embed_batch([question])
            if q_vec_list:
                q_vec = q_vec_list[0]
                emb = cache.embeddings                              
                for r in records:
                    v = emb.get(r.record_id)
                    if v is not None:
                        embed_scores[r.record_id] = _cosine(q_vec, v)

        ents = [e.lower() for e in _extract_entities(question)]
        lex_scores: dict[str, float] = {}
        if ents:
            for r in records:
                hay = ((r.compact or "") + " " + (r.title or "")).lower()
                hits = sum(1 for e in ents if e in hay)
                if hits:
                    lex_scores[r.record_id] = float(hits) / float(len(ents))

        rec_scores: dict[str, float] = {}
        n = len(records)
        for i, r in enumerate(records):
            rec_scores[r.record_id] = float(i) / float(max(1, n - 1))

        emb_top = sorted(embed_scores.items(), key=lambda kv: -kv[1])[: self.spc_K_e]
        lex_top = sorted(lex_scores.items(), key=lambda kv: -kv[1])[: self.spc_K_l]
        recent_top = [(records[n - 1 - i].record_id, 1.0) for i in range(min(self.spc_recent_top, n))]
        candidates: dict[str, float] = {}
        for rid, s in emb_top:
            candidates[rid] = max(candidates.get(rid, 0.0), s)
        for rid, s in lex_top:
            candidates[rid] = max(candidates.get(rid, 0.0), s)
        for rid, _ in recent_top:
            candidates[rid] = max(candidates.get(rid, 0.0), 0.99)                            
        if not candidates:
            return records                                          

        rid_to_idx = {r.record_id: i for i, r in enumerate(records)}
        ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], rid_to_idx.get(kv[0], 1 << 30)))

        budget = max(500, self.spc_token_budget)
        kept: list[DCIRecord] = []
        used = 0
        rec_by_id = {r.record_id: r for r in records}
        for rid, _ in ranked:
            r = rec_by_id.get(rid)
            if r is None:
                continue
            preview = (r.compact or self._compact_record(r.content))[: self.spc_preview_chars]
            cost = len(preview) // 4 + 12                                           
            if used + cost > budget and kept:
                break
            kept.append(r)
            used += cost
        return kept

    def _plan_search(
        self,
        memory: DCIMemory,
        question: str,
        *,
        retry_hint: str | None = None,
    ) -> dict[str, Any]:
        if not (self.enable_cost_gates and self.spc_enabled and self.enable_memory_primitives):
            plan = super()._plan_search(memory, question, retry_hint=retry_hint)
            if self.enable_cost_gates and retry_hint is None:
                self._first_plan = plan
            return plan

        if self.kvcache_friendly:
            records_filtered: list[DCIRecord] | None = []                                                        
        else:
            records_filtered = self._spc_filtered_records(memory, question)
            if records_filtered is None:

                self._fs_counters["automem_cost/spc_skipped"] += 1
                plan = super()._plan_search(memory, question, retry_hint=retry_hint)
                if retry_hint is None:
                    self._first_plan = plan
                return plan
            if not records_filtered:
                self._fs_counters["automem_cost/spc_skipped"] += 1
                plan = super()._plan_search(memory, question, retry_hint=retry_hint)
                if retry_hint is None:
                    self._first_plan = plan
                return plan
            self._fs_counters["automem_cost/spc_applied"] += 1
            self._fs_counters["automem_cost/spc_catalog_records"] = max(
                self._fs_counters["automem_cost/spc_catalog_records"], len(records_filtered)
            )

        catalog = [
            {
                "id": r.record_id,
                "title": r.title,
                "preview": (r.compact or self._compact_record(r.content))[: self.spc_preview_chars],
            }
            for r in records_filtered
        ]

        entity_hint = ""
        if self.enable_entity_grep:
            entities = self._get_or_build_entity_table(memory)
            if entities:
                first6 = list(entities.items())[:6]
                tokens = [f"{k}({','.join(v[:2])})" for k, v in first6]
                entity_hint = "Entities: " + "; ".join(tokens)

        total_chars = sum(len(r.content) for r in memory.records) if memory.records else 0
        n_records = len(memory.records) if memory.records else 0
        dump_hint = self._dump_tool_hint(total_chars, n_records) if self.enable_adaptive_dump else ""

        agg_hint = ""
        if self.enable_aggregate:
            agg_hint = (
                "You may request post-retrieval aggregation via 'aggregate_ops': "
                "{op: intersect|union|temporal_sort|count, args: [q_a, q_b?]}."
            )
        graph_hint = ""
        if self.enable_graph_query and self._get_graph_driver(memory) is not None:
            graph_hint = (
                "You may issue typed graph queries via 'graph_queries' (Cypher read-only, "
                "must include `{ns: $ns}` filters)."
            )

        plan_keys = "use_dump (bool), queries (list of strings), filters (list of strings), read_ids (list of record ids), aggregate_ops (list, optional)"
        if self.enable_graph_query and self._get_graph_driver(memory) is not None:
            plan_keys += ", graph_queries (list of Cypher strings, optional)"
        plan_keys += ", rationale"

        action_menu = (
            "You control a direct-corpus interaction harness. Pick actions:\n"
            "- `queries`/`filters` (rg-style regex; cheap)\n"
            "- `read_ids` (read full record content)\n"
            "- `aggregate_ops`\n"
            "- `graph_queries`\n"
            "- `use_dump: true` (single-LLM-call over the whole corpus)\n"
        )
        catalog_json = json.dumps(catalog, ensure_ascii=False)
        task_label = f"Task: {memory.task or 'unknown'}"
        summary_block = f"Prior summary: {memory.summary or 'None'}"
        stats_block = f"Corpus stats: {n_records} records, {total_chars} chars."
        format_instr = f"Return ONLY JSON with keys: {plan_keys}."

        if self.kvcache_friendly:
            self._fs_counters["automem_cost/kvcache_friendly_prompts"] += 1

            stable_records = sorted(
                self._all_search_records(memory)[: 80],
                key=lambda r: r.record_id,
            )
            stable_catalog = [
                {
                    "id": r.record_id,
                    "title": r.title,
                    "preview": (r.compact or self._compact_record(r.content))[: self.spc_preview_chars],
                }
                for r in stable_records
            ]
            stable_catalog_json = json.dumps(stable_catalog, ensure_ascii=False)
            corpus_stable_parts = [
                action_menu,
                task_label,
                summary_block,
                stats_block,
                dump_hint,
                entity_hint,
                agg_hint,
                graph_hint,
                f"Record catalog JSON:\n{stable_catalog_json}",
                format_instr,
            ]
            per_query_parts = [f"Question: {question}"]
            if retry_hint:
                per_query_parts.append(f"Planner hint: {retry_hint}")
            prompt = "\n\n".join(p for p in corpus_stable_parts + per_query_parts if p)
        else:

            parts = [
                action_menu,
                task_label,
                summary_block,
                f"Question: {question}",
                stats_block,
                dump_hint,
                entity_hint,
                agg_hint,
                graph_hint,
            ]
            if retry_hint:
                parts.append(f"Planner hint: {retry_hint}")
            parts.append(f"Record catalog JSON:\n{catalog_json}")
            parts.append(format_instr)
            prompt = "\n\n".join(p for p in parts if p)

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
        if retry_hint is None:

            self._first_plan = plan
        return plan

    def _gs_predicates(
        self,
        memory: DCIMemory,
        question: str,
        observations: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """Return (sufficient, regime). regime in {'predicates_pass',
        'predicate_failed_pN', 'empty_observations'}."""
        if not observations:
            return False, "empty_observations"

        ents = [e.lower() for e in _extract_entities(question)]
        if ents:
            haystack = " ".join(
                ((o.get("title", "") or "") + " " + " ".join(o.get("matches", []) or []) + " " + (o.get("read", "") or ""))
                for o in observations
            ).lower()
            missing = [e for e in ents if e not in haystack]
            if missing:
                return False, "predicate_failed_p1"

        scores = sorted([float(o.get("score", 0.0) or 0.0) for o in observations], reverse=True)
        top = scores[0] if scores else 0.0
        second = scores[1] if len(scores) > 1 else 0.0
        if not (top >= self.gs_theta_top or (top - second) >= self.gs_theta_margin):
            return False, "predicate_failed_p2"

        q_low = (question or "").lower()
        if any(k in q_low for k in _AGG_KEYWORDS):
            distinct_records = {o.get("record_id") for o in observations if o.get("record_id")}
            if len(distinct_records) < 2:
                return False, "predicate_failed_p3"

        renderable = False
        for o in observations:
            for m in (o.get("matches") or []):
                if isinstance(m, str) and len(m.strip()) >= 20:
                    renderable = True
                    break
            if renderable:
                break
            txt = o.get("read") or ""
            if isinstance(txt, str) and len(txt.strip()) >= 20:
                renderable = True
                break
        if not renderable:
            return False, "predicate_failed_p4"

        return True, "predicates_pass"

    def _judge_sufficient(
        self,
        memory: DCIMemory,
        question: str,
        plan: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not (self.enable_cost_gates and self.gs_enabled):
            return super()._judge_sufficient(memory, question, plan, observations)
        sufficient, regime = self._gs_predicates(memory, question, observations)
        if regime == "predicates_pass":
            self._fs_counters["automem_cost/gs_tier1_pass"] += 1
            return {"sufficient": True, "why_more": ""}
        if regime == "empty_observations":
            self._fs_counters["automem_cost/gs_empty_obs_fallback"] += 1
            return super()._judge_sufficient(memory, question, plan, observations)

        key = regime.replace("predicate_failed_", "gs_tier1_fail_")
        self._fs_counters[f"automem_cost/{key}"] = self._fs_counters.get(f"automem_cost/{key}", 0) + 1
        self._fs_counters["automem_cost/gs_tier2_fired"] += 1
        return super()._judge_sufficient(memory, question, plan, observations)

class AutoMemCostSPCOnly(AutoMemCostMethod):
    name = "automem_cost_spc"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("gs_enabled", False)
        kwargs.setdefault("dispatch_enabled", False)
        super().__init__(**kwargs)

class AutoMemCostSPCGS(AutoMemCostMethod):
    name = "automem_cost_spc_gs"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("dispatch_enabled", False)
        super().__init__(**kwargs)

class AutoMemCostKVFriendly(AutoMemCostMethod):
    """KV-cache-friendly variant: prompt ordering puts corpus-stable content
    first; question last. May send more tokens (no SPC compression) but should
    hit prefix cache on repeated queries against the same corpus.
    """

    name = "automem_kvcache"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("kvcache_friendly", True)

        kwargs.setdefault("spc_token_budget", 12000)
        kwargs.setdefault("spc_K_e", 60)
        super().__init__(**kwargs)
