"""AutoMem-KV: KV-cache-friendly variant of AutoMem / AutoMemGraph.

Goal: cut effective serving cost (after vLLM prefix-cache reuse) without
any per-task hyperparameter tuning. We use TWO architectural levers:

1. **Two-message chat structure with a fixed system message.** Each LLM
   call is split into a ``system`` message holding the stable AutoMem
   instruction block, and a ``user`` message holding everything else.
   The system message is identical across all calls (planner / judge /
   answer / dump / entity-table) for ALL queries, so the vLLM prefix
   cache caches it once per server.

2. **Tail-variable user-message ordering.** Inside the user message,
   stable-across-queries content (corpus catalog, graph schema, task
   label, action menu repeat) comes FIRST, and per-query content
   (question, retry hint, growing trace) comes LAST. With this layout
   the vLLM token-level prefix cache reuses every byte up to the first
   per-query byte, so cache hit grows with the number of queries asked
   against the same corpus.

Concretely we override only the heavy paths (`_plan_search`,
`_judge_evidence_via_llm`) to use the two-message API directly. Other
paths (entity-table build, dump-and-answer) keep their single-message
behaviour but still benefit from the stable system message that vLLM
caches at the chat-template level.

No new hyperparameters are introduced. All thresholds (top_k, dump_threshold,
max_context_chars, etc.) inherit from AutoMemMethod/AutoMemGraphMethod
unchanged and are NOT per-benchmark tuned.

Registry IDs:
  ``automem_kv``        : KV-friendly plain AutoMem (no graph backend)
  ``automem_graph_kv``  : KV-friendly AutoMemGraph
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from agentmem.methods.automem import AutoMemMethod, AutoMemGraphMethod

_STABLE_SYSTEM = (
    "You are an agentic memory harness controlling a direct-corpus interaction loop. "
    "You will see, in the user message, a corpus catalog, a task label, optional "
    "schema hints, and a question. You must decide which retrieval actions to call "
    "and return them as a JSON plan; an executor will run the plan and return "
    "evidence; a judge will inspect the evidence; if sufficient, you will write a "
    "final answer.\n"
    "\n"
    "Action menu (use only those marked available in the user message):\n"
    "  - queries / filters : regex/lexical search over records (cheapest)\n"
    "  - read_ids          : read full content of named records when ids are known\n"
    "  - aggregate_ops     : intersect / union / temporal_sort / count over prior hits\n"
    "  - graph_queries     : read-only typed query against a preloaded subgraph\n"
    "  - use_dump          : single LLM call over the whole corpus; only when small\n"
    "\n"
    "Output rules:\n"
    "  - Return exactly one JSON object. No prose around the JSON.\n"
    "  - Never invent record_ids; use only ones from the catalog.\n"
    "  - Prefer cheap regex with multiple variants over expensive whole-corpus dumps.\n"
)

def _chat(
    *, base_url: str, model: str, api_key: str,
    messages: list[dict[str, str]],
    temperature: float, max_tokens: int, timeout: float,
) -> tuple[str, dict[str, Any]]:
    """Send a 2-message chat request and return (text, raw_usage_dict)."""
    url = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return "", {}
    choices = body.get("choices") or []
    if not choices:
        return "", body.get("usage") or {}
    text = ((choices[0].get("message") or {}).get("content") or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    return text, body.get("usage") or {}

class _KVMixin:
    """Override LLM call to use 2-message chat with a fixed stable system.

    Single-message fallback (no per-call modifications) on paths we have
    not converted, so behavior outside the converted paths is unchanged
    except for an added stable system block — which is the same string
    across all variants and all benchmarks, so any cross-method comparison
    sees the same system overhead.
    """

    def _llm_text(self, prompt: str, *, phase: str) -> str:                          
        if not self.llm_base_url or not self.llm_model:                              
            return ""
        text, usage = _chat(
            base_url=self.llm_base_url,                              
            model=self.llm_model,                              
            api_key=self.llm_api_key,                              
            messages=[
                {"role": "system", "content": _STABLE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,                              
            max_tokens=self.controller_max_tokens,                              
            timeout=self.request_timeout,                              
        )

        prompt_tokens = int(usage.get("prompt_tokens") or len((prompt or "").split()))
        completion_tokens = int(usage.get("completion_tokens") or 0)

        self._counters.record_llm_call(                              
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            phase="build" if phase == "build" else "query",
        )

        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
        if hasattr(self, "_fs_counters"):
            self._fs_counters.setdefault("kv_cached_prompt_tokens", 0)
            self._fs_counters.setdefault("kv_total_prompt_tokens", 0)
            self._fs_counters["kv_cached_prompt_tokens"] += cached
            self._fs_counters["kv_total_prompt_tokens"] += prompt_tokens
        return text

    def _plan_search(self, memory, question: str, retry_hint: str = ""):                          
        """Tail-variable rebuild of the planner prompt.

        Layout (top → bottom of user message):
          1. Action availability flags (stable per method config)
          2. Graph schema hint, if graph_query enabled (stable per method config)
          3. Task label (stable per episode)
          4. Corpus stats and record catalog JSON (stable per episode)
          5. (Variable) Compacted prior summary
          6. (Variable) Question
          7. (Variable) Retry / planner hint

        Then we call the chat endpoint with `_STABLE_SYSTEM` as system.
        """

        search_records = self._all_search_records(memory)                              
        catalog = [
            {
                "id": r.record_id,
                "title": r.title,
                "preview": r.compact or self._compact_record(r.content),                              
            }
            for r in search_records[: min(len(search_records), 80)]
        ]
        total_chars = sum(len(r.content) for r in memory.records) if memory.records else 0
        n_records = len(memory.records) if memory.records else 0

        graph_on = bool(
            getattr(self, "enable_graph_query", False)
            and self._get_graph_driver(memory) is not None                              
        )
        agg_on = bool(getattr(self, "enable_aggregate", False))
        dump_on = bool(getattr(self, "enable_adaptive_dump", False))

        head_lines = [
            "Available actions for this call:",
            "  queries, filters, read_ids" + (", aggregate_ops" if agg_on else "")
                + (", graph_queries" if graph_on else "")
                + (", use_dump" if dump_on else ""),
        ]
        if graph_on:
            schema = getattr(self, "graph_schema_hint", "") or (
                "Graph schema (Memgraph, openCypher):\n"
                "  Node labels and properties depend on the corpus. "
                "Use a 'ns' property to scope nodes to the current corpus. "
                "Queries must be read-only (MATCH/RETURN; no CREATE/SET/DELETE)."
            )
            head_lines.append(schema)
        head_lines.append(f"Task label:\n{memory.task or 'unknown'}")
        head_lines.append(f"Corpus stats: {n_records} records, {total_chars} chars total.")
        head_lines.append(f"Record catalog JSON:\n{json.dumps(catalog, ensure_ascii=False)}")

        tail_lines = []
        summary = getattr(memory, "summary", "") or "None"
        tail_lines.append(f"Compacted prior summary:\n{summary}")
        tail_lines.append(f"Question:\n{question}")
        if retry_hint:
            tail_lines.append(f"Planner hint: {retry_hint}")
        plan_keys = "use_dump (bool), queries (list), filters (list), read_ids (list), aggregate_ops (list, optional)"
        if graph_on:
            plan_keys += ", graph_queries (list, optional)"
        plan_keys += ", rationale (string)"
        tail_lines.append(f"Return ONLY one JSON object with keys: {plan_keys}.")

        user_prompt = "\n\n".join(head_lines + tail_lines)
        raw = self._llm_text(user_prompt, phase="query")
        plan = self._parse_json(raw) or {}                              
        if not isinstance(plan, dict):
            plan = {}
        queries = [str(x).strip() for x in plan.get("queries", []) if str(x).strip()]
        if not queries:
            queries = self._fallback_queries(question)                              
        if getattr(self, "enable_entity_grep", False):
            queries = self._expand_with_entities(memory, queries)                              
        plan["queries"] = queries[:8]
        plan["filters"] = [str(x).strip() for x in plan.get("filters", []) if str(x).strip()][:6]
        plan["read_ids"] = [str(x).strip() for x in plan.get("read_ids", []) if str(x).strip()][: self.top_k]                              
        return plan

class AutoMemKVMethod(_KVMixin, AutoMemMethod):
    """Plain AutoMem with KV-friendly 2-message chat + tail-variable planner."""

    name = "automem_kv"

class AutoMemGraphKVMethod(_KVMixin, AutoMemGraphMethod):
    """AutoMemGraph with KV-friendly 2-message chat + tail-variable planner."""

    name = "automem_graph_kv"

class AutoMemGraphSinglePassMethod(AutoMemGraphMethod):
    """AutoMemGraph stripped to single-pass: 1 planner call, no judge, no dump."""

    name = "automem_graph_single"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("enable_adaptive_dump", False)
        kwargs.setdefault("max_agentic_iters", 0)
        super().__init__(**kwargs)

class AutoMemGraphOneRetryMethod(AutoMemGraphMethod):
    """AutoMemGraph with at most one judge-driven retry. No dump branch."""

    name = "automem_graph_oneretry"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("enable_adaptive_dump", False)
        kwargs.setdefault("max_agentic_iters", 1)
        super().__init__(**kwargs)

class AutoMemSinglePassMethod(AutoMemMethod):
    """Plain AutoMem stripped to single-pass: 1 planner call, no judge, no dump."""

    name = "automem_single"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("enable_adaptive_dump", False)
        kwargs.setdefault("max_agentic_iters", 0)
        super().__init__(**kwargs)

class AutoMemOneRetryMethod(AutoMemMethod):
    """Plain AutoMem with at most one judge-driven retry. No dump branch."""

    name = "automem_oneretry"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("enable_adaptive_dump", False)
        kwargs.setdefault("max_agentic_iters", 1)
        super().__init__(**kwargs)

_GRAPH_TEMPLATES: dict[str, tuple[str, str, list[str]]] = {
    "speaker_mentions": (
        "find memories spoken by SPEAKER containing KEYWORD",
        "MATCH (m:Memory)-[:SAID_BY]->(sp:Speaker)\n"
        "WHERE m.ns = $ns AND toLower(sp.name) = toLower($arg0) "
        "AND toLower(m.text) CONTAINS toLower($arg1)\n"
        "RETURN m.text AS text, m.time AS time, m.weekday AS day, sp.name AS speaker\n"
        "ORDER BY m.ftime LIMIT 10",
        ["speaker", "keyword"],
    ),
    "keyword_search": (
        "find memories containing KEYWORD (any speaker)",
        "MATCH (m:Memory)-[:SAID_BY]->(sp:Speaker)\n"
        "WHERE m.ns = $ns AND toLower(m.text) CONTAINS toLower($arg0)\n"
        "RETURN m.text AS text, m.time AS time, sp.name AS speaker\n"
        "ORDER BY m.ftime LIMIT 10",
        ["keyword"],
    ),
    "follow_chain": (
        "fetch the N memories following a given memory by id (temporal chain)",
        "MATCH (m0:Memory {ns: $ns, vid: $arg0})\n"
        "MATCH (m0)-[:FOLLOWS*1..10]->(m:Memory)\n"
        "WHERE m.ns = $ns\n"
        "WITH m ORDER BY m.ftime LIMIT toInteger($arg1)\n"
        "MATCH (m)-[:SAID_BY]->(sp:Speaker)\n"
        "RETURN m.text AS text, m.time AS time, sp.name AS speaker",
        ["mem_vid", "n"],
    ),
    "speakers_in_episode": (
        "list distinct speakers in the current episode",
        "MATCH (m:Memory)-[:SAID_BY]->(sp:Speaker)\n"
        "WHERE m.ns = $ns\n"
        "RETURN DISTINCT sp.name AS name, count(m) AS turns\n"
        "ORDER BY turns DESC",
        [],
    ),
    "speaker_first_last": (
        "first and last memory by SPEAKER",
        "MATCH (m:Memory)-[:SAID_BY]->(sp:Speaker)\n"
        "WHERE m.ns = $ns AND toLower(sp.name) = toLower($arg0)\n"
        "WITH m ORDER BY m.ftime\n"
        "WITH collect(m) AS ms\n"
        "RETURN ms[0].time AS first_time, ms[0].text AS first_text, "
        "ms[-1].time AS last_time, ms[-1].text AS last_text",
        ["speaker"],
    ),
    "between_dates": (
        "memories between FTIME_START and FTIME_END (unix seconds)",
        "MATCH (m:Memory)-[:SAID_BY]->(sp:Speaker)\n"
        "WHERE m.ns = $ns AND m.ftime >= toFloat($arg0) AND m.ftime <= toFloat($arg1)\n"
        "RETURN m.text AS text, m.time AS time, sp.name AS speaker\n"
        "ORDER BY m.ftime LIMIT 15",
        ["start_ftime", "end_ftime"],
    ),
}

def _render_template_block() -> str:
    """Render the template library as a short prompt block for the planner."""
    lines = ["Graph query templates (emit objects {t, args}):"]
    for tid, (desc, _cypher, params) in _GRAPH_TEMPLATES.items():
        argspec = ", ".join(params) if params else "(none)"
        lines.append(f"  - {tid}({argspec}) : {desc}")
    lines.append("All queries are scoped to the current corpus automatically.")
    return "\n".join(lines)

def _execute_template(driver: Any, template_id: str, args: list[Any], ns: str, max_rows: int) -> tuple[list[dict[str, Any]], str | None]:
    """Run a template against the graph and return (rows, error_string)."""
    spec = _GRAPH_TEMPLATES.get(template_id)
    if spec is None:
        return [], f"unknown template: {template_id}"
    _, cypher, params = spec
    if len(args) < len(params):
        return [], f"template {template_id} needs {len(params)} args, got {len(args)}"
    kwargs = {f"arg{i}": a for i, a in enumerate(args)}
    kwargs["ns"] = ns
    try:
        with driver.session() as sess:
            rows = [r.data() for r in sess.run(cypher, **kwargs)]
        return rows[:max_rows], None
    except Exception as exc:
        return [], f"cypher error: {str(exc)[:200]}"

class AutoMemGraphTemplateMethod(AutoMemGraphMethod):
    """AutoMemGraph using vetted template DSL instead of free-form Cypher.

    No new hyperparameters. The template list is fixed and generic over the
    LightMem-graph schema; the planner picks templates the same way it
    picks `rg` queries.
    """

    name = "automem_graph_template"

    def __init__(self, **kwargs: Any) -> None:

        kwargs.setdefault("graph_schema_hint", _render_template_block())
        super().__init__(**kwargs)

    def _execute_graph_queries(self, memory: Any, plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Replace base implementation: route template-style invocations
        through ``_execute_template``. Fall back to base for any free-form
        Cypher string still emitted (graceful upgrade)."""
        items = plan.get("graph_queries") or []
        if not items:
            return []
        driver = self._get_graph_driver(memory)                              
        if driver is None:
            return []
        ns = self._graph_namespace(memory)                              
        obs: list[dict[str, Any]] = []
        for idx, item in enumerate(items[:4]):
            self._fs_counters["u4_graph_query_calls"] += 1                              
            if isinstance(item, dict):
                tid = str(item.get("t") or item.get("template") or "")
                args = list(item.get("args") or [])
                rows, err = _execute_template(driver, tid, args, ns, self.graph_max_rows)                              
                if err:
                    self._fs_counters["u4_graph_query_errors"] += 1                              
                    obs.append({
                        "record_id": f"graph_q{idx + 1}_err",
                        "title": f"template:{tid}",
                        "score": 0.0,
                        "matches": [f"ERROR: {err}"],
                        "read": "",
                        "kind": "graph_err",
                    })
                    continue
                self._fs_counters["u4_graph_rows_returned"] += len(rows)                              
                summary_lines = [self._format_graph_row(r) for r in rows[:10]]                              
                obs.append({
                    "record_id": f"graph_q{idx + 1}",
                    "title": f"template:{tid}({', '.join(str(a) for a in args)})",
                    "score": 1.0,
                    "matches": summary_lines or ["(empty result)"],
                    "read": "",
                    "kind": "graph",
                    "template": tid,
                    "args": args,
                    "rows": rows,
                })
            elif isinstance(item, str):

                try:
                    with driver.session() as sess:
                        rows = [r.data() for r in sess.run(item, ns=ns)]
                    rows = rows[: self.graph_max_rows]                              
                    self._fs_counters["u4_graph_rows_returned"] += len(rows)                              
                    summary_lines = [self._format_graph_row(r) for r in rows[:10]]                              
                    obs.append({
                        "record_id": f"graph_q{idx + 1}",
                        "title": f"cypher:{item[:80]}",
                        "score": 1.0,
                        "matches": summary_lines or ["(empty result)"],
                        "read": "",
                        "kind": "graph",
                        "cypher": item,
                        "rows": rows,
                    })
                except Exception as exc:
                    self._fs_counters["u4_graph_query_errors"] += 1                              
                    obs.append({
                        "record_id": f"graph_q{idx + 1}_err",
                        "title": "cypher (error)",
                        "score": 0.0,
                        "matches": [f"ERROR: {str(exc)[:200]}"],
                        "read": "",
                        "kind": "graph_err",
                    })
        return obs
