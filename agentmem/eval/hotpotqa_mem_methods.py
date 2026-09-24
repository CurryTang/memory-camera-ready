"""Repo-owned HotpotQA runners for MemRL and corpus-backed methods."""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import tempfile
import time
from collections import Counter
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional

from data import DatasetFactory

from agentmem.eval.method_eval_utils import (
    aggregate_rows_resource_usage,
    append_jsonl,
    load_jsonl,
    utcnow_iso,
    write_json,
)
from agentmem.eval.locomo_runner.provider import _history_artifact_path, _shared_provider_kwargs
from agentmem.backends import EpisodicMemoryStore
from agentmem.providers.base import Message
from agentmem.providers.openai_compat import OpenAICompatibleProvider
from agentmem.memrl import MemRLConfig, MemRLRuntimeEngine
from agentmem.memrl.task_adaptation import (
    build_hotpot_answer_prompt,
    build_hotpot_memrl_queries,
    build_hotpot_qa_memory,
    chunk_hotpot_documents,
    compute_memrl_reward,
    format_hotpot_context,
    supporting_fact_bonus,
)

_SPECIAL_ANSWERS = {"yes", "no", "noanswer"}

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def _token_f1(pred: str, gold: str) -> float:
    norm_pred = _normalize(pred)
    norm_gold = _normalize(gold)
    if norm_pred in _SPECIAL_ANSWERS or norm_gold in _SPECIAL_ANSWERS:
        return float(norm_pred == norm_gold)
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    tp = sum(min(pred_counts[t], gold_counts[t]) for t in common)
    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def _exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))

def _build_docs(sample: dict[str, Any]) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    context = sample.get("context", [])

    if isinstance(context, dict):
        titles = context.get("title", [])
        sents = context.get("sentences", [])
        pairs = list(zip(titles, sents))
    else:
        pairs = list(context)
    for item in pairs:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title, sentences = item[0], item[1]
        title_text = str(title or "").strip() or "Untitled"
        if isinstance(sentences, list):
            body = " ".join(str(s).strip() for s in sentences if str(s).strip())
        else:
            body = str(sentences or "").strip()
        if body:
            docs.append((title_text, body))
    return docs

def _corpus_to_context(docs: Iterable[tuple[str, str]]) -> str:
    parts = []
    for index, (title, text) in enumerate(docs, start=1):
        parts.append(f"Document {index} — {title}:\n{text}")
    return "\n\n".join(parts)

def _hotpot_prompt(question: str, context: str) -> str:
    return build_hotpot_answer_prompt(question=question, context=context)

def _stringify_prediction_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                part = item.get("text") or item.get("content") or item.get("value")
                if part is not None:
                    parts.append(_stringify_prediction_content(part))
            else:
                part = getattr(item, "text", None) or getattr(item, "content", None)
                if part is not None:
                    parts.append(_stringify_prediction_content(part))
        flattened = "\n".join(part for part in parts if part.strip()).strip()
        return flattened if flattened else str(value)
    return str(value)

def _prediction_text_from_response(value: Any) -> str:
    candidates: list[Any] = []
    if hasattr(value, "content"):
        candidates.append(getattr(value, "content", None))
    if hasattr(value, "text"):
        candidates.append(getattr(value, "text", None))
    raw = getattr(value, "raw", None)
    if raw is not None:
        choice = raw.choices[0] if getattr(raw, "choices", None) else None
        message = getattr(choice, "message", None)
        candidates.extend([
            getattr(message, "content", None),
            getattr(message, "text", None),
        ])
    candidates.append(value)
    for candidate in candidates:
        text = _stringify_prediction_content(candidate).strip()
        if text:
            return text
    return ""

def _normalize_prediction(text: Any) -> str:
    raw = _prediction_text_from_response(text)
    raw = re.sub(r"<think>.*?</think>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    raw = re.sub(r"<tool_call>.*?</tool_call>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    raw = raw.strip().strip("`").strip("\"'")
    if not raw:
        return ""
    raw_fallback = raw.strip()

    for marker in reversed(list(re.finditer(r"(?i)#{2,}\s*answer\s*:\s*", raw))):
        tail = raw[marker.end() :].strip()
        if tail:
            raw = tail
            break

    for pattern in (
        r"(?im)^\s*final\s+answer\s*:\s*(.+?)\s*$",
        r"(?im)^\s*answer\s*:\s*(.+?)\s*$",
    ):
        matches = [m.group(1).strip() for m in re.finditer(pattern, raw) if m.group(1).strip()]
        if matches:
            raw = matches[-1]
            break

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", raw)
        if paragraph.strip()
    ]
    cleaned = paragraphs[-1] if paragraphs else raw.strip()
    lines = [
        line.strip()
        for line in cleaned.splitlines()
        if line.strip() and not re.fullmatch(r"(?i)#{2,}\s*(reasoning|answer)\s*:?", line.strip())
    ]
    if lines:
        cleaned = lines[-1]
    elif cleaned.strip():
        cleaned = cleaned.strip()
    else:
        cleaned = raw_fallback

    cleaned = re.sub(r"(?i)^#{2,}\s*(reasoning|answer)\s*:\s*", "", cleaned).strip()
    cleaned = re.sub(r"(?i)^reasoning\s*:\s*", "", cleaned).strip()
    cleaned = cleaned.strip().strip("\"'")
    lowered = cleaned.lower()

    if re.fullmatch(r"(?i)yes[.!]?", cleaned):
        return "yes"
    if re.fullmatch(r"(?i)no[.!]?", cleaned):
        return "no"
    if re.fullmatch(r"(?i)no\s*answer[.!]?", cleaned):
        return "noanswer"
    for prefix in ("the answer is ", "answer is ", "final answer: ", "answer: ", "it is "):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            lowered = cleaned.lower()
            break

    cleaned = re.sub(r"(?i)^(yes|no)[,;:]\s+", "", cleaned).strip()
    cleaned = re.split(r"[.!?]\s+", cleaned, maxsplit=1)[0].strip()
    return cleaned.strip().strip("\"'") or raw_fallback

def _merge_hotpot_candidates(
    retrieval_results: list[Any],
    *,
    topk: int,
) -> list[Any]:
    merged: OrderedDict[str, Any] = OrderedDict()
    for retrieval in retrieval_results:
        if retrieval is None:
            continue
        candidates = list(getattr(retrieval, "candidates", []) or [])
        for cand in candidates:
            memory_id = str(getattr(cand, "memory_id", "") or "")
            if not memory_id:
                continue
            existing = merged.get(memory_id)
            if existing is None or float(getattr(cand, "fused_score", 0.0)) > float(
                getattr(existing, "fused_score", 0.0)
            ):
                merged[memory_id] = cand
    ranked = sorted(
        merged.values(),
        key=lambda cand: (
            float(getattr(cand, "fused_score", 0.0)),
            float(getattr(cand, "utility", 0.0)),
            float(getattr(cand, "similarity", 0.0)),
        ),
        reverse=True,
    )
    return ranked[: max(1, int(topk))]

def _load_hotpot_samples(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    dataset = DatasetFactory.create("hotpotqa", datasets_dir=args.datasets_dir)
    dataset_path = Path(
        dataset.ensure_data(
            path=args.dataset_path,
            variant=args.hotpotqa_variant,
            auto_download=args.download,
        )
    )
    all_samples = dataset.load_samples(dataset_path)
    start = max(0, int(args.sample_start))
    stop = len(all_samples) if args.max_samples is None else min(len(all_samples), start + int(args.max_samples))
    return dataset_path, all_samples[start:stop]

def _prepare_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    out_dir = (Path(args.output_dir) / args.run_tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_label = "all" if args.max_samples is None else str(args.max_samples)
    stem = f"{args.method}_s{args.sample_start}_n{n_label}"
    return (
        out_dir / f"{stem}.jsonl",
        out_dir / f"{stem}_trajectories.jsonl",
        out_dir / f"{stem}_summary.json",
    )

def run_memrl_hotpotqa(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path, samples = _load_hotpot_samples(args)
    rows_path, traj_path, summary_path = _prepare_output_paths(args)
    completed = {str(row.get("sample_id", "")) for row in load_jsonl(rows_path)}

    provider = OpenAICompatibleProvider(
        api_key=args.memrl_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
        model=args.memrl_model,
        base_url=args.memrl_base_url,
        **_shared_provider_kwargs(args, task="memrl-hotpotqa"),
    )
    cfg = MemRLConfig(
        phase1_topk=max(1, int(args.memrl_phase1_topk)),
        topk=max(1, int(args.memrl_topk)),
        alpha=float(args.memrl_alpha),
        gamma=float(args.memrl_gamma),
        epsilon=float(args.memrl_epsilon),
        similarity_weight=float(args.memrl_similarity_weight),
        utility_weight=float(args.memrl_utility_weight),
    )

    for sample_offset, sample in enumerate(samples):
        sample_index = int(args.sample_start) + sample_offset
        sample_id = str(sample.get("_id") or sample.get("id") or sample_index)
        if sample_id in completed:
            continue

        docs = _build_docs(sample)
        memory_units = chunk_hotpot_documents(sample)
        store = EpisodicMemoryStore()
        runtime = MemRLRuntimeEngine(store=store, config=cfg)
        build_start = time.perf_counter()
        for unit_index, unit in enumerate(memory_units):
            title = str(unit.get("title") or f"Document {unit_index}")
            text = str(unit.get("text") or "").strip()
            if not text:
                continue
            runtime.add_experience(
                intent=title,
                experience=text,
                success=True,
                metadata={
                    "source": "hotpot_doc",
                    "title": title,
                    "sample_id": sample_id,
                    "doc_index": unit_index,
                    "chunk_kind": unit.get("chunk_kind"),
                    "sentence_span": unit.get("sentence_span"),
                    "is_supporting_fact": bool(unit.get("is_supporting_fact")),
                    "full_content": f"{title}\n{text}",
                },
                task_id=f"{sample_id}_{unit.get('task_id') or f'unit_{unit_index}'}",
            )
        build_seconds = time.perf_counter() - build_start
        question = str(sample.get("question", "")).strip()
        gold = str(sample.get("answer", "")).strip()

        start = time.perf_counter()
        retrieval_start = time.perf_counter()
        retrieval_queries = build_hotpot_memrl_queries(question)
        retrieval_runs = [
            runtime.retrieve(
                query_text,
                phase1_topk=int(args.memrl_phase1_topk),
                topk=int(args.memrl_topk),
            )
            for query_text in retrieval_queries
        ]
        retrieve_sec = time.perf_counter() - retrieval_start
        chosen = _merge_hotpot_candidates(
            retrieval_runs,
            topk=max(1, int(args.memrl_topk)),
        )
        retrieved_units: list[dict[str, Any]] = []
        for cand in chosen:
            title = str((cand.metadata or {}).get("title") or f"Document {cand.memory_id}")
            body = str((cand.metadata or {}).get("full_content") or cand.content or "")
            retrieved_units.append(
                {
                    "memory_id": cand.memory_id,
                    "title": title,
                    "chunk_kind": (cand.metadata or {}).get("chunk_kind"),
                    "sentence_span": (cand.metadata or {}).get("sentence_span"),
                    "similarity": float(cand.similarity),
                    "utility": float(cand.utility),
                    "fused_score": float(cand.fused_score),
                    "is_supporting_fact": bool((cand.metadata or {}).get("is_supporting_fact")),
                }
            )
        context = format_hotpot_context(chosen)

        llm_start = time.perf_counter()
        resp = provider.chat(
            [
                Message(
                    role="user",
                    content=_hotpot_prompt(question=question, context=context),
                )
            ],
            temperature=0.0,
            max_tokens=int(args.max_response_tokens),
        )
        llm_sec = time.perf_counter() - llm_start
        prediction = _normalize_prediction(resp)
        latency = time.perf_counter() - start
        reward = compute_memrl_reward(
            prediction,
            gold,
            scheme=getattr(args, "memrl_reward_scheme", "token-f1"),
        )
        reward += supporting_fact_bonus(
            chosen,
            sample.get("supporting_facts") or [],
            bonus=0.2,
        )
        runtime.update_utilities(
            memory_ids=[str(cand.memory_id) for cand in chosen],
            reward=reward,
        )
        qa_intent, qa_experience, qa_metadata = build_hotpot_qa_memory(
            question=question,
            prediction=prediction,
            reference=gold,
            reward=reward,
            selected=chosen,
        )
        runtime.add_experience(
            intent=qa_intent,
            experience=qa_experience,
            success=bool(reward >= 0.999),
            metadata=qa_metadata,
            retrieved_memory_ids=[str(cand.memory_id) for cand in chosen],
            task_id=f"{sample_id}_qa_feedback",
        )

        row = {
            "sample_id": sample_id,
            "sample_index": sample_index,
            "question": question,
            "gold": gold,
            "prediction": prediction,
            "f1": _token_f1(prediction, gold),
            "exact_match": _exact_match(prediction, gold),
            "latency_seconds": latency,
            "retrieve_sec": retrieve_sec,
            "llm_sec": llm_sec,
            "usage": dict(resp.usage or {}),
            "retrieved_context": context,
            "trajectory_text": _corpus_to_context(docs),
            "memory_construction_sec": build_seconds,
            "trajectory": {
                "retrieval": {
                    "phase1_topk": int(args.memrl_phase1_topk),
                    "topk": int(args.memrl_topk),
                    "queries": retrieval_queries,
                    "selected_ids": [str(cand.memory_id) for cand in chosen],
                    "retrieved_units": retrieved_units,
                },
                "reward": reward,
                "store_size": store.count(),
            },
            "method": "memrl",
            "benchmark": "hotpotqa",
            "recorded_at": utcnow_iso(),
        }
        append_jsonl(rows_path, row)

        trajectory_payload = dict(row)
        trajectory_payload["store_records"] = [
            {
                "id": record.id,
                "content": record.content,
                "metadata": dict(record.metadata or {}),
            }
            for record in store.iter_chronological()
        ]
        append_jsonl(traj_path, trajectory_payload)

    rows = load_jsonl(rows_path)
    summary = {
        "method": "memrl",
        "benchmark": "hotpotqa",
        "dataset_path": str(dataset_path),
        "rows_file": str(rows_path),
        "trajectory_file": str(traj_path),
        "num_samples": len(rows),
        "avg_f1": (sum(float(r.get("f1", 0.0) or 0.0) for r in rows) / len(rows)) if rows else 0.0,
        "avg_exact_match": (
            sum(float(r.get("exact_match", 0.0) or 0.0) for r in rows) / len(rows)
        ) if rows else 0.0,
        "avg_latency_seconds": (
            sum(float(r.get("latency_seconds", 0.0) or 0.0) for r in rows) / len(rows)
        ) if rows else 0.0,
        "resource_usage": aggregate_rows_resource_usage(rows, model=args.memrl_model),
        "history_artifact": str(_history_artifact_path(args)) if _history_artifact_path(args) else None,
        "finished_at": utcnow_iso(),
    }
    write_json(summary_path, summary)
    return summary

from agentmem.eval.metrics import compute_metrics
from agentmem.methods.base import EfficiencyCounters
from agentmem.providers.base import Message
from agentmem.providers.openai_compat import OpenAICompatibleProvider

DIRECT_ANSWER_PREFIX = "###Answer:"
HOTPOTQA_METHOD_REGISTRY = {

    "longcontext": "agentmem.eval.hotpotqa_agentic.HotpotQALongContextMethod",
    "memrl":       "agentmem.eval.hotpotqa_corpus_adapters.HotpotQAMemRLMethod",
    "hipporag":    "agentmem.eval.hotpotqa_corpus_adapters.HotpotQAHippoRAGMethod",
    "simplemem":   "agentmem.eval.hotpotqa_corpus_adapters.HotpotQASimpleMemMethod",
    "lightmem":    "agentmem.eval.hotpotqa_corpus_adapters.HotpotQALightMemMethod",
    "plugmem":     "agentmem.eval.hotpotqa_mem_methods.HotpotQAPlugMemMethod",
    "ama_agent":   "agentmem.eval.hotpotqa_mem_methods.HotpotQAAmaAgentMethod",
    "memt":        "agentmem.eval.hotpotqa_mem_methods.HotpotQAMemTMethod",
    "autoharness": "agentmem.methods.autoharness.AutoHarnessMethod",
    "dci_lite":    "agentmem.methods.dci_lite.DCILiteMethod",
    "dci_lite_sum": "agentmem.methods.dci_lite.DCILiteSummarizeMethod",
    "automem":       "agentmem.methods.automem.AutoMemMethod",
    "automem_nograph": "agentmem.methods.automem.AutoMemNoGraphMethod",
    "automem_u1":    "agentmem.methods.automem.AutoMemU1Method",
    "automem_u2":    "agentmem.methods.automem.AutoMemU2Method",
    "automem_u3":    "agentmem.methods.automem.AutoMemU3Method",
    "automem_sum":   "agentmem.methods.automem.AutoMemSumMethod",
    "automem_graph": "agentmem.methods.automem.AutoMemGraphMethod",
    "automem_cost":  "agentmem.methods.automem_cost.AutoMemCostMethod",
    "automem_cost_spc": "agentmem.methods.automem_cost.AutoMemCostSPCOnly",
    "automem_cost_spc_gs": "agentmem.methods.automem_cost.AutoMemCostSPCGS",
    "automem_kvcache": "agentmem.methods.automem_cost.AutoMemCostKVFriendly",
}

def _read_hotpot_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _hotpot_trajectory_text(sample: dict[str, Any]) -> str:
    return _corpus_to_context(_build_docs(sample))

def _counter_dict(method: Any) -> dict[str, Any]:
    counters = getattr(method, "counters", None)
    if counters is None:
        return EfficiencyCounters().to_dict()
    if hasattr(counters, "to_dict"):
        return counters.to_dict()
    return dict(counters)

def _tokens_from_counters(counters: dict[str, Any], usage: dict[str, Any]) -> dict[str, int]:
    input_tokens = int(counters.get("query_input_tokens", 0) or 0)
    output_tokens = int(counters.get("query_output_tokens", 0) or 0)
    if usage and input_tokens == 0 and output_tokens == 0:
        input_tokens += int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
    return {
        "input": input_tokens,
        "output": output_tokens,
        "retrieved": int(counters.get("retrieved_context_tokens", 0) or 0),
        "build": int(counters.get("build_input_tokens", 0) or 0)
        + int(counters.get("build_output_tokens", 0) or 0),
    }

def _record_answer_usage(method: Any, usage: dict[str, Any]) -> None:
    counters = getattr(method, "counters", None)
    if counters is None or not hasattr(counters, "record_llm_call"):
        return
    prompt_tokens = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
    if prompt_tokens or completion_tokens:
        counters.record_llm_call(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            phase="query",
        )

def _supporting_titles(sample: dict[str, Any]) -> set[str]:
    titles: set[str] = set()
    for item in sample.get("supporting_facts", []) or []:
        if isinstance(item, (list, tuple)) and item:
            titles.add(str(item[0]))
    return titles

def _extract_semnode_ids(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"Sem\s*Node\s*(\d+)", str(text or ""), flags=re.I)]

class HotpotQAPlugMemMethod:
    """PlugMem HotpotQA adapter using the upstream corpus-level HPQA path."""

    name = "plugmem"

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        embedding_model: str | None = None,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        top_k: int = 10,
        save_dir: str | None = None,
        source_root: str | None = None,
        corpus_path: str | None = None,
        token_counter: Any = None,
        **_kw: Any,
    ) -> None:
        from agentmem.eval.amabench_runner.methods.plugmem import _plugmem_env_overrides
        from agentmem.eval.hotpotqa_corpus_adapters import HotpotQAAdapterError, _load_corpus_docs
        from agentmem.methods.base import EfficiencyCounters
        from agentmem.plugmem.upstream import (
            PlugMemGraphMemory,
            PlugMemSession,
            PlugMemUpstreamAdapter,
            default_plugmem_source_root,
            plugmem_env,
        )

        self._GraphMemory = PlugMemGraphMemory
        self._PlugMemSession = PlugMemSession
        self._plugmem_env = plugmem_env
        import os as _os
        self.top_k = max(1, int(_os.environ.get("HOTPOTQA_PLUGMEM_TOP_K", top_k)))
        self.save_dir = str(Path(save_dir or "outputs/plugmem_hotpotqa").expanduser())
        self._token_counter = token_counter
        self._counters = EfficiencyCounters()
        self._adapter_error_cls = HotpotQAAdapterError

        plugmem_embedding_base_url = embedding_base_url
        if plugmem_embedding_base_url:
            normalized = str(plugmem_embedding_base_url).rstrip("/")
            if not normalized.endswith("/embeddings"):
                normalized = f"{normalized}/embeddings"
            plugmem_embedding_base_url = normalized

        self.adapter = PlugMemUpstreamAdapter(
            source_root=source_root or default_plugmem_source_root(),
            env_overrides=_plugmem_env_overrides(
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                embedding_model=embedding_model,
                embedding_base_url=plugmem_embedding_base_url,
                embedding_api_key=embedding_api_key,
            ),
            retrieval_topk=self.top_k,
            memory_modes=("semantic_memory",),
        )
        self._multi_hop_retrieval_sem = self._load_upstream_multi_hop_retrieval()
        self._docs_raw = _load_corpus_docs(corpus_path)
        self._memory = self._build_or_load_corpus_graph()

    @property
    def counters(self) -> Any:
        return self._counters

    def reset_counters(self) -> Any:
        from agentmem.methods.base import EfficiencyCounters

        self._counters = EfficiencyCounters()
        return self._counters

    def _count_tokens(self, text: Any) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0
        return len(str(text or "").split())

    def _ensure_plugmem_dirs(self, sample_dir: Path) -> None:
        for subdir in ("episodic_memory", "semantic_memory", "procedural_memory", "tag", "subgoal", "logs"):
            (sample_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _has_graph_artifacts(self, sample_dir: Path) -> bool:
        sem_dir = sample_dir / "semantic_memory"
        return sem_dir.is_dir() and any(sem_dir.glob("*.json"))

    def _load_upstream_multi_hop_retrieval(self) -> Any:
        import importlib

        source_root = str(self.adapter.source_root)

        eval_root = os.path.join(source_root, "eval")
        hotpotqa_eval_root = os.path.join(eval_root, "hotpotqa")
        path_roots = (hotpotqa_eval_root, eval_root, source_root)
        for p in path_roots:
            if p not in sys.path:
                sys.path.insert(0, p)
        saved_utils = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "utils" or name.startswith("utils.")
        }
        hidden_paths: list[tuple[int, str]] = []
        for idx in range(len(sys.path) - 1, -1, -1):
            path = sys.path[idx]
            if path in path_roots:
                continue
            candidate = os.path.join(path, "utils") if path else "utils"
            if os.path.isdir(candidate):
                hidden_paths.append((idx, sys.path.pop(idx)))

        for p in path_roots:
            try:
                sys.path.remove(p)
            except ValueError:
                pass
            sys.path.insert(0, p)
        try:
            module = importlib.import_module("eval.hotpotqa.eval_qa_all")
            fn = getattr(module, "multi_hop_retrieval_sem")
        except Exception as exc:
            raise self._adapter_error_cls(
                "PlugMem HotpotQA retrieval is unavailable: cannot import "
                "vendor/plugmem/src/eval/hotpotqa/eval_qa_all.py::multi_hop_retrieval_sem"
            ) from exc
        finally:
            for idx, path in sorted(hidden_paths):
                sys.path.insert(idx, path)
            for name in list(sys.modules):
                if name == "utils" or name.startswith("utils."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_utils)
        if not callable(fn):
            raise self._adapter_error_cls(
                "PlugMem HotpotQA retrieval is unavailable: multi_hop_retrieval_sem is not callable."
            )
        return fn

    def _build_single_upstream_memory(self, item: dict[str, Any], emb_model: str | None) -> Any:
        structuring = sys.modules.get("memory_structuring.structuring_inference")
        if structuring is None:
            from memory_structuring import structuring_inference as structuring

        obs = f"Title: {item['title']}\nText: {item['text']}"
        memory = self.adapter._Memory(goal="Answer user's question", observation=obs, time="")
        memory.memory["episodic"] = [
            {
                "observation": obs,
                "action": "",
                "state": "",
                "reward": "",
                "subgoal": "",
            }
        ]
        semantic_memory = structuring.get_semantic(
            step={"observation": obs},
            trajectory_num=0,
            turn_num=0,
            time="",
        )
        memory.memory["semantic"] = semantic_memory
        for sm in semantic_memory:
            memory.memory_embedding["semantic"].append(
                {
                    "semantic_memory": self.adapter._embed_text(sm["semantic_memory"]),
                    "tags": [self.adapter._embed_text(tag) for tag in sm["tags"]],
                }
            )

        procedural_memory, goal, _return = structuring.get_procedural(trajectory=obs)
        memory.memory["procedural"].append(
            {
                "subgoal": goal,
                "procedural_memory": procedural_memory,
                "trajectory_num": 1,
                "time": memory.time,
                "return": _return,
            }
        )
        memory.memory_embedding["procedural"].append(
            {
                "procedural_memory": self.adapter._embed_text(procedural_memory),
                "subgoal": self.adapter._embed_text(goal),
            }
        )
        return memory

    def _corpus_fingerprint(self) -> str:
        """Stable fingerprint of the indexed corpus.

        Combines paragraph count + a hash over title+text. If the fingerprint
        on disk doesn't match the in-memory corpus, the cache is stale and we
        rebuild instead of silently reusing artifacts from a different run.
        """
        import hashlib
        h = hashlib.sha256()
        for d in self._docs_raw:
            h.update(str(d.get("title", "")).encode("utf-8"))
            h.update(b"\x1f")
            h.update(str(d.get("text", "")).encode("utf-8"))
            h.update(b"\x1e")
        return f"plugmem_hotpotqa_corpus_v1__n{len(self._docs_raw)}__{h.hexdigest()[:16]}"

    def _build_or_load_corpus_graph(self) -> dict[str, Any]:
        sample_dir = Path(self.save_dir).resolve()
        self._ensure_plugmem_dirs(sample_dir)
        session = self._PlugMemSession(
            session_id="hotpotqa-shared-corpus",
            goal="Answer user's question",
            steps=[],
            metadata={
                "benchmark": "hotpotqa",
                "task_type": "answer the question based on objective knowledge or information.",
            },
        )
        graph = self.adapter._new_memory_graph(log_file=sample_dir / "plugmem.log")

        fingerprint = self._corpus_fingerprint()
        fp_path = sample_dir / "corpus_fingerprint.txt"
        cache_valid = (
            self._has_graph_artifacts(sample_dir)
            and fp_path.exists()
            and fp_path.read_text(encoding="utf-8").strip() == fingerprint
        )
        t0 = time.perf_counter()
        with self._plugmem_env(self.adapter.env_overrides, sample_dir=sample_dir):
            if cache_valid:
                print(f"[plugmem build] reusing cached graph at {sample_dir}", flush=True)
                graph.build_mem_from_disk_hpqa_ver(str(sample_dir))
            else:
                embedding_model = self.adapter.env_overrides.get("EMBEDDING_MODEL_NAME")
                total = len(self._docs_raw)
                log_every = max(1, total // 50) if total > 100 else 1

                from concurrent.futures import ThreadPoolExecutor, as_completed
                workers = max(1, int(os.environ.get("PLUGMEM_BUILD_CONCURRENCY", "16")))
                print(
                    f"[plugmem build] start: {total} docs (parallel build, serial insert), "
                    f"workers={workers}",
                    flush=True,
                )

                built = 0
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(self._build_single_upstream_memory, item, embedding_model): idx
                        for idx, item in enumerate(self._docs_raw)
                    }
                    for fut in as_completed(futures):
                        memory = fut.result()
                        graph.insert_hpqa_ver(memory)                                  
                        built += 1
                        if built == 1 or built % log_every == 0 or built == total:
                            elapsed = time.perf_counter() - t0
                            rate = built / max(elapsed, 1e-9)
                            eta = (total - built) / max(rate, 1e-9)
                            print(
                                f"[plugmem build] doc {built}/{total} "
                                f"({100*built/total:.1f}%) elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s",
                                flush=True,
                            )

                fp_path.write_text(fingerprint, encoding="utf-8")
                print(f"[plugmem build] DONE in {time.perf_counter()-t0:.0f}s, fingerprint persisted", flush=True)
        self._build_seconds = time.perf_counter() - t0
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_llm_build", self._build_seconds)
        self._counters.build_input_tokens += self._count_tokens(
            "\n\n".join(f"{d['title']}\n{d['text']}" for d in self._docs_raw)
        )
        self._counters.total_input_tokens += self._counters.build_input_tokens
        sem_id2text = {node.semantic_id: node.get_semantic_memory() for node in graph.semantic_nodes}
        return {
            "graph_memory": self._GraphMemory(graph=graph, session=session, sample_dir=sample_dir),
            "sem_id2text": sem_id2text,
            "docs": self._docs_raw,
            "sample_id": "hotpotqa-shared-corpus",
        }

    def build_from_sample(self, sample: dict[str, Any], *, sample_id: str) -> Any:
        return self._memory

    def build(self, traj_text: str, *, task: str = "") -> Any:
        return self._memory

    def answer(self, memory: Any, question: str) -> str:
        graph_memory = memory.get("graph_memory") if isinstance(memory, dict) else self._memory["graph_memory"]
        sem_id2text = memory.get("sem_id2text", self._memory["sem_id2text"]) if isinstance(memory, dict) else self._memory["sem_id2text"]
        t_tool = time.perf_counter()
        with self._plugmem_env(self.adapter.env_overrides, sample_dir=graph_memory.sample_dir):
            result = self._multi_hop_retrieval_sem(
                memgraph=graph_memory.graph,
                sem_id2text=sem_id2text,
                question=question,
                task_type="answer the question based on objective knowledge or information.",
                judge_model=str(self.adapter.env_overrides.get("LLM_NAME") or ""),
                max_rounds=3,
                n_facts_new_query=3,
            )
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_tool", time.perf_counter() - t_tool)
        context = str(result.get("memory_str_all", "") or "")
        self._counters.record_retrieval(
            candidates_scored=len(sem_id2text),
            evidence_injected=len(_extract_semnode_ids(context)),
            context_tokens=self._count_tokens(context),
        )
        return context

class HotpotQAAmaAgentMethod:
    """AMA-Agent HotpotQA adapter over the shared corpus-level memory."""

    name = "ama_agent"

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        embedding_model: str | None = None,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        top_k: int = 8,
        neighbor_radius: int = 1,
        retrieval_mode: str = "embed",
        enable_tools: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        corpus_path: str | None = None,
        token_counter: Any = None,
        **kwargs: Any,
    ) -> None:
        from agentmem.eval.amabench_runner.methods.ama_agent import AMAAgentMethod
        from agentmem.eval.hotpotqa_corpus_adapters import _docs_as_strings, _load_corpus_docs
        from agentmem.methods.base import EfficiencyCounters

        self._counters = EfficiencyCounters()
        self.inner = AMAAgentMethod(
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            neighbor_radius=neighbor_radius,
            causal=False,
            enable_tools=enable_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.top_k = max(1, int(top_k))
        self.neighbor_radius = int(neighbor_radius)
        self.retrieval_mode = retrieval_mode
        self.enable_tools = enable_tools
        self._token_counter = token_counter
        self._docs_raw = _load_corpus_docs(corpus_path)

        turn_lines: list[str] = []
        for idx, doc in enumerate(self._docs_raw):
            title = str(doc.get("title", "")).strip()
            text = str(doc.get("text", "")).strip()
            if not title or not text:
                continue
            turn_lines.append(f"Turn {idx}:")
            turn_lines.append(f"  Action: read_document")

            flat_text = " ".join(text.split())
            turn_lines.append(f"  Observation: Title: {title} | Text: {flat_text}")
        traj_text = "\n".join(turn_lines)
        t0 = time.perf_counter()
        self._memory = self.inner.memory_construction(
            traj_text=traj_text,
            task="hotpotqa_corpus",
        )
        self._build_seconds = time.perf_counter() - t0
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_llm_build", self._build_seconds)
        self._counters.build_input_tokens += self._count_tokens(traj_text)
        self._counters.total_input_tokens += self._counters.build_input_tokens

    @property
    def counters(self) -> Any:
        return self._counters

    def reset_counters(self) -> Any:
        from agentmem.methods.base import EfficiencyCounters

        self._counters = EfficiencyCounters()
        return self._counters

    def _count_tokens(self, text: Any) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0
        return len(str(text or "").split())

    def build_from_sample(self, sample: dict[str, Any], *, sample_id: str) -> dict[str, Any]:
        return self._memory

    def build(self, traj_text: str, *, task: str = "") -> dict[str, Any]:
        return self._memory

    def answer(self, memory: dict[str, Any], question: str) -> str:
        t_tool = time.perf_counter()
        context = self.inner.memory_retrieve(memory or self._memory, question)
        if hasattr(self._counters, "add_seconds"):
            self._counters.add_seconds("w_tool", time.perf_counter() - t_tool)
        debug = getattr(self.inner, "_last_trajectory", None) or {}
        self._counters.record_retrieval(
            candidates_scored=len(((memory or self._memory).get("graph_mem") or {}).get("nodes", [])),
            evidence_injected=len(debug.get("expanded_turn_indices", []) or []),
            context_tokens=self._count_tokens(context),
        )
        return str(context)

class HotpotQAMemTMethod:
    """HotpotQA Mem-T adapter — HippoRAG-2 corpus regime.

    Builds a single Mem-T engine ONCE at __init__ over the shared 9,811-paragraph
    corpus (``hotpotqa_corpus.json``). All 200 questions are answered against
    that shared engine. This matches the regime used by plugmem / ama_agent /
    simplemem / hipporag, replacing the older per-sample build that ran on the
    row's 10 distractor paragraphs.
    """

    name = "memt"

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen3-32B",
        llm_base_url: str = "http://localhost:8000/v1",
        llm_api_key: str | None = None,
        db_type: str = "persistent",
        db_host: str = "localhost",
        db_port: int = 8070,
        db_path: str | None = None,
        embedding_model: str = "BAAI/bge-m3",
        k_turns: int = 4,
        update_retrieval_topk: int = 3,
        retrieval_topk: int = 5,
        max_tool_steps: int = 6,
        benchmark_name: str = "hotpotqa",
        corpus_path: str | None = None,
        token_counter: Any = None,
        **_kw: Any,
    ) -> None:
        from agentmem.eval.amabench_runner.methods.memt import _memt_env

        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.db_type = db_type
        self.db_host = db_host
        self.db_port = db_port
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.k_turns = k_turns
        self.update_retrieval_topk = update_retrieval_topk
        self.retrieval_topk = retrieval_topk
        self.max_tool_steps = max_tool_steps
        self.benchmark_name = benchmark_name
        self._token_counter = token_counter
        self._episode_counter = 0
        self._counters = EfficiencyCounters()
        self._memt_env = _memt_env

        from agentmem.eval.hotpotqa_corpus_adapters import _load_corpus_docs

        docs = _load_corpus_docs(corpus_path)
        shared_sample = self._corpus_to_memt_sample(docs, sample_id="hotpotqa_shared_corpus")
        engine = self._make_engine()
        n_sessions = len(shared_sample.get("conversation", []))
        print(
            f"[memt build] start: {len(docs)} docs across {n_sessions} sessions "
            f"(parallel session fan-out — set MEMT_BUILD_SHARDS / MEMT_BUILD_PARALLEL to tune)",
            flush=True,
        )
        _memt_t0 = time.perf_counter()
        with self._counters.time_block("build_wallclock_seconds"):
            with self._memt_env(api_key=self.llm_api_key, base_url=self.llm_base_url):

                self._parallel_build_from_sample(engine._builder, shared_sample)
        print(f"[memt build] DONE in {time.perf_counter()-_memt_t0:.0f}s", flush=True)
        self._counters.build_input_tokens += self._count_tokens(
            "\n\n".join(f"{d['title']}\n{d['text']}" for d in docs)
        )
        self._counters.total_input_tokens += self._counters.build_input_tokens
        self._shared_engine = engine
        self._shared_sample_id = "hotpotqa_shared_corpus"
        self._shared_memory = {
            "engine": engine,
            "sample_id": self._shared_sample_id,
            "memt_sample": shared_sample,
        }

    @property
    def counters(self) -> EfficiencyCounters:
        return self._counters

    def reset_counters(self) -> EfficiencyCounters:
        self._counters = EfficiencyCounters()
        return self._counters

    def _count_tokens(self, text: Any) -> int:
        if self._token_counter is not None:
            try:
                return int(self._token_counter(str(text or "")))
            except Exception:
                return 0
        return len(str(text or "").split())

    def _make_engine(self) -> Any:
        from agentmem.memt import MemTConfig, MemTRuntimeEngine

        config = MemTConfig(
            db_type=self.db_type,
            db_host=self.db_host,
            db_port=self.db_port,
            db_path=self.db_path,
            k_turns=self.k_turns,
            update_retrieval_topk=self.update_retrieval_topk,
            retrieval_topk=self.retrieval_topk,
            max_tool_steps=self.max_tool_steps,
            strong_model=self.llm_model,
            data_name=self.benchmark_name,
            temperature=0.0,
        )
        with self._memt_env(api_key=self.llm_api_key, base_url=self.llm_base_url):
            return MemTRuntimeEngine(config=config)

    def build_from_sample(self, sample: dict[str, Any], *, sample_id: str) -> Any:

        self._episode_counter += 1
        return self._shared_memory

    def build(self, traj_text: str, *, task: str = "") -> Any:
        return self._shared_memory

    @staticmethod
    def _corpus_to_memt_sample(docs: list[dict[str, Any]], *, sample_id: str) -> dict[str, Any]:
        """Shape the 9,811-paragraph corpus as a Mem-T sample with N sessions.

        Each paragraph becomes a turn under one of N synthetic sessions.
        Mem-T's `MemoryBuilder._process_session` carries cross-batch state
        (``prev_summary``, ``prev_personas_map``) within a session, so each
        session is its own sequential thread of state. By chunking the corpus
        into N sessions, we expose N independent threads that
        `_parallel_build_from_sample` can fan out across worker threads (vLLM
        continuous-batches the concurrent LLM calls). Sessions all write to
        the same sample-id-prefixed vector-DB collections, so retrieval still
        sees one logical memory.

        Number of sessions is from MEMT_BUILD_SHARDS env var (default 16).
        """
        n_shards = max(1, int(os.environ.get("MEMT_BUILD_SHARDS", "16")))
        if not docs:
            return {
                "sample_id": sample_id,
                "question": "",
                "answer": "",
                "supporting_facts": [],
                "conversation": [],
            }

        shard_turns: list[list[dict[str, Any]]] = [[] for _ in range(n_shards)]
        for idx, doc in enumerate(docs):
            title = str(doc.get("title", ""))
            body = str(doc.get("text", ""))
            shard_turns[idx % n_shards].append(
                {
                    "turn_id": f"doc_{idx}",
                    "speaker": "document",
                    "text": f"[{title}] {body}",
                    "metadata": {"title": title, "doc_index": idx},
                }
            )
        sessions: list[dict[str, Any]] = []
        for shard_idx, turns in enumerate(shard_turns):
            if not turns:
                continue
            sessions.append(
                {
                    "session_id": f"hotpotqa_shard_{shard_idx:03d}",
                    "session_turns": turns,
                    "metadata": {
                        "sample_id": sample_id,
                        "speaker_a": "document",
                        "speaker_b": "question",
                        "session_time": "",
                        "task": "hotpotqa_corpus",
                    },
                }
            )
        return {
            "sample_id": sample_id,
            "question": "",
            "answer": "",
            "supporting_facts": [],
            "conversation": sessions,
        }

    @staticmethod
    def _parallel_build_from_sample(builder: Any, sample: dict[str, Any]) -> None:
        """Replacement for ``MemoryBuilder.build_from_sample`` that processes
        sessions in parallel via ThreadPoolExecutor.

        Mem-T's vendored builder iterates ``for session in sessions`` serially
        (memory_builder.py line 70). Each session is independent (its own
        cross-batch state) — but all sessions share the SAME sample-id-prefixed
        chromadb collections, and chromadb's SQLite backend serializes writes.
        Concurrent `add()` from multiple threads triggers ``code: 1032 attempt
        to write a readonly database`` (lock contention). Fix: wrap the
        vector-DB write/read methods in a global lock so the LLM calls (the
        slow part) run in parallel while the DB ops serialize.

        Concurrency: MEMT_BUILD_PARALLEL env var (default = number of sessions
        in the sample, capped at 32).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time as _time
        import threading

        sample_id = sample.get("conversation", [{}])[0].get("metadata", {}).get("sample_id", "") or "unknown_sample"
        builder._init_sample_collections(sample_id)
        sessions = sample.get("conversation", [])
        if not sessions:
            return

        vdb = builder.vector_db
        if not getattr(vdb, "_memt_parallel_locked", False):
            _vdb_lock = threading.Lock()
            _max_retries = int(os.environ.get("MEMT_VDB_RETRIES", "8"))
            for _method in ("add", "get", "search", "query", "delete", "delete_collection",
                            "create_collection", "update", "upsert"):
                if hasattr(vdb, _method):
                    _orig = getattr(vdb, _method)
                    def _wrap(orig, lock=_vdb_lock, retries=_max_retries):
                        import time as _wtime
                        def locked(*a, **kw):
                            for attempt in range(retries):
                                with lock:
                                    try:
                                        return orig(*a, **kw)
                                    except Exception as exc:
                                        msg = str(exc).lower()
                                        if "readonly database" in msg or "database is locked" in msg or "1032" in msg:
                                            if attempt + 1 == retries:
                                                raise

                                            pass
                                        else:
                                            raise

                                _wtime.sleep(0.05 * (2 ** attempt))
                            raise RuntimeError(f"vdb retry exhausted on {orig.__name__}")
                        return locked
                    try:
                        setattr(vdb, _method, _wrap(_orig))
                    except (AttributeError, TypeError):
                        pass                           
            try:
                vdb._memt_parallel_locked = True
            except (AttributeError, TypeError):
                pass

        total_batches = sum(
            (len(s.get("session_turns", [])) + builder.k_turns - 1) // builder.k_turns
            for s in sessions if s.get("session_turns")
        )
        workers = min(
            int(os.environ.get("MEMT_BUILD_PARALLEL", str(min(32, len(sessions))))),
            len(sessions),
        )
        print(
            f"[memt build] PARALLEL session fan-out: {len(sessions)} sessions, "
            f"{total_batches} batches total, workers={workers}",
            flush=True,
        )
        t0 = _time.perf_counter()

        completed_batches_lock = __import__("threading").Lock()
        completed_batches = [0]

        def _process_one(session: dict[str, Any]) -> str:
            sess_id = session.get("session_id", "?")
            sess_t0 = _time.perf_counter()
            n_turns = len(session.get("session_turns", []))
            n_batches = (n_turns + builder.k_turns - 1) // builder.k_turns
            try:

                builder._process_session(session, sample_id, None)
            except Exception as exc:
                return f"  [memt build] session {sess_id} FAILED: {type(exc).__name__}: {exc}"
            with completed_batches_lock:
                completed_batches[0] += n_batches
                done = completed_batches[0]
            elapsed = _time.perf_counter() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (total_batches - done) / max(rate, 1e-9)
            return (
                f"  [memt build] session {sess_id} done "
                f"({n_batches} batches in {_time.perf_counter()-sess_t0:.0f}s); "
                f"agg {done}/{total_batches} ({100*done/total_batches:.1f}%) "
                f"elapsed={elapsed:.0f}s rate={rate:.2f}b/s eta={eta:.0f}s"
            )

        if os.environ.get("MEMT_LARGE_CORPUS", "0") in ("1", "true", "True"):
            HotpotQAMemTMethod._large_corpus_process(builder, sessions, sample_id, workers, total_batches, t0, completed_batches, completed_batches_lock)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_process_one, s) for s in sessions]
                for fut in as_completed(futures):
                    print(fut.result(), flush=True)

        print(
            f"[memt build] PARALLEL session fan-out DONE in {_time.perf_counter()-t0:.0f}s",
            flush=True,
        )

    @staticmethod
    def _large_corpus_process(builder, sessions, sample_id, workers, total_batches, t0, completed_batches, completed_batches_lock):
        """Large-corpus build: facts-only, no per-batch state growth.

        Per batch: 1 LLM call to extract facts with empty prev_summary +
        empty personas. Direct-insert into c_facts (no search+merge update).
        Skips persona/summary upserts entirely. Quadratic factor removed.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time as _time
        import uuid as _uuid

        c_facts = f"{sample_id}_{builder.BASE_C_FACTS}"
        c_turns = f"{sample_id}_{builder.BASE_C_TURNS}"
        formation = builder.formation
        vdb = builder.vector_db

        lc_workers = int(os.environ.get("MEMT_LARGE_CORPUS_PARALLEL", "64"))
        print(
            f"[memt build LARGE_CORPUS] facts-only mode, workers={lc_workers}, "
            f"sessions={len(sessions)}, total_batches={total_batches}",
            flush=True,
        )

        def _process_batch(session_id, batch_turns, batch_idx):
            current_text = ""
            current_turns_list = []
            batch_turn_ids = []
            for turn in batch_turns:
                turn_text = turn.get("text", "")
                turn_speaker = turn.get("speaker", "Unknown")
                formatted = f"{turn_speaker}: {turn_text}"
                current_text += formatted + "\n"
                current_turns_list.append(formatted)
                batch_turn_ids.append(turn.get("turn_id", ""))
            try:

                messages = formation.construct_prompt(current_text, "", "")
                response = formation.llm_executor.get_completion(messages)
                tool_calls = builder._parse_tool_calls(response)
                facts_added = 0
                for tc in tool_calls:
                    if tc.get("name") not in ("create_fact", "create_experience"):
                        continue
                    args = tc.get("arguments", {}) or {}
                    fact_text = args.get("fact") or args.get("experience") or ""
                    if not fact_text:
                        continue
                    fact_id = f"{session_id}_b{batch_idx}_{_uuid.uuid4().hex[:8]}"
                    vdb.add(
                        c_facts,
                        ids=[fact_id],
                        documents=[fact_text],
                        metadatas=[{
                            "id": fact_id,
                            "col_name": c_facts,
                            "session_id": session_id,
                            "source_turn_ids": batch_turn_ids,
                            "type": "fact" if tc["name"] == "create_fact" else "experience",
                        }],
                    )
                    facts_added += 1

                batch_id = f"{session_id}_batch_{batch_idx}"
                vdb.add(
                    c_turns,
                    ids=[batch_id],
                    documents=[current_text],
                    metadatas=[{"id": batch_id, "col_name": c_turns, "session_id": session_id}],
                )
                return facts_added, None
            except Exception as exc:
                return 0, f"{type(exc).__name__}: {exc}"

        builder._init_sample_collections(sample_id)
        jobs = []
        for session in sessions:
            sess_id = session.get("session_id", "?")
            turns = session.get("session_turns", [])
            for i in range(0, len(turns), builder.k_turns):
                batch = turns[i:i+builder.k_turns]
                jobs.append((sess_id, batch, i // builder.k_turns))
        total_jobs = len(jobs)
        print(f"[memt build LARGE_CORPUS] {total_jobs} batches queued", flush=True)

        done = [0]
        facts = [0]
        errors = [0]
        last_log = [_time.perf_counter()]
        log_lock = __import__("threading").Lock()

        with ThreadPoolExecutor(max_workers=lc_workers) as executor:
            futures = [executor.submit(_process_batch, sid, b, idx) for (sid, b, idx) in jobs]
            for fut in as_completed(futures):
                fa, err = fut.result()
                with log_lock:
                    done[0] += 1
                    facts[0] += fa
                    if err:
                        errors[0] += 1
                    now = _time.perf_counter()
                    if now - last_log[0] >= 30 or done[0] == total_jobs:
                        elapsed = now - t0
                        rate = done[0] / max(elapsed, 1e-9)
                        eta = (total_jobs - done[0]) / max(rate, 1e-9)
                        print(
                            f"[memt build LARGE_CORPUS] {done[0]}/{total_jobs} batches "
                            f"({100*done[0]/total_jobs:.1f}%) facts={facts[0]} errors={errors[0]} "
                            f"elapsed={elapsed:.0f}s rate={rate:.2f}b/s eta={eta:.0f}s",
                            flush=True,
                        )
                        last_log[0] = now

    def answer(self, memory: Any, question: str) -> str:
        with self._counters.time_block("wallclock_seconds"):
            with self._memt_env(api_key=self.llm_api_key, base_url=self.llm_base_url):
                result = memory["engine"].retrieve_and_answer(
                    question,
                    sample_id=memory["sample_id"],
                    category="",
                )
        if isinstance(result, dict):
            answer = result.get("answer", "")
            retrieved = result.get("retrieved_context") or result.get("context") or ""
        else:
            answer = getattr(result, "answer", "")
            retrieved = getattr(result, "retrieved_context", "") or getattr(result, "context", "")
        self._counters.record_retrieval(
            candidates_scored=0,
            evidence_injected=1 if retrieved else 0,
            context_tokens=self._count_tokens(retrieved),
        )
        return f"{DIRECT_ANSWER_PREFIX} {answer}"

    def _sample_to_memt(self, sample: dict[str, Any], *, sample_id: str) -> dict[str, Any]:
        supporting = {
            (str(item[0]), int(item[1]))
            for item in sample.get("supporting_facts", []) or []
            if isinstance(item, (list, tuple)) and len(item) >= 2
        }
        turns: list[dict[str, Any]] = []
        for doc_idx, (title, body) in enumerate(_build_docs(sample)):
            turns.append(
                {
                    "turn_id": f"doc_{doc_idx}",
                    "speaker": "document",
                    "text": f"[{title}] {body}",
                    "metadata": {
                        "title": title,
                        "doc_index": doc_idx,
                        "has_supporting_fact": any(sf_title == title for sf_title, _ in supporting),
                    },
                }
            )
        return {
            "sample_id": sample_id,
            "question": str(sample.get("question", "")),
            "answer": str(sample.get("answer", "")),
            "supporting_facts": sample.get("supporting_facts", []),
            "conversation": [
                {
                    "session_id": "hotpotqa_documents",
                    "session_turns": turns,
                    "metadata": {
                        "sample_id": sample_id,
                        "speaker_a": "document",
                        "speaker_b": "question",
                        "session_time": "",
                        "task": "hotpotqa",
                    },
                }
            ],
        }

def get_hotpotqa_method(name: str, **kwargs: Any) -> Any:
    key = name.strip().lower().replace("-", "_")
    if key == "ama-agent":
        key = "ama_agent"
    if key not in HOTPOTQA_METHOD_REGISTRY:
        raise ValueError(f"Unknown HotpotQA method {name!r}. Available: {sorted(HOTPOTQA_METHOD_REGISTRY)}")
    if key == "plugmem":
        return HotpotQAPlugMemMethod(**kwargs)
    if key == "ama_agent":
        return HotpotQAAmaAgentMethod(**kwargs)
    if key == "memt":
        return HotpotQAMemTMethod(**kwargs)
    if key == "longcontext":
        from agentmem.eval.hotpotqa_agentic import HotpotQALongContextMethod
        return HotpotQALongContextMethod(**kwargs)
    if key == "memrl":
        from agentmem.eval.hotpotqa_corpus_adapters import HotpotQAMemRLMethod
        return HotpotQAMemRLMethod(**kwargs)
    if key == "hipporag":
        from agentmem.eval.hotpotqa_corpus_adapters import HotpotQAHippoRAGMethod
        return HotpotQAHippoRAGMethod(**kwargs)
    if key == "simplemem":
        from agentmem.eval.hotpotqa_corpus_adapters import HotpotQASimpleMemMethod
        return HotpotQASimpleMemMethod(**kwargs)
    if key == "lightmem":
        from agentmem.eval.hotpotqa_corpus_adapters import HotpotQALightMemMethod
        return HotpotQALightMemMethod(**kwargs)
    from agentmem.methods import build_method

    return build_method(key, **kwargs)

def run_hotpotqa50_method(
    *,
    method_name: str,
    samples: list[dict[str, Any]],
    output_dir: str | Path,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str = "EMPTY",
    embedding_model: str = "Qwen3-Embedding-4B",
    embedding_base_url: str | None = None,
    embedding_api_key: str = "EMPTY",
    config_path: str | None = None,
    max_response_tokens: int = 512,
) -> dict[str, Any]:
    """Run one HotpotQA method and write answers_<method>_<backbone>.jsonl."""
    method_key = method_name.strip().lower().replace("-", "_")
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    backbone_tag = re.sub(r"[^a-z0-9]+", "-", str(llm_model or "unknown").lower()).strip("-") or "unknown"
    outfile = outdir / f"answers_{method_key}_{backbone_tag}.jsonl"
    done = {str(row.get("id") or row.get("sample_id")) for row in load_jsonl(outfile)} if outfile.exists() else set()

    if not embedding_base_url:

        import warnings
        warnings.warn(
            "embedding_base_url not set; falling back to llm_base_url. For split "
            "vLLM serving, pass --embedding-base-url http://localhost:30001/v1 "
            "(or set EMBEDDING_BASE_URL).",
            RuntimeWarning,
            stacklevel=2,
        )
    kwargs: dict[str, Any] = {
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "llm_api_key": llm_api_key,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url or llm_base_url,
        "embedding_api_key": embedding_api_key,
    }
    if config_path:
        kwargs["config_path"] = config_path
    method = get_hotpotqa_method(method_key, **kwargs)
    answer_provider = OpenAICompatibleProvider(
        api_key=llm_api_key,
        model=llm_model,
        base_url=llm_base_url,
        max_retries=5,
    )

    build_snapshot = _counter_dict(method)

    rows_written = 0
    mode = "a" if done else "w"
    with outfile.open(mode, encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            sample_id = str(sample.get("_id") or sample.get("id") or index)
            if sample_id in done:
                continue
            if hasattr(method, "reset_counters"):
                method.reset_counters()
            question = str(sample.get("question", "")).strip()
            gold = str(sample.get("answer", "")).strip()
            traj_text = _hotpot_trajectory_text(sample)
            started = time.perf_counter()
            usage: dict[str, Any] = {}
            retrieved_context = ""
            error = None
            try:
                if hasattr(method, "build_from_sample"):
                    memory = method.build_from_sample(sample, sample_id=sample_id)
                elif hasattr(method, "build_from_document_text"):
                    memory = method.build_from_document_text(
                        traj_text,
                        task="hotpotqa",
                        domain="hotpotqa",
                        sub_domain=sample_id,
                    )
                else:
                    memory = method.build(traj_text, task="hotpotqa")
                retrieved_context = str(method.answer(memory, question) or "")
                direct_prediction = ""
                if retrieved_context.lstrip().startswith(DIRECT_ANSWER_PREFIX):
                    direct_prediction = _normalize_prediction(retrieved_context)
                if direct_prediction and direct_prediction != DIRECT_ANSWER_PREFIX:
                    prediction = direct_prediction
                else:
                    prompt = _hotpot_prompt(question=question, context=retrieved_context)
                    response = answer_provider.chat(
                        [Message(role="user", content=prompt)],
                        temperature=0.0,
                        max_tokens=max_response_tokens,
                    )
                    usage = dict(response.usage or {})
                    _record_answer_usage(method, usage)
                    prediction = _normalize_prediction(response)
            except Exception as exc:
                prediction = ""
                error = str(exc)
            wall_time = time.perf_counter() - started
            counters = _counter_dict(method)
            metrics = compute_metrics(prediction, gold)
            row = {
                "id": sample_id,
                "sample_id": sample_id,
                "sample_index": index,
                "method": method_key,
                "benchmark": "hotpotqa50",
                "question": question,
                "gold": gold,
                "prediction": prediction,
                "em": metrics.get("exact_match", 0.0),
                "f1": metrics.get("token_f1", 0.0),
                "exact_match": metrics.get("exact_match", 0.0),
                "token_f1": metrics.get("token_f1", 0.0),
                "tokens": _tokens_from_counters(counters, usage),
                "wall_time": wall_time,
                "latency_seconds": wall_time,
                "usage": usage,
                "counters": counters,
                "build_counters": build_snapshot,
                "retrieved_context": retrieved_context,
                "supporting_facts": sample.get("supporting_facts", []),

                "context": sample.get("context", []),
                "question_type": sample.get("type"),
                "error": error,
                "recorded_at": utcnow_iso(),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows_written += 1

    rows = load_jsonl(outfile)
    summary = {
        "method": method_key,
        "benchmark": "hotpotqa50",
        "answers_file": str(outfile),
        "answers_path": str(outfile),                                            
        "build_counters": build_snapshot,
        "num_samples": len(rows),
        "rows_written": rows_written,
        "avg_em": (sum(float(r.get("em", r.get("exact_match", 0.0)) or 0.0) for r in rows) / len(rows)) if rows else 0.0,
        "avg_f1": (sum(float(r.get("f1", r.get("token_f1", 0.0)) or 0.0) for r in rows) / len(rows)) if rows else 0.0,
    }
    write_json(outdir / "summary.json", summary)
    return summary
