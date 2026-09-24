#!/usr/bin/env python3
"""Unified benchmark runner for the cross-scenario memory evaluation.

Runs any combination of {method} × {benchmark} and writes results locally.
This is the main entry point for reproducing the paper's benchmark cells.

Benchmarks:
    hotpotqa     — Multi-hop question answering (datasets/hotpotqa/ or HF)
    locomo       — Multi-session conversational QA (HF: locomo-bench/LoCoMo-Plus)
    alfworld     — Embodied household agent tasks (TextWorld)
    amabench     — Long-horizon agent memory, 6 domains
    memoryagentbench — Incremental multi-turn memory benchmark (HF: ai-hyz/MemoryAgentBench)

Methods:
    longcontext  — Full-context baseline (no retrieval)
    simplemem    — Density gating + atomic units + query-aware retrieval
    lightmem     — Sleep-time consolidation + topic segmentation
    memrl        — RL-trained memory read/write policy
    mem0         — Mem0 semantic/episodic memory store
    memoryos     — Hierarchical OS-style personal memory manager
    amem         — Agentic memory notes with dynamic memory evolution
    memt         — Typed hierarchical DB + learned retrieval
    hipporag     — KG-enhanced indexing + PPR diffusion
    plugmem      — Proposition + prescriptive KG
    ama_agent    — Causality graph + tool-augmented retrieval

Usage:
    # Single method on single benchmark
    python examples/run_benchmark.py --benchmark locomo --method simplemem

    # Multiple methods
    python examples/run_benchmark.py --benchmark amabench --method simplemem hipporag plugmem

    # Full sweep (all methods × all benchmarks)
    python examples/run_benchmark.py --sweep

    # With efficiency tracking
    python examples/run_benchmark.py --benchmark amabench --method plugmem --track-efficiency

    # Dry run (show what would be executed)
    python examples/run_benchmark.py --sweep --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from agentmem.methods.registry import AVAILABLE_METHODS

ALL_BENCHMARKS = ("hotpotqa", "locomo", "alfworld", "amabench", "memoryarena", "memoryagentbench")

ALL_METHODS = AVAILABLE_METHODS

_ANSWER_FN_METHODS = {
    "autoharness",
    "dci_lite",
    "dci_lite_sum",
    "dci_memory",
    "automem",
    "automem_u1",
    "automem_u2",
    "automem_u3",
    "automem_sum",
    "automem_nograph",
    "automem_graph",
    "automem_cost",
    "automem_cost_spc",
    "automem_cost_spc_gs",
    "automem_kvcache",
    "automem_kv",
    "automem_graph_kv",
    "automem_single",
    "automem_oneretry",
    "automem_graph_single",
    "automem_graph_oneretry",
    "automem_graph_template",
    "longcontext",
    "simplemem",
    "lightmem",
    "hipporag",
    "plugmem",
    "ama_agent",
    "memt",
    "memrl",
    "mem0",
    "memoryos",
    "amem",
}

DIRECT_ANSWER_PREFIX = "###Answer:"

def _split_direct_answer(context: Any) -> str | None:
    text = str(context or "").lstrip()
    if not text.startswith(DIRECT_ANSWER_PREFIX):
        return None
    return text.split(DIRECT_ANSWER_PREFIX, 1)[1].strip()

_COMPATIBILITY: dict[str, set[str]] = {
    "hotpotqa":     set(ALL_METHODS),
    "locomo":       set(ALL_METHODS),
    "alfworld":     {"longcontext", "simplemem", "lightmem", "memrl", "memt", "ama_agent", "dci_lite", "dci_lite_sum", "dci_memory", "automem", "hipporag", "plugmem"},
    "amabench":     set(ALL_METHODS),
    "memoryagentbench": set(ALL_METHODS),
    "memoryarena":  {
        "longcontext",
        "simplemem",
        "lightmem",
        "hipporag",
        "plugmem",
        "ama_agent",
        "memt",
        "memrl",
        "mem0",
        "memoryos",
        "amem",
        "autoharness",
        "dci_lite",
        "dci_lite_sum",
        "automem",
    },
}

def _run_hotpotqa(method_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run a method on HotpotQA under the unified HippoRAG-2 corpus regime.

    All methods see the same shared 9K-paragraph corpus
    (``hotpotqa_corpus.json``). Per-question distractor packaging is no longer
    used; each method's HotpotQA adapter loads the corpus once at init and
    answers each question against that shared memory. Routing happens through
    ``agentmem.eval.hotpotqa_mem_methods.run_hotpotqa50_method`` which dispatches
    via ``HOTPOTQA_METHOD_REGISTRY``.
    """
    from agentmem.eval.hotpotqa_mem_methods import run_hotpotqa50_method

    samples = _load_hotpotqa_dataset(args)
    cap = args.max_episodes or getattr(args, "max_questions", None)
    if cap:
        samples = samples[:cap]

    out_dir = _output_dir(args, "hotpotqa", method_name)
    summary = run_hotpotqa50_method(
        method_name=method_name,
        samples=samples,
        output_dir=str(out_dir),
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        llm_api_key=getattr(args, "llm_api_key", None) or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        embedding_model=getattr(args, "embedding_model", None) or "Qwen3-Embedding-4B",
        embedding_base_url=getattr(args, "embedding_base_url", None)
        or os.environ.get("EMBEDDING_BASE_URL")
        or args.llm_base_url,
        embedding_api_key=getattr(args, "embedding_api_key", None)
        or os.environ.get("EMBEDDING_API_KEY")
        or getattr(args, "llm_api_key", None)
        or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        config_path=args.config_path,
    )

    rows: list[dict[str, Any]] = []
    answers_path = summary.get("answers_path")
    if answers_path and Path(answers_path).exists():
        with open(answers_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return {"benchmark": "hotpotqa", "method": method_name, "results": rows, "summary": summary}

def _run_locomo(method_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run a method on LoCoMo benchmark."""
    from agentmem.methods import build_method

    method = build_method(method_name, **_method_kwargs(args))
    answer_fn = _make_answer_fn(args) if method_name in _ANSWER_FN_METHODS else None
    progress_file = _output_dir(args, "locomo", method_name) / f"answers_{method_name}_{_backbone_tag(args)}.jsonl"
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if progress_file.exists():
        with progress_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (str(row.get("sample_id", row.get("episode_id", ""))), str(row.get("question", "")))
                if key in seen:
                    continue
                seen.add(key)
                results.append(row)

    dataset = _load_locomo_dataset(args)
    pool_file = getattr(args, "locomo_pool_file", None)
    if pool_file:
        pool_rows = _load_jsonl(pool_file)
        if getattr(args, "max_questions", None):
            pool_rows = pool_rows[: args.max_questions]
        episodes_by_id = {
            str(ep.get("episode_id", ep.get("sample_id", i))): ep
            for i, ep in enumerate(dataset)
        }
        missing = [row for row in pool_rows if str(row.get("sample_id", "")) not in episodes_by_id]
        if missing:
            preview = ", ".join(str(row.get("sample_id")) for row in missing[:5])
            raise ValueError(
                f"LoCoMo pool has {len(missing)} rows whose sample_id is absent from "
                f"--test-file episodes. First missing sample_id values: {preview}"
            )
        memory_by_sample: dict[str, Any] = {}
        for row in pool_rows:
            sample_id = str(row.get("sample_id", ""))
            question = str(row.get("question", ""))
            key = (sample_id, question)
            if key in seen:
                continue
            episode = episodes_by_id[sample_id]
            if sample_id not in memory_by_sample:
                traj_text = episode.get("trajectory") or _locomo_episode_to_text(episode)
                memory_by_sample[sample_id] = method.build(traj_text, task=episode.get("task", ""))
            context = method.answer(memory_by_sample[sample_id], question)
            direct_answer = _split_direct_answer(context)
            if direct_answer is not None:
                answer = direct_answer
            elif answer_fn is not None:
                from agentmem.eval.locomo_runner.prompts import (
                    _build_locomo_answer_prompt,
                    _coerce_int_category,
                )

                prompt = _build_locomo_answer_prompt(
                    question=question,
                    context=context,
                    category=_coerce_int_category(row.get("category")),
                )
                answer = answer_fn(prompt, max_tokens=256)
            else:
                answer = context
            result_row = {
                "episode_id": sample_id,
                "sample_id": sample_id,
                "method": method_name,
                "question": question,
                "prediction": answer,
                "gold": row.get("gold", row.get("answer", "")),
                "category": row.get("category", ""),
                "category_name": row.get("category_name", ""),
                "retrieved_context": context if answer_fn is not None and direct_answer is None else "",
            }
            with progress_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                handle.flush()
            seen.add(key)
            results.append(result_row)

        payload = {"benchmark": "locomo", "method": method_name, "results": results}
        if getattr(args, "track_efficiency", False) and hasattr(method, "_counters"):
            try:
                payload["counters"] = method._counters.to_dict()
                payload["fs_counters"] = dict(getattr(method, "_fs_counters", {}))
            except Exception:
                pass
        return payload

    for i, episode in enumerate(dataset):
        if args.max_episodes and i >= args.max_episodes:
            break
        traj_text = episode.get("trajectory") or _locomo_episode_to_text(episode)
        memory = method.build(traj_text, task=episode.get("task", ""))

        qas = episode.get("questions") or episode.get("qa") or []
        if getattr(args, "max_questions", None):
            qas = qas[: args.max_questions]
        for qa in qas:
            episode_id = str(episode.get("episode_id", episode.get("sample_id", str(i))))
            question = str(qa["question"])
            key = (episode_id, question)
            if key in seen:
                continue
            context = method.answer(memory, question)
            direct_answer = _split_direct_answer(context)
            if direct_answer is not None:
                answer = direct_answer
            elif answer_fn is not None:
                from agentmem.eval.locomo_runner.prompts import (
                    _build_locomo_answer_prompt,
                    _coerce_int_category,
                )

                prompt = _build_locomo_answer_prompt(
                    question=question,
                    context=context,
                    category=_coerce_int_category(qa.get("category")),
                )
                answer = answer_fn(prompt, max_tokens=256)
            else:
                answer = context
            result_row = {
                "episode_id": episode_id,
                "method": method_name,
                "question": question,
                "prediction": answer,
                "gold": qa.get("answer", ""),
                "category": qa.get("category", ""),
                "retrieved_context": context if answer_fn is not None and direct_answer is None else "",
            }
            with progress_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                handle.flush()
            seen.add(key)
            results.append(result_row)

        if hasattr(method, "reset_counters") and not str(getattr(method, "name", "")).startswith("automem"):
            method.reset_counters()

    payload = {"benchmark": "locomo", "method": method_name, "results": results}
    if getattr(args, "track_efficiency", False) and hasattr(method, "_counters"):
        try:
            payload["counters"] = method._counters.to_dict()
            payload["fs_counters"] = dict(getattr(method, "_fs_counters", {}))
        except Exception:
            pass
    return payload

def _run_memoryagentbench(method_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run MemoryAgentBench through the unified memory-method interface."""
    from agentmem.eval.memoryagentbench_runner import load_memoryagentbench
    from agentmem.eval.metrics import compute_metrics
    from agentmem.methods import build_method

    method_kwargs = _method_kwargs(args)
    if method_name == "hipporag":
        method_kwargs.setdefault("retrieval_top_k", int(os.environ.get("MAB_HIPPORAG_RETRIEVAL_TOP_K", "32")))
        method_kwargs.setdefault("top_k", int(os.environ.get("MAB_HIPPORAG_TOP_K", "5")))
    elif method_name == "lightmem":
        method_kwargs.setdefault("retrieve_limit", int(os.environ.get("MAB_LIGHTMEM_RETRIEVE_LIMIT", "5")))
    method = build_method(method_name, **method_kwargs)
    out_dir = _output_dir(args, "memoryagentbench", method_name)
    answers_path = out_dir / f"answers_{method_name}_{_backbone_tag(args)}.jsonl"
    summary_path = out_dir / "summary.json"
    categories = [str(c).upper() for c in getattr(args, "mab_categories", ["AR", "TTL", "LRU", "CR"])]
    run_config_path = out_dir / "run_config.json"

    dataset = load_memoryagentbench(categories=categories, test_file=args.test_file)
    if args.max_episodes:
        dataset = dataset[: args.max_episodes]
    target_by_category = _memoryagentbench_category_targets(dataset, categories, getattr(args, "max_questions", None))
    desired_categories = set(target_by_category)

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    included_existing: list[dict[str, Any]] = []
    extra_existing: list[dict[str, Any]] = []
    completed_by_category: dict[str, int] = {cat: 0 for cat in categories}
    if answers_path.exists():
        with answers_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (str(row.get("haystack_id", row.get("episode_id", ""))), str(row.get("qa_pair_id", row.get("question", ""))))
                if key in seen:
                    extra_existing.append(row)
                    continue
                seen.add(key)
                cat = str(row.get("category") or "unknown").upper()
                if cat in desired_categories and completed_by_category.get(cat, 0) < target_by_category.get(cat, 0):
                    included_existing.append(row)
                    completed_by_category[cat] = completed_by_category.get(cat, 0) + 1
                else:
                    extra_existing.append(row)
    if extra_existing:
        backup_path = answers_path.with_suffix(answers_path.suffix + f".pre_rebalance_{int(time.time())}.bak")
        answers_path.replace(backup_path)
        with answers_path.open("w", encoding="utf-8") as handle:
            for row in included_existing:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    results.extend(included_existing)

    run_config = {
        "benchmark": "memoryagentbench",
        "method": method_name,
        "backbone": _backbone_tag(args),
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "test_file": args.test_file,
        "mab_manifest": getattr(args, "mab_manifest", None),
        "categories": categories,
        "target_by_category": target_by_category,
        "answers_rebalance_backup": str(backup_path) if extra_existing else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stop = False
    for haystack_index, haystack in enumerate(dataset):
        category = str(haystack.get("category") or "unknown").upper()
        if completed_by_category.get(category, 0) >= target_by_category.get(category, 0):
            continue
        haystack_id = str(haystack.get("haystack_id") or haystack_index)
        qas = list(haystack.get("qas") or [])
        pending_qas: list[tuple[int, dict[str, Any], str]] = []
        for qa_index, qa in enumerate(qas):
            if completed_by_category.get(category, 0) + len(pending_qas) >= target_by_category.get(category, 0):
                break
            question = str(qa.get("question") or "")
            qa_pair_id = str(qa.get("qa_pair_id") or qa.get("question_id") or qa_index)
            key = (haystack_id, qa_pair_id)
            if key not in seen:
                pending_qas.append((qa_index, qa, qa_pair_id))
        if not pending_qas:
            continue
        context_text = str(haystack.get("context_text") or "")
        task = _memoryagentbench_task_prompt(haystack)
        if hasattr(method, "reset_counters"):
            method.reset_counters()
        build_t0 = time.time()
        memory = method.build(context_text, task=task)
        build_sec = time.time() - build_t0
        build_counters = _counter_snapshot(method)
        persistent_bytes = 0
        if hasattr(method, "persistent_store_bytes"):
            try:
                persistent_bytes = int(method.persistent_store_bytes(memory) or 0)
            except Exception:
                persistent_bytes = 0

        for qa_index, qa, qa_pair_id in pending_qas:
            if completed_by_category.get(category, 0) >= target_by_category.get(category, 0):
                stop = True
                break
            question = str(qa.get("question") or "")
            key = (haystack_id, qa_pair_id)
            if key in seen:
                continue

            before_retrieve = _counter_snapshot(method)
            retrieve_t0 = time.time()
            context = method.answer(memory, question)
            retrieve_sec = time.time() - retrieve_t0
            after_retrieve = _counter_snapshot(method)
            retrieval_delta = _counter_delta(before_retrieve, after_retrieve)
            direct_answer = _split_direct_answer(context)
            if direct_answer is not None:
                answer = direct_answer
                retrieved_context = ""
                qa_usage = {"prompt_tokens": 0, "completion_tokens": 0}
                llm_sec = 0.0
            else:
                prompt = _build_generic_memory_answer_prompt(
                    question=question,
                    context=str(context),
                    task=task,
                    category=str(haystack.get("category", "")),
                )
                answer, qa_usage, llm_sec = _call_shared_answer_llm(
                    args,
                    prompt,
                    max_tokens=512,
                )
                retrieved_context = str(context)

            metrics = _metrics_for_gold(answer, qa.get("gold", ""), compute_metrics)
            selection = qa.get("selection") if isinstance(qa.get("selection"), dict) else {}
            t_build_input = int(build_counters.get("build_input_tokens", 0) or 0)
            t_build_output = int(build_counters.get("build_output_tokens", 0) or 0)
            t_query_input = int(retrieval_delta.get("query_input_tokens", 0) or 0) + int(qa_usage.get("prompt_tokens", 0) or 0)
            t_query_output = int(retrieval_delta.get("query_output_tokens", 0) or 0) + int(qa_usage.get("completion_tokens", 0) or 0)
            w_tool = float(retrieval_delta.get("wallclock_seconds", 0.0) or retrieve_sec)
            w_llm = float(retrieval_delta.get("w_llm", 0.0) or 0.0) + float(llm_sec or 0.0)

            row = {
                "benchmark": "memoryagentbench",
                "haystack_id": haystack_id,
                "episode_id": haystack_id,
                "qa_pair_id": qa_pair_id,
                "category": haystack.get("category", ""),
                "source": haystack.get("source", ""),
                "question": question,
                "prediction": answer,
                "gold": qa.get("gold", ""),
                "retrieved_context": retrieved_context,
                "retrieved_chars": len(retrieved_context),
                "question_type": qa.get("question_type", ""),
                "question_date": qa.get("question_date", ""),
                "previous_event": qa.get("previous_event", ""),
                "qa_index": qa_index,
                "context_tokens": haystack.get("context_tokens", selection.get("context_tokens")),
                "context_chars": haystack.get("context_chars", selection.get("context_chars", len(context_text))),
                "length_quartile": selection.get("length_quartile"),
                "metrics": metrics,
                "build_sec": round(build_sec, 3),
                "retrieve_sec": round(retrieve_sec, 3),
                "llm_sec": round(llm_sec, 3),
                "efficiency": {
                    "t_build_input": t_build_input,
                    "t_build_output": t_build_output,
                    "t_query_input": t_query_input,
                    "t_query_output": t_query_output,
                    "w_llm": round(w_llm, 3),
                    "w_tool": round(w_tool, 3),
                    "gpu_util_mean": None,
                },
                "method_counters_build": build_counters,
                "method_counters_query_delta": retrieval_delta,
                "persistent_store_bytes": persistent_bytes,
                "run_config": str(run_config_path),
            }
            with answers_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            seen.add(key)
            results.append(row)
            completed_by_category[category] = completed_by_category.get(category, 0) + 1

        if stop:
            continue

    by_category: dict[str, int] = {}
    for row in results:
        cat = str(row.get("category") or "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    metric_summary = _summarize_memoryagentbench_rows(results)
    summary = {
        "benchmark": "memoryagentbench",
        "method": method_name,
        "backbone": _backbone_tag(args),
        "categories": categories,
        "num_haystacks": len(dataset),
        "num_questions": len(results),
        "by_category": by_category,
        "metrics": metric_summary,
        "answers_path": str(answers_path),
        "run_config": str(run_config_path),
        "manifest": getattr(args, "mab_manifest", None),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"benchmark": "memoryagentbench", "method": method_name, "results": results, "summary": summary}

def _run_alfworld(method_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run a method on ALFWorld benchmark."""
    from agentmem.eval.alfworld_runner.runner import run_alfworld_eval

    summary = run_alfworld_eval(
        method_name=method_name,
        max_episodes=args.max_episodes or 10,
        max_steps=getattr(args, "alfworld_max_steps", 30),
        max_trials=getattr(args, "alfworld_max_trials", 1),
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=getattr(args, "llm_api_key", None),
        embedding_model=getattr(args, "embedding_model", None),
        embedding_base_url=getattr(args, "embedding_base_url", None),
        embedding_api_key=getattr(args, "embedding_api_key", None),
        output_dir=str(_output_dir(args, "alfworld", method_name)),
    )
    return {"benchmark": "alfworld", "method": method_name, "results": [], "summary": summary}

def _run_amabench(method_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run a method on AMABench.

    Delegates to the existing amabench runner infrastructure for methods
    that have amabench-specific adapters, and falls back to the unified
    method interface for others.
    """
    from agentmem.methods import build_method

    method = build_method(method_name, **_method_kwargs(args))
    answer_fn = _make_answer_fn(args) if method_name in _ANSWER_FN_METHODS else None
    progress_file = _output_dir(args, "amabench", method_name) / f"answers_{method_name}_{_backbone_tag(args)}.jsonl"
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    if progress_file.exists():
        with progress_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qa_index = str(row.get("qa_index", ""))
                key = (str(row.get("episode_id", "")), qa_index, str(row.get("question", "")))
                if key in seen:
                    continue
                seen.add(key)
                results.append(row)

    dataset = _load_amabench_dataset(args)
    domain_filter = _amabench_domain_filter(args)
    if domain_filter is not None:
        dataset = [ep for ep in dataset if (ep.get("domain") or "").lower() in domain_filter]

    for i, episode in enumerate(dataset):
        if args.max_episodes and i >= args.max_episodes:
            break

        traj_text = _amabench_trajectory_to_text(episode.get("trajectory", []))
        task = episode.get("task", "")
        memory = method.build(traj_text, task=task)

        qas = episode.get("qa_pairs", [])
        if getattr(args, "max_questions", None):
            qas = qas[: args.max_questions]
        for qa_index, qa in enumerate(qas):
            episode_id = str(episode.get("episode_id", str(i)))
            key = (episode_id, str(qa_index), str(qa.get("question", "")))
            if key in seen:
                continue
            context = method.answer(memory, qa["question"])
            direct_answer = _split_direct_answer(context)
            if direct_answer is not None:
                answer = direct_answer
            elif answer_fn is not None:
                answer = answer_fn(
                    _build_generic_memory_answer_prompt(
                        question=qa["question"],
                        context=context,
                        task="",
                        category=str(qa.get("category", "")),
                    ),
                    max_tokens=512,
                )
            else:
                answer = context
            row = {
                "episode_id": episode_id,
                "method": method_name,
                "qa_index": qa_index,
                "question": qa["question"],
                "prediction": answer,
                "gold": qa.get("answer", ""),
                "domain": episode.get("domain", ""),
                "retrieved_context": context if answer_fn is not None and direct_answer is None else "",
            }
            with progress_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            seen.add(key)
            results.append(row)

        if hasattr(method, "reset_counters") and not str(getattr(method, "name", "")).startswith("automem"):
            method.reset_counters()

    payload = {"benchmark": "amabench", "method": method_name, "results": results}
    if getattr(args, "track_efficiency", False) and hasattr(method, "_counters"):
        try:
            payload["counters"] = method._counters.to_dict()
            payload["fs_counters"] = dict(getattr(method, "_fs_counters", {}))
        except Exception:
            pass
    return payload

def _amabench_domain_filter(args: argparse.Namespace) -> set[str] | None:
    """Return the set of `domain` values to keep, or None for no filter.

    The CLI flag --amabench-domain selects one of the three reported subsets:
        text2sql -> {"sql"}
        embodied -> {"game", "embodied"}
        webarena -> {"web"}
    """
    raw = getattr(args, "amabench_domain", "all")
    if not raw or raw == "all":
        return None
    buckets = {
        "text2sql": {"text2sql", "sql"},
        "embodied": {"embodied_ai", "embodied", "game"},
        "webarena": {"web", "webarena"},
    }
    return buckets.get(raw, None)

def _run_memoryarena(method_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run MemoryArena (multi-session) for one config × one method.

    Wires the MultiSessionMemory adapter for the given method into the
    streaming runner. The method's adapter must live under
    ``agentmem/eval/memoryarena_runner/adapters/<method_name>.py``.
    """
    from agentmem.eval.memoryarena_runner import run_memoryarena
    from agentmem.eval.memoryarena_runner import adapters as ma_adapters
    from agentmem.eval.memoryarena_runner.group_travel_env import (
        ReconstructedGroupTravelEnv,
        render_group_travel_environment,
    )
    from agentmem.eval.memoryarena_runner.shopping_env import (
        ReconstructedBundledShoppingEnv,
        render_bundled_shopping_environment,
    )

    config = getattr(args, "memoryarena_config", None) or "progressive_search"

    adapter_factory = getattr(ma_adapters, _adapter_class_name(method_name), None)
    if adapter_factory is None:
        raise NotImplementedError(
            f"No multi-session adapter for method {method_name!r} yet. "
            f"See agentmem/eval/memoryarena_runner/adapters/longcontext.py for a reference."
        )
    if method_name == "longcontext":
        memory = adapter_factory(
            max_chars=getattr(args, "memoryarena_memory_max_chars", None)
        )
    else:
        memory = adapter_factory(
            method_kwargs=_memoryarena_method_kwargs(method_name, args),
            max_chars=getattr(args, "memoryarena_memory_max_chars", 24000),
        )

    agent_mode = getattr(args, "memoryarena_agent_mode", "search")
    agent_env_fn = None
    if agent_mode in {"group_travel_tools", "bundled_shopping_tools"}:
        agent_fn = _make_memoryarena_agent_fn(args)
        agent_env_fn = _make_memoryarena_tool_agent_fn(args)
    else:
        agent_fn = _make_memoryarena_agent_fn(args)
    env_mode = getattr(args, "memoryarena_env_mode", "none")
    if agent_mode == "group_travel_tools" and env_mode == "none":
        env_mode = "paper_sim"
    if agent_mode == "bundled_shopping_tools" and env_mode == "none":
        env_mode = "paper_sim"
    if env_mode == "oracle" and os.environ.get("ALLOW_ORACLE_MEMORYARENA", "").lower() not in {"1", "true", "yes"}:
        raise ValueError(
            "MemoryArena env_mode='oracle' exposes current-session gold values and is blocked "
            "for paper-grade evaluation. Set ALLOW_ORACLE_MEMORYARENA=1 only for debugging."
        )
    env_max_candidates = getattr(args, "memoryarena_env_max_candidates", 12)

    def environment_fn(task: dict[str, Any], session_id: int, question: str) -> str:
        if env_mode in {"", "none", None}:
            return ""
        if config == "group_travel_planner" and agent_mode != "group_travel_tools":
            return render_group_travel_environment(
                task,
                session_id=session_id,
                question=question,
                mode=env_mode,
                max_candidates_per_slot=env_max_candidates,
            )
        if config == "bundled_shopping" and agent_mode != "bundled_shopping_tools":
            return render_bundled_shopping_environment(
                task,
                session_id=session_id,
                question=question,
                mode=env_mode,
                max_candidates=env_max_candidates,
            )
        return ""

    def environment_factory(task: dict[str, Any], session_id: int, question: str):
        if config == "group_travel_planner" and agent_mode == "group_travel_tools":
            return ReconstructedGroupTravelEnv(
                task,
                session_id=session_id,
                question=question,
                mode=env_mode,
                max_candidates_per_slot=env_max_candidates,
            )
        if config == "bundled_shopping" and agent_mode == "bundled_shopping_tools":
            return ReconstructedBundledShoppingEnv(
                task,
                session_id=session_id,
                question=question,
                mode=env_mode,
                max_candidates=env_max_candidates,
            )
        return None

    progress_dir = _output_dir(args, "memoryarena", method_name)
    backbone = (args.llm_model or "unknown").lower().replace("/", "-").replace("_", "-")
    limit_tag = args.max_episodes if args.max_episodes is not None else "all"
    progress_path = progress_dir / f"partial_results_{config}_{backbone}_n{limit_tag}.jsonl"

    out = run_memoryarena(
        config=config,
        memory=memory,
        agent_fn=agent_fn,
        llm_judge_fn=None,
        limit=args.max_episodes,
        use_manifest=True,
        source=getattr(args, "memoryarena_source", "auto"),
        progress_path=str(progress_path),
        resume=True,
        environment_fn=environment_fn if env_mode not in {"", "none", None} else None,
        environment_factory=environment_factory if agent_env_fn is not None else None,
        agent_env_fn=agent_env_fn,
    )
    out["method"] = method_name
    out["memoryarena_env_mode"] = env_mode
    out["memoryarena_agent_mode"] = getattr(args, "memoryarena_agent_mode", "search")
    return out

def _adapter_class_name(method_name: str) -> str:
    return {
        "longcontext": "LongContextAdapter",
        "simplemem": "SimpleMemAdapter",
        "lightmem": "LightMemAdapter",
        "hipporag": "HippoRAGAdapter",
        "plugmem": "PlugMemAdapter",
        "ama_agent": "AMAAgentAdapter",
        "memt": "MemTAdapter",
        "memrl": "MemRLAdapter",
        "mem0": "Mem0Adapter",
        "memoryos": "MemoryOSAdapter",
        "amem": "AMemAdapter",
        "autoharness": "AutoHarnessAdapter",
        "dci_lite": "DCILiteAdapter",
        "dci_lite_sum": "DCILiteSummarizeAdapter",
        "automem": "AutoMemAdapter",
    }.get(method_name, "")

def _memoryarena_method_kwargs(method_name: str, args: argparse.Namespace) -> dict[str, Any]:
    kwargs = _method_kwargs(args)
    kwargs.setdefault("llm_api_key", os.environ.get("OPENAI_API_KEY", "EMPTY"))
    kwargs.setdefault(
        "embedding_api_key",
        getattr(args, "embedding_api_key", None)
        or os.environ.get("EMBEDDING_API_KEY")
        or os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    save_root = _output_dir(args, "memoryarena", method_name) / "method_state"
    kwargs.setdefault("save_dir", str(save_root))

    if method_name in {"autoharness", "dci_lite", "dci_lite_sum"}:
        if args.llm_model:
            kwargs.setdefault("llm_model", args.llm_model)
            kwargs.setdefault("model", args.llm_model)
        if args.llm_base_url:
            kwargs.setdefault("llm_base_url", args.llm_base_url)
            kwargs.setdefault("base_url", args.llm_base_url)
        kwargs.setdefault("llm_api_key", os.environ.get("OPENAI_API_KEY", "EMPTY"))
        kwargs.setdefault("api_key", kwargs.get("llm_api_key", "EMPTY"))
        kwargs.setdefault("top_k", int(os.environ.get("DCI_TOP_K", "10")))
        if method_name == "dci_lite":
            kwargs.setdefault("context_level", "level3")
        if method_name == "dci_lite_sum":
            kwargs.setdefault("context_level", "level4")

    return kwargs

def _make_memoryarena_agent_fn(args: argparse.Namespace):
    """Build a (prompt) -> (prediction, trace) callable using args.llm_*."""
    from openai import OpenAI

    from agentmem.eval.memoryarena_runner.search_tools import render_multi_search_observations

    client = OpenAI(
        base_url=args.llm_base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    model = args.llm_model
    mode = getattr(args, "memoryarena_agent_mode", "search")

    def agent_fn(prompt: str) -> tuple[str, str]:
        search_block = ""
        if mode == "search":
            question = _extract_memoryarena_question(prompt)
            overall = _extract_memoryarena_overall_question(prompt)
            queries = _memoryarena_search_queries(question, overall)
            if queries:
                search_text, _docs = render_multi_search_observations(
                    queries,
                    top_k=getattr(args, "memoryarena_search_top_k", 5),
                    truncate_tokens=getattr(args, "memoryarena_search_truncate_tokens", 512),
                )
                search_block = (
                    "\n\nSearch environment observations for the current subtask:\n"
                    f"{search_text}\n"
                )
        augmented_prompt = "/no_think\n" + prompt + search_block
        _ma_kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": augmented_prompt}],
            temperature=getattr(args, "memoryarena_temperature", 0.1),
            max_tokens=getattr(args, "memoryarena_max_tokens", 15000),
        )
        try:
            resp = client.chat.completions.create(**_ma_kwargs)
        except Exception as exc:
            if "maximum context length" not in str(exc):
                raise
            resp = None
            last_exc: Exception = exc
            for retry_max_tokens in (4096, 2048, 1024, 512, 256):
                _ma_retry_kwargs = dict(
                    model=model,
                    messages=[{"role": "user", "content": augmented_prompt}],
                    temperature=getattr(args, "memoryarena_temperature", 0.1),
                    max_tokens=retry_max_tokens,
                )
                try:
                    resp = client.chat.completions.create(**_ma_retry_kwargs)
                    break
                except Exception as retry_exc:
                    last_exc = retry_exc
                    if "maximum context length" not in str(retry_exc):
                        raise
            if resp is None:
                raise last_exc
        text = _strip_hidden_reasoning(resp.choices[0].message.content or "")
        trace = f"{search_block}\n\nModel response:\n{text}" if search_block else text
        return text, trace

    return agent_fn

def _make_memoryarena_tool_agent_fn(args: argparse.Namespace):
    """Build an agent_fn that interacts with a reconstructed MemoryArena env.

    The env object owns the domain-specific tool schema. The model emits one
    JSON object per turn, either ``{"tool": ...}`` or ``{"final": ...}``.
    """
    from openai import OpenAI

    from agentmem.eval.memoryarena_runner.group_travel_env import parse_tool_or_final

    client = OpenAI(
        base_url=args.llm_base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    model = args.llm_model
    max_steps = getattr(args, "memoryarena_env_max_steps", 8)

    def agent_fn(prompt: str, env) -> tuple[str, str]:
        messages = [
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    f"{prompt}\n\n"
                    "You are in tool-use mode. Follow the environment tool schema above. "
                    "Emit exactly one JSON object at a time. First call the relevant tool if the current answer "
                    "depends on an external environment result. After enough observations, return only a JSON "
                    "object with a `final` key."
                ),
            }
        ]
        trace_parts: list[str] = []
        final_text = ""
        for step in range(max_steps):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=getattr(args, "memoryarena_temperature", 0.1),
                max_tokens=min(getattr(args, "memoryarena_max_tokens", 4096), 4096),
            )
            text = _strip_hidden_reasoning(resp.choices[0].message.content or "")
            trace_parts.append(f"Assistant step {step}:\n{text}")
            action = parse_tool_or_final(text)
            if isinstance(action, dict) and "final" in action:
                final_value = action.get("final")
                final_text = json.dumps(final_value, ensure_ascii=False, indent=2) if not isinstance(final_value, str) else final_value
                break
            if not isinstance(action, dict) or "tool" not in action:
                final_text = text
                break

            observation = env.handle_action(action)
            obs_text = json.dumps(observation, ensure_ascii=False)
            trace_parts.append(f"Environment step {step}:\n{obs_text}")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Tool observation:\n{obs_text}\nNow continue or return final."})
        if not final_text:
            final_text = trace_parts[-1] if trace_parts else ""
        return final_text, "\n\n".join(trace_parts)

    return agent_fn

def _extract_memoryarena_question(prompt: str) -> str:
    marker = "Current subtask:\n"
    if marker not in prompt:
        return prompt
    tail = prompt.split(marker, 1)[1]
    return tail.split("\n\nInstructions:", 1)[0].strip()

def _extract_memoryarena_overall_question(prompt: str) -> str:
    marker = "Original full question / task goal:\n"
    if marker not in prompt:
        return ""
    tail = prompt.split(marker, 1)[1]
    return tail.split("\n\nBackground:", 1)[0].strip()

def _memoryarena_search_queries(question: str, overall: str) -> list[str]:
    condensed = _condense_memoryarena_query((question + " " + overall).strip())
    queries = [condensed] if condensed else []
    queries.append(question)
    if overall and overall != question:
        queries.append(overall)
    return queries[:1]

def _condense_memoryarena_query(text: str) -> str:
    quoted = re.findall(r'"([^"]+)"|' + r"'([^']+)'", text)
    phrases = [a or b for a, b in quoted]
    years = re.findall(r"\b(?:18|19|20)\d{2}\b", text)
    caps = [
        item
        for item in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b", text)
        if item.lower() not in {"what is", "there is"}
    ]
    keywords = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z-]{3,}", text)
        if token.lower()
        not in {
            "that",
            "with",
            "from",
            "between",
            "inclusive",
            "there",
            "which",
            "what",
            "when",
            "where",
            "college",
            "person",
            "someone",
            "question",
        }
    ]
    priority = [
        token
        for token in keywords
        if token.lower()
        in {
            "thesis",
            "battalion",
            "drum",
            "broadcasting",
            "writer",
            "french",
            "majoring",
            "graduated",
            "quit",
            "quitting",
        }
    ]
    pieces = phrases + caps[:8] + years[:6] + priority + keywords[:18]
    deduped: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        norm = piece.lower()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(piece)
    return " ".join(deduped[:24])

_BENCHMARK_RUNNERS = {
    "hotpotqa": _run_hotpotqa,
    "locomo": _run_locomo,
    "memoryagentbench": _run_memoryagentbench,
    "alfworld": _run_alfworld,
    "amabench": _run_amabench,
    "memoryarena": _run_memoryarena,
}

def _load_hotpotqa_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load HotpotQA dataset from local file or HF."""
    if args.test_file:
        return _load_jsonl(args.test_file)
    local = _REPO_ROOT / "datasets" / "hotpotqa"
    candidates = list(local.glob("hotpot_dev_distractor*.json")) + list(local.glob("*.jsonl"))
    if candidates:
        return _load_jsonl(str(candidates[0]))
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
        return list(ds)
    except Exception:
        raise FileNotFoundError(
            "HotpotQA dataset not found. Provide --test-file or place data at "
            "datasets/hotpotqa/hotpot_dev_distractor_v1.json"
        )

def _load_locomo_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load LoCoMo dataset from HF or local file."""
    if args.test_file:
        return _load_jsonl(args.test_file)
    try:
        from datasets import load_dataset
        ds = load_dataset("locomo-bench/LoCoMo-Plus", split="test")
        return list(ds)
    except Exception:
        local_roots = [
            _REPO_ROOT / "datasets" / "locomo",
            _REPO_ROOT / "datasets" / "locomo-plus",
        ]
        candidates = [
            p
            for local in local_roots
            for p in (list(local.glob("*.jsonl")) + list(local.glob("*.json")))
        ]
        if candidates:
            return _load_jsonl(str(candidates[0]))
        raise FileNotFoundError(
            "LoCoMo dataset not found. Provide --test-file, install datasets library, "
            "or place JSON/JSONL under datasets/locomo or datasets/locomo-plus."
        )

def _load_amabench_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load AMABench dataset from local file."""
    if args.test_file:
        return _load_jsonl(args.test_file)
    default = _REPO_ROOT / "datasets" / "amabench" / "open_end_qa_set_medium.jsonl"
    if default.exists():
        return _load_jsonl(str(default))
    raise FileNotFoundError(
        "AMABench dataset not found. Provide --test-file or place data at "
        "datasets/amabench/open_end_qa_set_medium.jsonl"
    )

def _load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a JSONL or JSON file."""
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _method_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build kwargs for build_method() from CLI args."""
    kwargs: dict[str, Any] = {}
    if args.config_path:
        kwargs["config_path"] = args.config_path
    if args.llm_model:
        kwargs["llm_model"] = args.llm_model
    if args.llm_base_url:
        kwargs["llm_base_url"] = args.llm_base_url
    llm_key = getattr(args, "llm_api_key", None)
    if llm_key:
        kwargs["llm_api_key"] = llm_key
    emb_model = getattr(args, "embedding_model", None)
    emb_base = getattr(args, "embedding_base_url", None)
    emb_key = getattr(args, "embedding_api_key", None)
    if emb_model:
        kwargs["embedding_model"] = emb_model
    if emb_base:
        kwargs["embedding_base_url"] = emb_base
    if emb_key:
        kwargs["embedding_api_key"] = emb_key
    return kwargs

def _backbone_tag(args: argparse.Namespace) -> str:
    """Sluggify ``--llm-model`` into a filename-safe backbone tag."""
    raw = (getattr(args, "llm_model", None) or "unknown").lower()
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-") or "unknown"

def _benchmark_subdir(args: argparse.Namespace, benchmark: str) -> str:
    """Canonical per-benchmark subdir under results/.

    Layout: ``results/<benchmark>_<config>_<backbone>/<method>/``. We synthesize
    ``<config>`` from args (n=200 + hippo2 for hotpotqa, n=100 for locomo, etc.).
    Override via ``--output-dir`` if you want a custom path.
    """
    bb = _backbone_tag(args)
    if benchmark == "hotpotqa":
        return f"hotpotqa_n200_{bb}_hippo2"
    if benchmark == "locomo":
        return f"locomo_n100_{bb}"
    if benchmark == "memoryagentbench":
        cats = "-".join(str(c).lower() for c in getattr(args, "mab_categories", ["AR", "TTL", "LRU", "CR"]))
        return f"memoryagentbench_{cats}_{bb}"
    if benchmark == "alfworld":
        return f"alfworld_n10_{bb}"
    if benchmark == "memoryarena":
        cfg = getattr(args, "memoryarena_config", "bundled_shopping")
        return f"memoryarena_{cfg}_{bb}"
    if benchmark == "amabench":
        domain = getattr(args, "amabench_domain", "all")
        return f"amabench_{domain}_{bb}"
    return f"{benchmark}_{bb}"

def _output_dir(args: argparse.Namespace, benchmark: str, method: str) -> Path:
    base = Path(args.output_dir) if args.output_dir else _REPO_ROOT / "results"
    if getattr(args, "flat_output_dir", False):
        return base / method
    d = base / _benchmark_subdir(args, benchmark) / method
    d.mkdir(parents=True, exist_ok=True)
    return d

def _amabench_trajectory_to_text(trajectory: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for step in trajectory:
        idx = step.get("turn_idx", 0)
        parts.append(f"Turn {idx}:")
        parts.append(f"  Action: {step.get('action', '')}")
        parts.append(f"  Observation: {step.get('observation', '')}")
    return "\n".join(parts)

def _locomo_episode_to_text(episode: dict[str, Any]) -> str:
    """Normalize LoCoMo / LoCoMo-Plus rows into a flat conversation transcript."""
    if episode.get("trajectory"):
        return str(episode["trajectory"])
    conversation = episode.get("conversation") or {}
    if isinstance(conversation, str):
        return conversation
    parts: list[str] = []
    if isinstance(conversation, dict):
        for key in sorted(conversation):
            value = conversation[key]
            if key.endswith("_date_time"):
                continue
            if not key.startswith("session_") or not isinstance(value, list):
                continue
            date = conversation.get(f"{key}_date_time")
            header = f"\n# {key}"
            if date:
                header += f" | date_time={date}"
            parts.append(header)
            for turn in value:
                if not isinstance(turn, dict):
                    parts.append(str(turn))
                    continue
                dia_id = turn.get("dia_id", "")
                speaker = turn.get("speaker", "Speaker")
                text = turn.get("text", "")
                parts.append(f"{dia_id} {speaker}: {text}".strip())
    for key in ("session_summary", "event_summary", "observation"):
        value = episode.get(key)
        if value:
            parts.append(f"\n# {key}\n{json.dumps(value, ensure_ascii=False)}")
    return "\n".join(parts).strip()

def _build_generic_memory_answer_prompt(
    *,
    question: str,
    context: str,
    task: str = "",
    category: str = "",
) -> str:
    task_block = f"Task:\n{task}\n\n" if task else ""
    category = category.upper()

    if (
        os.environ.get("MAB_DISABLE_MC_VERBATIM", "") != "1"
        and "list of possible subsequent events" in question.lower()
    ):
        return (
            "You are given a list of candidate events and a book excerpt in the "
            "retrieved memory context. Choose the single candidate that happens "
            "next according to the excerpt. Copy that candidate EXACTLY as written "
            "(verbatim, character-for-character); output nothing else.\n\n"
            f"{task_block}"
            f"Retrieved memory context:\n{context or 'None'}\n\n"
            f"Question: {question}\n"
            "Answer (one candidate, copied verbatim):"
        )
    if category == "TTL":
        return (
            "Answer the test-time-learning query using only the retrieved examples. "
            "The examples map utterances to labels or outputs. Return exactly the learned label/output, "
            "with no words, explanation, or semantic answer.\n\n"
            f"{task_block}"
            f"Retrieved memory context:\n{context or 'None'}\n\n"
            f"Query: {question}\n"
            "Answer:"
        )
    return (
        "Answer the question using only the retrieved memory context. "
        "If the answer is unsupported, say so briefly. Return only the answer.\n\n"
        f"{task_block}"
        f"Retrieved memory context:\n{context or 'None'}\n\n"
        f"Question: {question}"
    )

def _estimate_tokens(text: Any) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(str(text or "")))
    except Exception:
        return max(0, int(len(str(text or "")) / 4))

def _cap_text_to_token_budget(text: Any, max_tokens: int) -> str:
    content = str(text or "")
    if max_tokens <= 0:
        return content
    marker = "\n\n[Retrieved memory context truncated to fit the model context window.]\n\n"
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(content)
        if len(tokens) <= max_tokens:
            return content
        marker_tokens = enc.encode(marker)
        remaining = max(1, max_tokens - len(marker_tokens))
        head_budget = max(128, min(1024, remaining // 4))
        tail_budget = max(1, remaining - head_budget)
        return enc.decode(tokens[:head_budget]) + marker + enc.decode(tokens[-tail_budget:])
    except Exception:
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        marker_chars = len(marker)
        remaining_chars = max(1, max_chars - marker_chars)
        head_chars = max(512, min(4096, remaining_chars // 4))
        tail_chars = max(1, remaining_chars - head_chars)
        return content[:head_chars] + marker + content[-tail_chars:]

def _call_shared_answer_llm(
    args: argparse.Namespace,
    prompt: str,
    *,
    max_tokens: int,
) -> tuple[str, dict[str, int], float]:
    from openai import OpenAI

    client = OpenAI(
        base_url=args.llm_base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key=getattr(args, "llm_api_key", None) or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        timeout=300.0,
    )
    context_tokens = int(os.environ.get("MAB_LLM_CONTEXT_TOKENS", "32768"))
    margin_tokens = int(os.environ.get("MAB_ANSWER_PROMPT_MARGIN_TOKENS", "1024"))
    input_cap = int(
        os.environ.get(
            "MAB_ANSWER_MAX_INPUT_TOKENS",
            str(min(25000, max(1024, context_tokens - max_tokens - margin_tokens))),
        )
    )
    content = _cap_text_to_token_budget("/no_think\n" + prompt, input_cap)
    t0 = time.time()

    _shared_extra = (
        {} if os.environ.get("AGENTMEM_DISABLE_THINKING_KWARGS", "").lower() in {"1", "true", "yes"}
        else {"chat_template_kwargs": {"enable_thinking": False}}
    )
    resp = client.chat.completions.create(
        model=args.llm_model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body=_shared_extra,
    )
    elapsed = time.time() - t0
    text = _strip_hidden_reasoning(resp.choices[0].message.content or "")
    usage_obj = getattr(resp, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
    }
    if usage["prompt_tokens"] == 0:
        usage["prompt_tokens"] = _estimate_tokens(content)
    if usage["completion_tokens"] == 0:
        usage["completion_tokens"] = _estimate_tokens(text)
    return text, usage, elapsed

def _counter_snapshot(method: Any) -> dict[str, Any]:
    counters = getattr(method, "counters", None)
    if counters is None or not hasattr(counters, "to_dict"):
        return {}
    try:
        return dict(counters.to_dict())
    except Exception:
        return {}

def _counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in after.items():
        if isinstance(value, (int, float)) and isinstance(before.get(key, 0), (int, float)):
            delta[key] = value - before.get(key, 0)
        elif key == "seconds_by_phase" and isinstance(value, dict):
            prior = before.get(key, {}) if isinstance(before.get(key), dict) else {}
            delta[key] = {
                phase: float(sec) - float(prior.get(phase, 0.0) or 0.0)
                for phase, sec in value.items()
                if isinstance(sec, (int, float))
            }
    if "seconds_by_phase" in delta:
        for phase, sec in delta["seconds_by_phase"].items():
            delta[phase] = sec
    return delta

def _metrics_for_gold(answer: str, gold: Any, compute_metrics_fn: Any) -> dict[str, float]:
    refs = gold if isinstance(gold, list) else [gold]
    if not refs:
        refs = [""]
    scored = [compute_metrics_fn(answer, ref) for ref in refs]
    keys = sorted({k for item in scored for k in item})
    return {key: max(float(item.get(key, 0.0) or 0.0) for item in scored) for key in keys}

def _summarize_memoryagentbench_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("category") or "unknown")].append(row)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        metric_keys = sorted({k for row in items for k in (row.get("metrics") or {})})
        eff_keys = sorted({k for row in items for k in (row.get("efficiency") or {}) if k != "gpu_util_mean"})
        metrics = {
            key: sum(float((row.get("metrics") or {}).get(key, 0.0) or 0.0) for row in items) / max(len(items), 1)
            for key in metric_keys
        }
        efficiency_totals = {
            key: sum(float((row.get("efficiency") or {}).get(key, 0.0) or 0.0) for row in items)
            for key in eff_keys
        }
        return {
            "num_questions": len(items),
            "metrics": metrics,
            "efficiency_totals": efficiency_totals,
            "efficiency_per_question": {
                key: value / max(len(items), 1) for key, value in efficiency_totals.items()
            },
        }

    by_category = {cat: aggregate(items) for cat, items in sorted(groups.items())}
    macro_token_f1 = (
        sum((item["metrics"].get("token_f1", 0.0) or 0.0) for item in by_category.values()) / max(len(by_category), 1)
    )
    return {
        "overall": aggregate(rows),
        "by_category": by_category,
        "macro_category_token_f1": macro_token_f1,
    }

def _memoryagentbench_category_targets(
    dataset: list[dict[str, Any]],
    categories: list[str],
    max_questions: int | None,
) -> dict[str, int]:
    available: dict[str, int] = {str(cat).upper(): 0 for cat in categories}
    for haystack in dataset:
        cat = str(haystack.get("category") or "").upper()
        if cat in available:
            available[cat] += len(haystack.get("qas") or [])
    if max_questions is None:
        return available

    remaining_total = min(int(max_questions), sum(available.values()))
    targets: dict[str, int] = {cat: 0 for cat in available}
    open_categories = [cat for cat in categories if available.get(cat, 0) > 0]
    while remaining_total > 0 and open_categories:
        base, rem = divmod(remaining_total, len(open_categories))
        if base == 0:
            for cat in open_categories[:rem]:
                targets[cat] += 1
            break
        next_open: list[str] = []
        assigned = 0
        for index, cat in enumerate(open_categories):
            quota = base + (1 if index < rem else 0)
            room = available[cat] - targets[cat]
            take = min(quota, room)
            targets[cat] += take
            assigned += take
            if targets[cat] < available[cat]:
                next_open.append(cat)
        remaining_total -= assigned
        if assigned == 0:
            break
        open_categories = next_open
    return targets

def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

def _memoryagentbench_task_prompt(haystack: dict[str, Any]) -> str:
    category = str(haystack.get("category") or "").upper()
    source = str(haystack.get("source") or "")
    instructions = {
        "TTL": (
            "MemoryAgentBench Test-Time Learning. Use the examples in memory to infer the requested label or output. "
            "Follow the question's requested output format exactly."
        ),
        "CR": (
            "MemoryAgentBench Conflict Resolution. The memory is a knowledge pool where newer facts have larger "
            "serial numbers; answer from the newest non-conflicting relevant fact only."
        ),
        "LRU": (
            "MemoryAgentBench Long-Range Understanding. Answer using the long memorized context and follow any "
            "strict output format in the question."
        ),
        "AR": "MemoryAgentBench Accurate Retrieval. Answer using only the memorized context.",
    }
    base = instructions.get(category, "MemoryAgentBench. Answer using only the memorized context.")
    return f"{base}\nCategory: {category}\nSource: {source}".strip()

def _make_answer_fn(args: argparse.Namespace):
    from openai import OpenAI

    client = OpenAI(
        base_url=args.llm_base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key=getattr(args, "llm_api_key", None) or os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    model = args.llm_model
    base_url = str(args.llm_base_url or os.environ.get("OPENAI_BASE_URL", ""))
    extra_body = None
    if base_url and "api.openai.com" not in base_url.lower():
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def answer(prompt: str, *, max_tokens: int = 512) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": "/no_think\n" + prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        return _strip_hidden_reasoning(resp.choices[0].message.content or "")

    return answer

def _strip_hidden_reasoning(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    text = re.sub(r"<think>.*$", "", text, flags=re.I | re.S).strip()
    text = re.sub(r"^\s*(?:final answer|answer)\s*:\s*", "", text, flags=re.I).strip()
    return text

def _save_results(result: dict[str, Any], args: argparse.Namespace) -> Path:
    """Save results to a JSONL file (plus a counters JSON if present)."""
    benchmark = result["benchmark"]
    method = result["method"]
    out_dir = _output_dir(args, benchmark, method)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backbone = (args.llm_model or "unknown").lower().replace("/", "-").replace("_", "-")
    out_file = out_dir / f"results_{backbone}_{ts}.jsonl"
    with open(out_file, "w") as f:
        for row in result.get("results", []):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counters = result.get("counters")
    fs_counters = result.get("fs_counters")
    if counters or fs_counters:
        cfile = out_dir / f"counters_{backbone}_{ts}.json"
        with open(cfile, "w") as cf:
            json.dump({"counters": counters, "fs_counters": fs_counters}, cf, ensure_ascii=False, indent=2)
    return out_file

def run_single(benchmark: str, method: str, args: argparse.Namespace) -> dict[str, Any] | None:
    """Run a single (benchmark, method) pair."""
    compat = _COMPATIBILITY.get(benchmark, set())
    if method not in compat:
        print(f"  [SKIP] {method} is not compatible with {benchmark}")
        return None

    print(f"\n{'='*60}")
    print(f"  Benchmark: {benchmark}  |  Method: {method}")
    print(f"{'='*60}")

    runner = _BENCHMARK_RUNNERS[benchmark]
    t0 = time.time()
    try:
        result = runner(method, args)
    except Exception as e:
        print(f"  [ERROR] {benchmark}/{method}: {e}")
        if args.fail_fast:
            raise
        return None
    elapsed = time.time() - t0

    n = len(result.get("results", []))
    print(f"  Completed: {n} results in {elapsed:.1f}s")

    out_file = _save_results(result, args)
    print(f"  Saved to: {out_file}")
    return result

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for the cross-scenario memory evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--benchmark", "-b", nargs="+", choices=ALL_BENCHMARKS,
                        help="Benchmark(s) to evaluate on")
    parser.add_argument("--method", "-m", nargs="+", choices=ALL_METHODS,
                        help="Method(s) to evaluate")
    parser.add_argument("--sweep", action="store_true",
                        help="Run all compatible method×benchmark combinations")

    parser.add_argument("--test-file", type=str, default=None,
                        help="Override dataset path (JSONL/JSON)")
    parser.add_argument("--datasets-dir", type=str, default="datasets",
                        help="Root directory for managed dataset downloads/cache")
    parser.add_argument("--auto-download", action="store_true",
                        help="Download managed datasets when missing")
    parser.add_argument("--locomo-pool-file", type=str, default=None,
                        help="Locked LoCoMo question pool JSON/JSONL; preserves row order exactly.")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit number of episodes/samples per benchmark")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit QA pairs per selected episode/sample")
    parser.add_argument("--alfworld-max-steps", type=int, default=30,
                        help="Maximum ALFWorld environment steps per episode")
    parser.add_argument("--alfworld-max-trials", type=int, default=1,
                        help="Maximum bounded ALFWorld trials per episode")

    parser.add_argument("--config-path", type=str, default=None,
                        help="Method-specific config file (YAML/JSON)")

    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini",
                        help="LLM model name (paper default Qwen/Qwen3-32B)")
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Embedding model name (paper default Qwen3-Embedding-4B)")
    parser.add_argument("--embedding-base-url", type=str, default=None,
                        help="Embedding API base URL (e.g. http://localhost:30001/v1 when co-hosting on B200)")
    parser.add_argument("--embedding-api-key", type=str, default=None,
                        help="Embedding API key (defaults to OPENAI_API_KEY)")
    parser.add_argument("--llm-base-url", type=str, default=None,
                        help="LLM API base URL")
    parser.add_argument("--llm-api-key", type=str, default=None,
                        help="LLM API key (defaults to OPENAI_API_KEY)")

    parser.add_argument("--amabench-domain",
                        choices=["all", "text2sql", "embodied", "webarena"], default="all",
                        help="Restrict AMABench to a single reported domain")
    parser.add_argument("--mab-categories", nargs="+", default=["AR", "TTL", "LRU", "CR"],
                        choices=["AR", "TTL", "LRU", "CR"],
                        help="MemoryAgentBench categories to run")
    parser.add_argument("--mab-manifest", default=None,
                        help="Optional MemoryAgentBench selection manifest path for provenance")
    parser.add_argument("--memoryarena-config",
                        choices=["bundled_shopping", "progressive_search", "group_travel_planner"],
                        default="progressive_search",
                        help="Which MemoryArena config to evaluate")
    parser.add_argument("--memoryarena-source",
                        choices=["auto", "local", "hf"], default="auto",
                        help="MemoryArena data source. auto prefers local datasets/memoryarena before HF.")
    parser.add_argument("--memoryarena-temperature", type=float, default=0.1,
                        help="MemoryArena task-agent decoding temperature from the paper setup")
    parser.add_argument("--memoryarena-max-tokens", type=int, default=15000,
                        help="MemoryArena task-agent max output tokens from the paper setup")
    parser.add_argument("--memoryarena-agent-mode",
                        choices=["search", "single", "group_travel_tools", "bundled_shopping_tools"], default="search",
                        help=(
                            "search runs a top-k web-search observation before each subtask; "
                            "single is no-search offline smoke; *_tools modes run reconstructed task environments"
                        ))
    parser.add_argument("--memoryarena-search-top-k", type=int, default=5,
                        help="Progressive Web Search retriever top-k from the paper setup")
    parser.add_argument("--memoryarena-search-truncate-tokens", type=int, default=512,
                        help="Maximum tokens kept per search result from the paper setup")
    parser.add_argument("--memoryarena-memory-max-chars", type=int, default=24000,
                        help="Maximum characters of retrieved MemoryArena memory context injected into the agent prompt")
    parser.add_argument("--memoryarena-env-mode",
                        choices=["none", "catalog", "oracle", "paper_sim"],
                        default="none",
                        help=(
                            "Optional reconstructed MemoryArena environment. "
                            "'catalog' reconstructs an unlabeled diagnostic candidate catalog from the HF task snapshot; "
                            "'paper_sim' avoids current-session gold leakage when original catalogs are unavailable; "
                            "'oracle' exposes/ranks current-session gold values and is only for debugging."
                        ))
    parser.add_argument("--memoryarena-env-max-candidates", type=int, default=12,
                        help="Maximum reconstructed group-travel environment candidates per requested slot")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Results output directory (default: results/)")
    parser.add_argument("--flat-output-dir", action="store_true",
                        help="Write each method directly under --output-dir/<method>")
    parser.add_argument("--track-efficiency", action="store_true",
                        help="Track and report EfficiencyCounters")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be executed without running")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on first error instead of continuing")

    args = parser.parse_args()

    if args.sweep:
        benchmarks = list(ALL_BENCHMARKS)
        methods = list(ALL_METHODS)
    elif args.benchmark and args.method:
        benchmarks = args.benchmark
        methods = args.method
    else:
        parser.error("Specify --benchmark and --method, or use --sweep")

    plan: list[tuple[str, str]] = []
    for b in benchmarks:
        compat = _COMPATIBILITY.get(b, set())
        for m in methods:
            if m in compat:
                plan.append((b, m))

    print(f"Run plan: {len(plan)} combinations")
    for b, m in plan:
        print(f"  {b} × {m}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without executing.")
        return

    all_results: list[dict[str, Any]] = []
    for b, m in plan:
        result = run_single(b, m, args)
        if result:
            all_results.append(result)

    print(f"\n{'='*60}")
    print(f"  Summary: {len(all_results)}/{len(plan)} completed")
    print(f"{'='*60}")
    for r in all_results:
        n = len(r.get("results", []))
        print(f"  {r['benchmark']:15s} × {r['method']:15s}: {n} results")

if __name__ == "__main__":
    main()
