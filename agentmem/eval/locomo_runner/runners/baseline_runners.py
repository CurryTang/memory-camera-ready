"""Thin runner wrappers for C1-C9 baselines, LongContext, and HippoRAGv2."""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any, Optional

from agentmem.eval.locomo_runner.types import TaskResult
from agentmem.eval.locomo_runner.runners.shared import _run_baseline_locomo
from agentmem.eval.locomo_runner.adapters.baselines import (
    C1LoCoMoAdapter, C2LoCoMoAdapter, C3LoCoMoAdapter,
    C4LoCoMoAdapter, C5LoCoMoAdapter, C6LoCoMoAdapter,
    C7LoCoMoAdapter, C8LoCoMoAdapter, C9LoCoMoAdapter,
)
from agentmem.eval.locomo_runner.adapters.longcontext import LongContextLoCoMoAdapter
from agentmem.eval.locomo_runner.adapters.hipporagv2 import HippoRAGv2LoCoMoAdapter

def run_c1_bm25_locomo(args: argparse.Namespace, index_cache: Optional[Any] = None) -> TaskResult:
    return _run_baseline_locomo(
        task="c1-bm25-locomo", adapter_cls=C1LoCoMoAdapter,
        adapter_kwargs={
            "answer_model": args.c1_answer_model, "answer_api_key": args.c1_answer_api_key,
            "answer_base_url": args.c1_answer_base_url, "retrieval_topk": args.c1_retrieval_topk,
        },
        args=args, index_cache=index_cache,
    )

def run_c2_keyword_locomo(args: argparse.Namespace, index_cache: Optional[Any] = None) -> TaskResult:
    return _run_baseline_locomo(
        task="c2-keyword-locomo", adapter_cls=C2LoCoMoAdapter,
        adapter_kwargs={
            "answer_model": args.c2_answer_model, "answer_api_key": args.c2_answer_api_key,
            "answer_base_url": args.c2_answer_base_url, "retrieval_topk": args.c2_retrieval_topk,
        },
        args=args, index_cache=index_cache,
    )

def run_c3_dense_locomo(args: argparse.Namespace, index_cache: Optional[Any] = None) -> TaskResult:
    return _run_baseline_locomo(
        task="c3-dense-locomo", adapter_cls=C3LoCoMoAdapter,
        adapter_kwargs={
            "answer_model": args.c3_answer_model, "answer_api_key": args.c3_answer_api_key,
            "answer_base_url": args.c3_answer_base_url, "retrieval_topk": args.c3_retrieval_topk,
            "chunk_size": args.c3_chunk_size, "chunk_overlap": args.c3_chunk_overlap,
            "embedding_model": args.c3_embedding_model,
        },
        args=args, index_cache=index_cache,
    )

def run_c4_compressed_locomo(args: argparse.Namespace, index_cache: Optional[Any] = None) -> TaskResult:
    return _run_baseline_locomo(
        task="c4-compressed-locomo", adapter_cls=C4LoCoMoAdapter,
        adapter_kwargs={
            "answer_model": args.c4_answer_model, "answer_api_key": args.c4_answer_api_key,
            "answer_base_url": args.c4_answer_base_url, "retrieval_topk": args.c4_retrieval_topk,
            "compress_model": args.c4_compress_model, "compress_api_key": args.c4_compress_api_key,
            "embedding_model": args.c4_embedding_model,
        },
        args=args, index_cache=index_cache,
    )

def run_c5_fusion_locomo(args: argparse.Namespace, index_cache: Optional[Any] = None) -> TaskResult:

    from examples.paper_task_eval import run_simplemem_locomo
    return run_simplemem_locomo(args, _task="c5-fusion-locomo")

def run_c6_causal_locomo(args: argparse.Namespace) -> TaskResult:
    return _run_baseline_locomo(
        task="c6-causal-locomo", adapter_cls=C6LoCoMoAdapter,
        adapter_kwargs={
            "answer_model": args.c6_answer_model, "answer_api_key": args.c6_answer_api_key,
            "answer_base_url": args.c6_answer_base_url, "retrieval_topk": args.c6_retrieval_topk,
            "llm_model": args.c6_llm_model, "llm_api_key": args.c6_llm_api_key,
            "state_sufficiency_threshold": args.c6_state_sufficiency_threshold,
        },
        args=args,
    )

def run_c7_concept_locomo(args: argparse.Namespace) -> TaskResult:
    return _run_baseline_locomo(
        task="c7-concept-locomo", adapter_cls=C7LoCoMoAdapter,
        adapter_kwargs={
            "answer_model": args.c7_answer_model, "answer_api_key": args.c7_answer_api_key,
            "answer_base_url": args.c7_answer_base_url, "retrieval_topk": args.c7_retrieval_topk,
            "llm_model": args.c7_llm_model, "llm_api_key": args.c7_llm_api_key,
            "max_hops": args.c7_max_hops,
        },
        args=args,
    )

def run_c8_kg_locomo(args: argparse.Namespace) -> TaskResult:
    return _run_baseline_locomo(
        task="c8-kg-locomo", adapter_cls=C8LoCoMoAdapter,
        adapter_kwargs={
            "answer_model": args.c8_answer_model, "answer_api_key": args.c8_answer_api_key,
            "answer_base_url": args.c8_answer_base_url, "retrieval_topk": args.c8_retrieval_topk,
            "llm_model": args.c8_llm_model, "llm_api_key": args.c8_llm_api_key,
        },
        args=args,
    )

def run_c9_colbert_locomo(args: argparse.Namespace, index_cache: Optional[Any] = None) -> TaskResult:
    return _run_baseline_locomo(
        task="c9-colbert-locomo", adapter_cls=C9LoCoMoAdapter,
        adapter_kwargs={
            "answer_model": args.c9_answer_model, "answer_api_key": args.c9_answer_api_key,
            "answer_base_url": args.c9_answer_base_url, "retrieval_topk": args.c9_retrieval_topk,
            "compress_model": args.c9_compress_model, "compress_api_key": args.c9_compress_api_key,
            "model_name": args.c9_model_name,
        },
        args=args, index_cache=index_cache,
    )

def run_longcontext_locomo(args: argparse.Namespace) -> TaskResult:
    return _run_baseline_locomo(
        task="longcontext-locomo", adapter_cls=LongContextLoCoMoAdapter,
        adapter_kwargs={
            "answer_model": getattr(args, "longcontext_answer_model", None) or args.c1_answer_model,
            "answer_api_key": getattr(args, "longcontext_answer_api_key", None) or args.c1_answer_api_key,
            "answer_base_url": getattr(args, "longcontext_answer_base_url", None) or args.c1_answer_base_url,
            "max_context_tokens": getattr(args, "longcontext_max_tokens", 128000),
        },
        args=args,
    )

def run_hipporagv2_locomo(args: argparse.Namespace) -> TaskResult:
    return _run_baseline_locomo(
        task="hipporagv2-locomo", adapter_cls=HippoRAGv2LoCoMoAdapter,
        adapter_kwargs={
            "answer_model": getattr(args, "hipporag_answer_model", None) or args.c1_answer_model,
            "answer_api_key": getattr(args, "hipporag_answer_api_key", None) or args.c1_answer_api_key,
            "answer_base_url": getattr(args, "hipporag_answer_base_url", None) or args.c1_answer_base_url,
            "embedding_model": getattr(args, "hipporag_embedding_model", "text-embedding-3-small"),
            "embedding_base_url": getattr(args, "hipporag_embedding_base_url", None),
            "llm_model": getattr(args, "hipporag_llm_model", None),
            "llm_base_url": getattr(args, "hipporag_llm_base_url", None),
            "retrieval_topk": getattr(args, "hipporag_retrieval_topk", 15),
            "save_dir": getattr(
                args,
                "hipporag_save_dir",
                os.path.join(tempfile.gettempdir(), "hipporag_locomo"),
            ),
        },
        args=args,
    )
