"""Repo-owned MemRL LoCoMo runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from agentmem.eval.locomo_runner.adapters.memrl import MemRLLoCoMoAdapter
from agentmem.eval.locomo_runner.runners.shared import _run_baseline_locomo
from agentmem.eval.locomo_runner.types import TaskResult

def run_memrl_locomo(args: argparse.Namespace, index_cache: Optional[Any] = None) -> TaskResult:
    artifact_root = Path(
        getattr(args, "memrl_artifact_root", None)
        or getattr(args, "locomo_artifact_root", None)
        or Path("results") / "locomo_artifacts"
    )
    return _run_baseline_locomo(
        task="memrl-locomo",
        adapter_cls=MemRLLoCoMoAdapter,
        adapter_kwargs={
            "answer_model": getattr(args, "memrl_model", "Qwen/Qwen3-32B"),
            "answer_api_key": getattr(args, "memrl_api_key", None),
            "answer_base_url": getattr(args, "memrl_base_url", None),
            "retrieval_topk": getattr(args, "memrl_topk", 5),
            "phase1_topk": getattr(args, "memrl_phase1_topk", 20),
            "alpha": getattr(args, "memrl_alpha", 0.3),
            "gamma": getattr(args, "memrl_gamma", 0.0),
            "epsilon": getattr(args, "memrl_epsilon", 0.1),
            "similarity_weight": getattr(args, "memrl_similarity_weight", 0.5),
            "utility_weight": getattr(args, "memrl_utility_weight", 0.5),
            "use_zscore_normalization": getattr(args, "memrl_use_zscore", True),
            "max_tokens": getattr(args, "memrl_max_tokens", None),
            "artifact_root": artifact_root,
            "task": "memrl-locomo",
        },
        args=args,
        index_cache=index_cache,
    )
