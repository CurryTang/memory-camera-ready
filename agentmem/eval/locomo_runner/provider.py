"""Provider configuration helpers for evaluation tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

def _shared_provider_kwargs(args: argparse.Namespace, *, task: str) -> dict[str, Any]:
    """Build shared kwargs for OpenAICompatibleProvider from CLI args."""
    enable_history = bool(getattr(args, "openai_enable_history", False))
    history_dir = getattr(args, "openai_history_dir", Path("results/llm_history"))
    run_tag = str(getattr(args, "run_tag", "default"))
    call_cache_dir = Path(str(history_dir)).parent / "llm_cache"
    return {
        "enable_history": enable_history,
        "history_dir": history_dir,
        "run_tag": run_tag,
        "enable_call_cache": True,
        "call_cache_dir": call_cache_dir,
        "max_concurrent_requests": int(getattr(args, "openai_max_concurrent_requests", 8)),
        "min_remaining_requests_threshold": int(
            getattr(args, "openai_min_remaining_requests_threshold", 1)
        ),
        "min_remaining_tokens_threshold": int(
            getattr(args, "openai_min_remaining_tokens_threshold", 1024)
        ),
        "reset_safety_seconds": float(getattr(args, "openai_reset_safety_seconds", 0.5)),
        "history_metadata": {
            "task": task,
            "run_tag": run_tag,
        },
    }

def _history_artifact_path(args: argparse.Namespace) -> Optional[Path]:
    """Return the history artifact directory, or None if history is disabled."""
    if not bool(getattr(args, "openai_enable_history", False)):
        return None
    return (Path(getattr(args, "openai_history_dir")) / str(getattr(args, "run_tag", "default"))).resolve()
