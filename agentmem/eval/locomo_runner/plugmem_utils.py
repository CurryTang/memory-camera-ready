"""PlugMem environment and path resolution utilities."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

def _resolve_plugmem_root(raw_root: Path) -> Optional[Path]:
    """Normalize PlugMem root so that both repo-root and ``code/src`` styles work."""
    try:
        from agentmem.plugmem.bridge import PlugMemBridge
        return PlugMemBridge.resolve_source_root(raw_root)
    except (FileNotFoundError, ImportError):
        return None

def _ensure_plugmem_source_on_path(source_root: Path) -> None:
    """Add PlugMem source root to sys.path if not already present."""
    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)

def _build_plugmem_env_overrides(
    args: argparse.Namespace,
    *,
    task: str,
) -> dict[str, str]:
    """Build environment variable overrides for PlugMem subprocess."""
    from agentmem.eval.amabench_runner.methods.plugmem import _plugmem_qwen_model_name

    del task
    overrides: dict[str, str] = {}
    env_map = {
        "LLM_NAME": getattr(args, "plugmem_llm_name", None),
        "QWEN_MODEL_NAME": getattr(args, "plugmem_qwen_model_name", None)
        or getattr(args, "llm_model", None)
        or _plugmem_qwen_model_name(None),
        "EMBEDDING_MODEL_NAME": getattr(args, "plugmem_embedding_model_name", None),
        "EMBEDDING_BASE_URL": getattr(args, "plugmem_embedding_base_url", None),
        "QWEN_BASE_URL": getattr(args, "plugmem_qwen_base_url", None),
        "VLLM_QWEN_API_KEY": getattr(args, "plugmem_vllm_qwen_api_key", None),
        "OPENAI_API_KEY": getattr(args, "plugmem_openai_api_key", None),
        "OPENROUTER_API_KEY": getattr(args, "plugmem_openrouter_api_key", None),
        "OPENROUTER_DISABLE_REASONING": getattr(args, "plugmem_openrouter_disable_reasoning", None),
        "AZURE_ENDPOINT": getattr(args, "plugmem_azure_endpoint", None),
        "AZURE_DPSK_API_KEY": getattr(args, "plugmem_azure_dpsk_api_key", None),
        "AZURE_DPSK_ENDPOINT": getattr(args, "plugmem_azure_dpsk_endpoint", None),
    }
    for env_key, explicit_value in env_map.items():
        value = explicit_value if explicit_value is not None else os.environ.get(env_key)
        if value:
            overrides[env_key] = str(value)

    token_usage_file = getattr(args, "plugmem_token_usage_file", None)
    if token_usage_file is not None:
        overrides["TOKEN_USAGE_FILE"] = str(Path(token_usage_file).expanduser().resolve())
    elif os.environ.get("TOKEN_USAGE_FILE"):
        overrides["TOKEN_USAGE_FILE"] = os.environ["TOKEN_USAGE_FILE"]

    if getattr(args, "plugmem_write_retrieval_trace", False):
        overrides["WRITE"] = "TRUE"
    elif os.environ.get("WRITE"):
        overrides["WRITE"] = os.environ["WRITE"]

    return overrides
