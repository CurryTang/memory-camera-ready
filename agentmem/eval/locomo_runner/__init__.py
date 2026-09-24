"""LoCoMo evaluation runner — modular framework for memory system evaluation.

Extracted from the monolithic examples/paper_task_eval.py.
"""

from agentmem.eval.locomo_runner.types import TaskResult
from agentmem.eval.locomo_runner.provider import _shared_provider_kwargs, _history_artifact_path
from agentmem.eval.locomo_runner.prompts import (
    _build_locomo_answer_prompt,
    _coerce_int_category,
    _normalize_locomo_prediction,
    _parse_prefixed_dialogue,
    _question_is_binary,
)
from agentmem.eval.locomo_runner.io_utils import (
    _collect_artifacts,
    _count_lines,
    _dump_memrl_store_records,
    _repo_root,
    _run_subprocess,
    _write_jsonl,
)
from agentmem.eval.locomo_runner.locomo_utils import (
    _LOCOMO_CATEGORY_LABELS,
    _LOCOMO_NAMED_TASK_CATEGORY_MAP,
    _compute_locomo_metrics,
    _locomo_category_breakdown,
    _locomo_paper_table_summary,
    _locomo_reference_answer,
    _locomo_rows_path,
    _mean_latency_seconds,
    _mean_metrics,
    _parse_categories,
    _resolve_locomo_category_filter,
    _resolve_max_questions,
    _write_locomo_rows,
)
from agentmem.eval.locomo_runner.plugmem_utils import (
    _build_plugmem_env_overrides,
    _ensure_plugmem_source_on_path,
    _resolve_plugmem_root,
)
from agentmem.eval.locomo_runner.adapters import (
    MemRLLoCoMoAdapter,
)
from agentmem.eval.locomo_runner.runners import run_memrl_locomo

__all__ = [

    "TaskResult",

    "_shared_provider_kwargs",
    "_history_artifact_path",

    "_build_locomo_answer_prompt",
    "_coerce_int_category",
    "_normalize_locomo_prediction",
    "_parse_prefixed_dialogue",
    "_question_is_binary",

    "_collect_artifacts",
    "_count_lines",
    "_dump_memrl_store_records",
    "_repo_root",
    "_run_subprocess",
    "_write_jsonl",

    "_LOCOMO_CATEGORY_LABELS",
    "_LOCOMO_NAMED_TASK_CATEGORY_MAP",
    "_compute_locomo_metrics",
    "_locomo_category_breakdown",
    "_locomo_paper_table_summary",
    "_locomo_reference_answer",
    "_locomo_rows_path",
    "_mean_latency_seconds",
    "_mean_metrics",
    "_parse_categories",
    "_resolve_locomo_category_filter",
    "_resolve_max_questions",
    "_write_locomo_rows",

    "_build_plugmem_env_overrides",
    "_ensure_plugmem_source_on_path",
    "_resolve_plugmem_root",

    "MemRLLoCoMoAdapter",

    "run_memrl_locomo",
]
