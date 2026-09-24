from agentmem.eval.locomo_table import (
    LOCOMO_CATEGORY_LABELS,
    bleu1,
    bleu2,
    bleu3,
    bleu4,
    load_rows_jsonl,
    paper_token_f1,
    per_row_locomo_metrics,
    summarize_locomo_rows,
)
from agentmem.eval.metrics import compute_metrics, exact_match, rouge_l_f1, token_f1
from agentmem.eval.resource_metrics import (
    estimate_text_tokens,
    estimate_trajectory_tokens,
    summarize_resource_usage,
    summarize_resource_usage_by_method,
)
from agentmem.eval.locomo_runner.runners import run_memrl_locomo
try:
    from agentmem.eval.alfworld_runner import (
        AlfWorldEvalReport,
        AlfWorldEvalRunner,
        AlfWorldTask,
        create_agent as create_alfworld_agent,
        create_memory_backend as create_alfworld_memory_backend,
        load_alfworld_tasks,
    )
    _HAS_ALFWORLD = True
except Exception:
    _HAS_ALFWORLD = False

__all__ = [
    "compute_metrics",
    "exact_match",
    "token_f1",
    "rouge_l_f1",
    "estimate_text_tokens",
    "estimate_trajectory_tokens",
    "summarize_resource_usage",
    "summarize_resource_usage_by_method",
    "run_memrl_hotpotqa",
    "run_memrl_locomo",
    "bleu1",
    "bleu2",
    "bleu3",
    "bleu4",
    "paper_token_f1",
    "per_row_locomo_metrics",
    "summarize_locomo_rows",
    "load_rows_jsonl",
    "LOCOMO_CATEGORY_LABELS",
]

if _HAS_ALFWORLD:
    __all__[:0] = [
        "AlfWorldEvalRunner",
        "AlfWorldEvalReport",
        "AlfWorldTask",
        "create_alfworld_agent",
        "create_alfworld_memory_backend",
        "load_alfworld_tasks",
    ]

def __getattr__(name: str):
    if name in {"run_memrl_hotpotqa"}:
        from agentmem.eval.hotpotqa_mem_methods import run_memrl_hotpotqa
        globals()["run_memrl_hotpotqa"] = run_memrl_hotpotqa
        return run_memrl_hotpotqa
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
