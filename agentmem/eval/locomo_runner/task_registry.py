"""Task registry: normalizes task names, maps configs to tasks, dispatches execution."""

from __future__ import annotations

from typing import Any, Optional

_CONFIG_TO_TASK = {
    "C1": "c1-bm25-locomo",
    "C2": "c2-keyword-locomo",
    "C3": "c3-dense-locomo",
    "C4": "c4-compressed-locomo",
    "C5": "c5-fusion-locomo",
    "C6": "c6-causal-locomo",
    "C7": "c7-concept-locomo",
    "C8": "c8-kg-locomo",
    "C9": "c9-colbert-locomo",
    "LONGCONTEXT": "longcontext-locomo",
    "HIPPORAGV2": "hipporagv2-locomo",
}

def _normalize_task_list(raw: str) -> list[str]:
    """Parse and normalize a comma-separated task string into canonical task names."""
    raw = (raw or "").strip().lower()
    if not raw:
        return ["amem-locomo"]

    locomo_paper2601_methods = [
        "simplemem-locomo", "mem0-locomo", "amem-locomo", "plugmem-locomo",
    ]
    locomo_all_methods = [
        "simplemem-locomo", "mem0-locomo", "memrl-locomo", "memt-locomo",
        "remem-locomo", "amem-locomo", "plugmem-locomo",
    ]
    locomo_baselines = [
        "c1-bm25-locomo", "c2-keyword-locomo", "c3-dense-locomo",
        "c4-compressed-locomo", "c5-fusion-locomo",
        "c6-causal-locomo", "c7-concept-locomo", "c8-kg-locomo",
        "c9-colbert-locomo",
    ]
    group_aliases = {
        "locomo-all-methods": locomo_all_methods,
        "locomo-4systems": locomo_paper2601_methods,
        "paper2601-locomo": locomo_paper2601_methods,
        "locomo-baselines": locomo_baselines,
        "locomo-all-configs": locomo_baselines,
        "sweep-locomo": locomo_baselines,
    }

    if raw in group_aliases:
        return list(group_aliases[raw])

    if raw == "all":
        return [
            "simplemem-locomo", "mem0-locomo", "memrl-locomo", "memt-locomo",
            "remem-locomo", "amem-locomo", "plugmem-locomo",
            "plugmem-amabench", "plugmem-hotpotqa",
            "plugmem-webarena",
        ]

    tasks = [item.strip() for item in raw.split(",") if item.strip()]
    resolved_tasks: list[str] = []
    for task in tasks:
        if task in group_aliases:
            resolved_tasks.extend(group_aliases[task])
            continue
        if task == "locomo-amem":
            resolved_tasks.append("amem-locomo")
        elif task in {"simple-mem-locomo", "simple-mem", "simplemem"}:
            resolved_tasks.append("simplemem-locomo")
        elif task in {"locomo-mem0", "mem0"}:
            resolved_tasks.append("mem0-locomo")
        elif task in {"locomo-memrl", "memrl"}:
            resolved_tasks.append("memrl-locomo")
        elif task in {"locomo-memt", "memt", "mem-t"}:
            resolved_tasks.append("memt-locomo")
        elif task in {"amabench-memt", "memt-amabench-text2sql"}:
            resolved_tasks.append("memt-amabench")
        elif task in {"locomo-remem", "remem", "re-mem"}:
            resolved_tasks.append("remem-locomo")
        else:
            resolved_tasks.append(task)

    seen: set[str] = set()
    deduped: list[str] = []
    for task in resolved_tasks:
        if task in seen:
            continue
        seen.add(task)
        deduped.append(task)

    valid = {
        "simplemem-locomo", "mem0-locomo", "memrl-locomo", "memt-locomo",
        "remem-locomo", "amem-locomo", "plugmem-locomo",
        "plugmem-amabench", "plugmem-hotpotqa",
        "plugmem-webarena",
        "c1-bm25-locomo", "c2-keyword-locomo", "c3-dense-locomo",
        "c4-compressed-locomo", "c5-fusion-locomo",
        "c6-causal-locomo", "c7-concept-locomo", "c8-kg-locomo",
        "c9-colbert-locomo",
        "c1-bm25-amabench", "c2-keyword-amabench",
        "c6-causalgraph-amabench", "ama-agent-amabench", "memt-amabench",
        "longcontext-locomo", "hipporagv2-locomo",
    }

    unknown = [task for task in deduped if task not in valid]
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}. Valid tasks: {sorted(valid)}")

    return deduped
