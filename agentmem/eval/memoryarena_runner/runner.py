"""Main MemoryArena evaluation loop.

External shape:
    run_memoryarena(config, method_name, *, llm_model, llm_base_url, ...) -> dict
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from agentmem.eval.memoryarena_runner import judges, loader, prompts
from agentmem.methods.multi_session import MultiSessionMemory, SessionFeedback

def _run_task(
    *,
    task: dict[str, Any],
    config: str,
    memory: MultiSessionMemory,
    agent_fn: Callable[[str], tuple[str, str]],
    llm_judge_fn: Callable[..., dict[str, Any]] | None,
    environment_fn: Callable[[dict[str, Any], int, str], str] | None = None,
    environment_factory: Callable[[dict[str, Any], int, str], Any] | None = None,
    agent_env_fn: Callable[[str, Any], tuple[str, str]] | None = None,
) -> dict[str, Any]:
    task_id = str(task.get("id", task.get("_hf_index", "?")))
    memory.reset(task_id, schema_hint=config)
    sessions: list[dict[str, Any]] = []
    judge_kind = judges.judge_for(config)
    questions = [str(q) for q in (task.get("questions") or [])]
    overall_question = questions[-1] if questions else ""

    for i, question, gold, background in loader.iter_sessions(task):
        memory_context = memory.retrieve(question, session_id=i)
        prompt = prompts.build_prompt(
            question=question,
            overall_question=overall_question,
            background=background,
            memory_context=memory_context,
            domain=config,
        )
        env_obj = environment_factory(task, i, question) if environment_factory is not None else None
        if env_obj is not None and hasattr(env_obj, "instructions"):
            environment_observations = str(env_obj.instructions())
        else:
            environment_observations = (
                environment_fn(task, i, question) if environment_fn is not None else ""
            )
        if environment_observations.strip():
            prompt = (
                f"{prompt}\n\n"
                "External environment/tool observations for this session:\n"
                f"{environment_observations.strip()}\n\n"
                "Use these observations as tool results. Do not treat them as memory records from previous sessions."
            )

        t0 = time.time()
        if agent_env_fn is not None and env_obj is not None:
            prediction, trace = agent_env_fn(prompt, env_obj)
        else:
            prediction, trace = agent_fn(prompt)
        latency = time.time() - t0

        if env_obj is not None and hasattr(env_obj, "judge_prediction"):
            verdict = env_obj.judge_prediction(prediction, gold)
        elif judge_kind == "asin":
            verdict = judges.asin_judge(prediction, gold)
        elif judge_kind == "progressive_search":
            verdict = judges.progressive_search_judge(prediction, gold)
        elif judge_kind == "leaf_recall":
            verdict = judges.leaf_recall_judge(prediction, gold)
        elif judge_kind == "llm":
            if llm_judge_fn is None:
                verdict = judges.substring_judge(prediction, gold)
            else:
                verdict = judges.llm_strict_judge(
                    prediction, gold, question, judge_fn=llm_judge_fn
                )
        else:
            verdict = judges.substring_judge(prediction, gold)

        fb = SessionFeedback(
            session_id=i,
            question=question,
            prediction=prediction,
            correct=bool(verdict.get("correct")),
            judge_signal={
                k: v
                for k, v in verdict.items()
                if k not in {"gold_excerpt", "gold_asin", "gold_exact"}
            },
            trace=trace,
            observations=[],
        )
        memory.update(fb)

        sessions.append({
            "session_id": i,
            "question": question,
            "prediction": prediction,
            "trace": trace,
            "correct": fb.correct,
            "judge_signal": fb.judge_signal,
            "environment_observations": environment_observations,
            "latency_seconds": latency,
            "score": verdict.get("score", 1.0 if fb.correct else 0.0),
        })

    n = len(sessions)
    n_correct = sum(1 for s in sessions if s["correct"])
    progress_score = (n_correct / n) if n else 0.0
    soft_progress_score = (
        sum(float(s.get("score", 1.0 if s.get("correct") else 0.0)) for s in sessions) / n
        if n
        else 0.0
    )
    success = _task_success(config, sessions)
    return {
        "task_id": task_id,
        "config": config,
        "num_subtasks": n,
        "num_correct": n_correct,
        "process_score": progress_score,
        "progress_score": progress_score,
        "soft_progress_score": soft_progress_score,
        "success": success,
        "sessions": sessions,
    }

def _task_success(config: str, sessions: list[dict[str, Any]]) -> int:
    """MemoryArena task-level SR protocol.

    The paper defines Progressive Web Search success by the concluding/final
    subtask, while shopping and travel require every session in the final
    bundle/plan to satisfy its checked constraints.
    """
    if not sessions:
        return 0
    if config == "progressive_search":
        return int(bool(sessions[-1].get("correct")))
    return int(all(bool(s.get("correct")) for s in sessions))

def run_memoryarena(
    *,
    config: str,
    memory: MultiSessionMemory,
    agent_fn: Callable[[str], tuple[str, str]],
    llm_judge_fn: Callable[..., dict[str, Any]] | None = None,
    limit: int | None = None,
    use_manifest: bool = True,
    source: str = "auto",
    progress_path: str | None = None,
    resume: bool = True,
    environment_fn: Callable[[dict[str, Any], int, str], str] | None = None,
    environment_factory: Callable[[dict[str, Any], int, str], Any] | None = None,
    agent_env_fn: Callable[[str, Any], tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Run one MemoryArena config end-to-end.

    Parameters
    ----------
    config : one of memoryarena_runner.loader.CONFIGS
    memory : a MultiSessionMemory adapter (e.g. LongContextAdapter)
    agent_fn : callable prompt -> (prediction, trace). Plug in any LLM here.
    llm_judge_fn : optional strict-judge callable for non-shopping configs.
    limit : truncate the manifest to this many tasks.
    use_manifest : if True, use the checked-in manifest; else full split.
    source : "auto", "local", or "hf" dataset source.
    progress_path : optional JSONL file appended after each completed task.
    resume : if True, load ``progress_path`` and skip completed task ids.
    environment_fn : optional callable ``(task, session_id, question) -> str``
        whose returned tool observations are appended to the session prompt.
    """
    task_results: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    progress_file = Path(progress_path) if progress_path else None
    if progress_file and resume and progress_file.exists():
        for line in progress_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_results.append(row)
            completed_ids.add(str(row.get("task_id")))
    if progress_file:
        progress_file.parent.mkdir(parents=True, exist_ok=True)

    for task in loader.load_tasks(
        config,
        limit=limit,
        use_manifest=use_manifest,
        source=source,
    ):
        task_id = str(task.get("id", task.get("_hf_index", "?")))
        if task_id in completed_ids:
            print(f"[MemoryArena] skip completed task_id={task_id}", flush=True)
            continue
        result = _run_task(
            task=task,
            config=config,
            memory=memory,
            agent_fn=agent_fn,
            llm_judge_fn=llm_judge_fn,
            environment_fn=environment_fn,
            environment_factory=environment_factory,
            agent_env_fn=agent_env_fn,
        )
        task_results.append(result)
        completed_ids.add(str(result.get("task_id")))
        if progress_file:
            with progress_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"[MemoryArena] completed {len(task_results)}/{limit or '?'} "
            f"task_id={result.get('task_id')} "
            f"PS={result.get('progress_score', 0):.3f} "
            f"sPS={result.get('soft_progress_score', 0):.3f} "
            f"SR={result.get('success', 0)}",
            flush=True,
        )
    n = len(task_results)
    mean_ps = sum(r["progress_score"] for r in task_results) / n if n else 0.0
    mean_sps = sum(r["soft_progress_score"] for r in task_results) / n if n else 0.0
    mean_sr = sum(r["success"] for r in task_results) / n if n else 0.0
    return {
        "benchmark": "memoryarena",
        "config": config,
        "dataset_source": source,
        "evaluation_protocol": {
            "process_score": "fraction of correct subtasks within each multi-session task",
            "progress_score": "alias of process_score for MemoryArena reporting",
            "soft_progress_score": (
                "per-session judge score. For paper-facing group_travel_planner, use the "
                "corrected requested-slot constraint sPS produced by "
                "scripts/judge_memoryarena_group_travel_sps.py instead of the legacy "
                "leaf-recall diagnostic in older outputs"
            ),
            "success_rate": "progressive_search uses final-session correctness; shopping/travel require all sessions correct",
            "memory_schedule": "retrieve once at session start, update once at session end",
            "environment": "optional per-session reconstructed environment/tool observations appended after prompt construction",
            "gold_leakage": "gold answers are used only by the judge, never passed to memory.update",
        },
        "num_tasks": n,
        "process_score_mean": mean_ps,
        "progress_score_mean": mean_ps,
        "soft_progress_score_mean": mean_sps,
        "success_rate_mean": mean_sr,
        "results": task_results,
    }
