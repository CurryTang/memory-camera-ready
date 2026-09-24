"""ALFWorld evaluation runner for prompt-based ReAct and Reflexion agents."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from agentmem.eval.alfworld_runner.agents import BaseAlfWorldAgent
from agentmem.eval.alfworld_runner.journal import Journal
from agentmem.eval.alfworld_runner.env import ALFWorldConfig, ALFWorldEnv

logger = logging.getLogger(__name__)

@dataclass
class AlfWorldTask:
    game_file: str
    task_desc: str = ""
    task_type: str = ""
    train_eval: str = ""
    use_gym: bool = True

    def as_task_data(self) -> dict[str, Any]:
        data = {
            "game_file": self.game_file,
            "task_desc": self.task_desc,
            "task_type": self.task_type,
            "use_gym": self.use_gym,
        }
        if self.train_eval:
            data["train_eval"] = self.train_eval
        return data

@dataclass
class TaskResult:
    episode_idx: int
    game_file: str
    task_type: str
    task_desc: str
    success: bool
    reward: float
    num_steps: int
    elapsed_seconds: float
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    num_trials: int = 1
    trials: list[dict[str, Any]] = field(default_factory=list)
    had_memory: bool = False
    error: Optional[str] = None

@dataclass
class AlfWorldEvalReport:
    num_tasks: int
    num_success: int
    success_rate: float
    avg_steps: float
    avg_time_seconds: float
    tasks: list[TaskResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"ALFWorld Eval: {self.num_success}/{self.num_tasks} success "
            f"({self.success_rate:.1%}), avg_steps={self.avg_steps:.1f}"
        )

def load_alfworld_tasks(
    split: str = "eval_in_distribution",
    limit: Optional[int] = None,
    seed: int = 42,
) -> list[AlfWorldTask]:
    """Load ALFWorld game files from the installed dataset."""
    try:
        import alfworld.agents.environment as _              
    except ImportError as exc:
        raise ImportError(
            "alfworld is required for ALFWorld evaluation. Install with "
            "`pip install alfworld && alfworld-download`."
        ) from exc

    alfworld_data = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    json_dir = Path(alfworld_data) / "json_2.1.1"
    if not json_dir.exists():
        raise FileNotFoundError(
            f"ALFWorld data not found at {json_dir}. Set ALFWORLD_DATA or run alfworld-download."
        )

    split_dir_name = _resolve_split_dir_name(split)
    split_dir = json_dir / split_dir_name
    split_file = split_dir / "game_files.json"
    if split_file.exists():
        game_files = json.loads(split_file.read_text(encoding="utf-8"))
    else:
        game_files = sorted(
            str(path.relative_to(json_dir))
            for path in split_dir.rglob("*.tw-pddl")
        )
        if not game_files:
            game_files = sorted(
                str(path.relative_to(json_dir))
                for path in split_dir.rglob("game.tw")
            )

    tasks = [
        AlfWorldTask(
            game_file=str(game_file),
            task_type=infer_task_type_from_game_file(str(game_file)),
            train_eval=split,
        )
        for game_file in game_files
    ]
    if limit is not None and len(tasks) > limit:
        rng = random.Random(seed)
        tasks = rng.sample(tasks, limit)
    return tasks

class AlfWorldEvalRunner:
    """Runs sequential ALFWorld evaluation so memory can accumulate across episodes."""

    def __init__(
        self,
        agent: BaseAlfWorldAgent,
        env_config: Optional[ALFWorldConfig] = None,
        checkpoint_dir: Optional[str | Path] = None,
    ) -> None:
        self.agent = agent
        self.env_config = env_config or ALFWorldConfig(use_gym=True)
        self.env_config.use_gym = True
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._completed: dict[str, TaskResult] = {}

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._load_checkpoint()

    def run_task(
        self,
        task: AlfWorldTask,
        episode_idx: int,
        *,
        shared_env: Optional[ALFWorldEnv] = None,
        game_pool: Optional[Sequence[str]] = None,
    ) -> TaskResult:
        task_data = task.as_task_data()
        overall_t0 = time.time()
        agent_begin_task = getattr(self.agent, "begin_task", None)
        if callable(agent_begin_task):
            agent_begin_task(task_data)
        final_result: Optional[TaskResult] = None
        trial_results: list[dict[str, Any]] = []
        max_trials = max(1, int(getattr(self.agent, "max_trials", 1)))
        max_episode_steps = max(1, int(getattr(self.env_config, "max_steps", 30)))
        local_env: Optional[ALFWorldEnv] = None
        if shared_env is None:
            local_env = ALFWorldEnv(self.env_config)

        for trial_idx in range(max_trials):
            env = shared_env if shared_env is not None else local_env
            if env is None:
                raise RuntimeError("ALFWorld environment was not initialized")
            trial_t0 = time.time()
            trial_task_data = dict(task_data)
            trial_task_data["trial_idx"] = trial_idx
            if shared_env is not None and game_pool is not None:
                trial_task_data["game_files"] = list(game_pool)
            try:
                observation = env.reset(trial_task_data)
                task_desc = task.task_desc or extract_task_desc(observation)
                trial_task_data["task_desc"] = task_desc
                trial_task_data["task_type"] = (
                    task.task_type
                    or getattr(env, "get_task_type", lambda: "")()
                    or infer_task_type_from_game_file(task.game_file)
                )
                self.agent.reset(observation, trial_task_data)

                while not env.done and env.step_count < max_episode_steps:
                    if bool(getattr(self.agent, "should_stop_episode", lambda: False)()):
                        break
                    action = self.agent.act(observation, env.get_admissible_commands())
                    step_result = env.step(action)
                    observation = step_result.observation
                    self.agent.observe(action, step_result.observation)

                    if step_result.done or bool(getattr(self.agent, "should_stop_episode", lambda: False)()):
                        break

                success = env.get_success()
                final_result = TaskResult(
                    episode_idx=episode_idx,
                    game_file=task.game_file,
                    task_type=trial_task_data["task_type"],
                    task_desc=task_desc,
                    success=success,
                    reward=env.total_reward if env.total_reward else (1.0 if success else 0.0),
                    num_steps=env.step_count,
                    elapsed_seconds=time.time() - overall_t0,
                    trajectory=env.get_trajectory(),
                    agent_trace=self.agent.get_debug_trace(),
                    num_trials=trial_idx + 1,
                    trials=[],
                    had_memory=bool(getattr(self.agent, "_memory_prompt", "")),
                )
                trial_results.append(
                    {
                        "trial_idx": trial_idx,
                        "success": success,
                        "num_steps": env.step_count,
                        "elapsed_seconds": time.time() - trial_t0,
                        "error": None,
                    }
                )
                agent_finish_trial = getattr(self.agent, "finish_trial", None)
                if callable(agent_finish_trial):
                    agent_finish_trial(
                        task_info=trial_task_data,
                        trajectory=final_result.trajectory,
                        success=success,
                        trial_idx=trial_idx,
                    )
                if success:
                    break
            except Exception as exc:
                logger.exception("ALFWorld task failed for %s (trial %d)", task.game_file, trial_idx)
                final_result = TaskResult(
                    episode_idx=episode_idx,
                    game_file=task.game_file,
                    task_type=task.task_type,
                    task_desc=task.task_desc,
                    success=False,
                    reward=0.0,
                    num_steps=env.step_count,
                    elapsed_seconds=time.time() - overall_t0,
                    trajectory=env.get_trajectory(),
                    agent_trace=self.agent.get_debug_trace(),
                    num_trials=trial_idx + 1,
                    trials=[],
                    had_memory=bool(getattr(self.agent, "_memory_prompt", "")),
                    error=str(exc),
                )
                trial_results.append(
                    {
                        "trial_idx": trial_idx,
                        "success": False,
                        "num_steps": env.step_count,
                        "elapsed_seconds": time.time() - trial_t0,
                        "error": str(exc),
                    }
                )
                agent_finish_trial = getattr(self.agent, "finish_trial", None)
                if callable(agent_finish_trial):
                    agent_finish_trial(
                        task_info=trial_task_data,
                        trajectory=env.get_trajectory(),
                        success=False,
                        trial_idx=trial_idx,
                        error=str(exc),
                    )
            if final_result is not None and final_result.success:
                break

        if local_env is not None:
            local_env.close()

        if final_result is None:
            final_result = TaskResult(
                episode_idx=episode_idx,
                game_file=task.game_file,
                task_type=task.task_type,
                task_desc=task.task_desc,
                success=False,
                reward=0.0,
                num_steps=0,
                elapsed_seconds=time.time() - overall_t0,
                num_trials=len(trial_results),
                trials=[],
                had_memory=False,
                error="task did not produce a result",
            )
        final_result.trials = trial_results
        final_result.num_trials = len(trial_results) or 1
        agent_finish_episode = getattr(self.agent, "finish_episode", None)
        if callable(agent_finish_episode):
            agent_finish_episode(
                episode_idx=episode_idx,
                task_info={
                    "task_desc": final_result.task_desc,
                    "task_type": final_result.task_type,
                },
                trajectory=final_result.trajectory,
                success=final_result.success,
            )
        return final_result

    def run(self, tasks: Sequence[AlfWorldTask], skip_completed: bool = True) -> AlfWorldEvalReport:
        results: list[TaskResult] = []
        shared_env = ALFWorldEnv(self.env_config)
        game_pool = [task.game_file for task in tasks]
        try:
            for episode_idx, task in enumerate(tasks):
                key = task.game_file
                if skip_completed and key in self._completed:
                    results.append(self._completed[key])
                    continue

                result = self.run_task(
                    task,
                    episode_idx=episode_idx,
                    shared_env=shared_env,
                    game_pool=game_pool,
                )
                results.append(result)
                self._completed[key] = result
                self._save_result(result)
                logger.info(
                    "Episode %d/%d success=%s steps=%d task_type=%s",
                    episode_idx + 1,
                    len(tasks),
                    result.success,
                    result.num_steps,
                    result.task_type,
                )
        finally:
            shared_env.close()

        num_tasks = len(results)
        num_success = sum(1 for result in results if result.success)
        avg_steps = sum(result.num_steps for result in results) / max(num_tasks, 1)
        avg_time = sum(result.elapsed_seconds for result in results) / max(num_tasks, 1)
        return AlfWorldEvalReport(
            num_tasks=num_tasks,
            num_success=num_success,
            success_rate=num_success / max(num_tasks, 1),
            avg_steps=avg_steps,
            avg_time_seconds=avg_time,
            tasks=results,
        )

    def _load_checkpoint(self) -> None:
        if self.checkpoint_dir is None:
            return
        ckpt_file = self.checkpoint_dir / "alfworld_results.jsonl"
        if not ckpt_file.exists():
            return
        for line in ckpt_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            result = TaskResult(
                episode_idx=int(data["episode_idx"]),
                game_file=str(data["game_file"]),
                task_type=str(data.get("task_type", "")),
                task_desc=str(data.get("task_desc", "")),
                success=bool(data["success"]),
                reward=float(data.get("reward", 0.0)),
                num_steps=int(data["num_steps"]),
                elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
                error=data.get("error"),
            )
            self._completed[result.game_file] = result

    def _save_result(self, result: TaskResult) -> None:
        if self.checkpoint_dir is None:
            return
        ckpt_file = self.checkpoint_dir / "alfworld_results.jsonl"
        with ckpt_file.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "episode_idx": result.episode_idx,
                        "game_file": result.game_file,
                        "task_type": result.task_type,
                        "task_desc": result.task_desc,
                        "success": result.success,
                        "reward": result.reward,
                        "num_steps": result.num_steps,
                        "elapsed_seconds": result.elapsed_seconds,
                        "error": result.error,
                    }
                )
                + "\n"
            )

    def run_train_then_test(
        self,
        *,
        train_tasks: Sequence[AlfWorldTask],
        test_tasks: Sequence[AlfWorldTask],
        output_dir: str | Path,
        method_name: str,
        backbone_tag: str,
        llm_model: Optional[str] = None,
        journal: Optional[Journal] = None,
        write_train_trajectories: bool = True,
    ) -> dict[str, Any]:
        """Run ALFWorld in two phases: train (memory-population) then test (paper-grade).

        - Phase 1: iterate `train_tasks`. After each episode, the agent's
          `finish_episode` populates whatever memory backend was wired in. If a
          `Journal` is supplied (long-context baseline), the runner also appends
          the trajectory to it.
        - Phase 2: iterate `test_tasks` against the SAME agent + memory + journal.
          The trained memory is kept; new test trajectories are also appended so
          memory continues to grow within-eval.

        Outputs land under `output_dir`:
          - `train_trajectories.jsonl`  (phase=train rows; diagnostic)
          - `train_summary.json`        (per-task-type success during training)
          - `answers_<method>_<backbone>.jsonl`  (phase=test rows; paper-grade)
          - `test_summary.json`
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        train_path = out / "train_trajectories.jsonl"
        if write_train_trajectories and train_path.exists():
            train_path.unlink()

        test_path = out / f"answers_{method_name}_{backbone_tag}.jsonl"
        if test_path.exists():
            test_path.unlink()

        train_results: list[TaskResult] = []
        shared_env = ALFWorldEnv(self.env_config)
        game_pool = [task.game_file for task in [*train_tasks, *test_tasks]]
        try:
            for episode_idx, task in enumerate(train_tasks):
                result = self.run_task(
                    task,
                    episode_idx=episode_idx,
                    shared_env=shared_env,
                    game_pool=game_pool,
                )
                train_results.append(result)
                if write_train_trajectories:
                    self._write_phase_row(
                        path=train_path,
                        result=result,
                        phase="train",
                        method=method_name,
                        backbone=backbone_tag,
                        llm_model=llm_model,
                    )
                if journal is not None:
                    journal.add(
                        episode_idx=result.episode_idx,
                        phase="train",
                        task_desc=result.task_desc,
                        task_type=result.task_type,
                        success=result.success,
                        trajectory=result.trajectory,
                    )
                logger.info(
                    "[train %d/%d] success=%s steps=%d task_type=%s",
                    episode_idx + 1,
                    len(train_tasks),
                    result.success,
                    result.num_steps,
                    result.task_type,
                )
        finally:
            shared_env.close()

        train_summary = _summarize_phase(train_results, phase="train")
        (out / "train_summary.json").write_text(
            json.dumps(train_summary, indent=2) + "\n", encoding="utf-8"
        )

        test_results: list[TaskResult] = []
        test_offset = len(train_tasks)
        shared_env = ALFWorldEnv(self.env_config)
        try:
            for test_idx, task in enumerate(test_tasks):
                episode_idx = test_offset + test_idx
                result = self.run_task(
                    task,
                    episode_idx=episode_idx,
                    shared_env=shared_env,
                    game_pool=game_pool,
                )
                test_results.append(result)
                self._write_phase_row(
                    path=test_path,
                    result=result,
                    phase="test",
                    method=method_name,
                    backbone=backbone_tag,
                    llm_model=llm_model,
                )
                if journal is not None:
                    journal.add(
                        episode_idx=result.episode_idx,
                        phase="test",
                        task_desc=result.task_desc,
                        task_type=result.task_type,
                        success=result.success,
                        trajectory=result.trajectory,
                    )
                logger.info(
                    "[test %d/%d] success=%s steps=%d task_type=%s",
                    test_idx + 1,
                    len(test_tasks),
                    result.success,
                    result.num_steps,
                    result.task_type,
                )
        finally:
            shared_env.close()

        test_summary = _summarize_phase(test_results, phase="test")
        (out / "test_summary.json").write_text(
            json.dumps(test_summary, indent=2) + "\n", encoding="utf-8"
        )

        return {"train": train_summary, "test": test_summary}

    @staticmethod
    def _write_phase_row(
        *,
        path: Path,
        result: TaskResult,
        phase: str,
        method: str,
        backbone: str,
        llm_model: Optional[str],
    ) -> None:
        row = {
            "episode_idx": result.episode_idx,
            "phase": phase,
            "method": method,
            "backbone": backbone,
            "llm_model": llm_model or backbone,
            "game_file": result.game_file,
            "task_type": result.task_type,
            "task_desc": result.task_desc,
            "success": result.success,
            "reward": result.reward,
            "num_steps": result.num_steps,
            "elapsed_seconds": result.elapsed_seconds,
            "wall_time": result.elapsed_seconds,
            "had_memory": result.had_memory,
            "num_trials": result.num_trials,
            "error": result.error,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def _summarize_phase(results: Sequence[TaskResult], *, phase: str) -> dict[str, Any]:
    n = len(results)
    n_success = sum(1 for r in results if r.success)
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})
    for r in results:
        bucket = by_type[r.task_type or "unknown"]
        bucket["total"] += 1
        if r.success:
            bucket["success"] += 1

    avg_steps = sum(r.num_steps for r in results) / max(n, 1)
    avg_time = sum(r.elapsed_seconds for r in results) / max(n, 1)
    return {
        "phase": phase,
        "total": n,
        "success": n_success,
        "rate": (n_success / n) if n else 0.0,
        "avg_steps": avg_steps,
        "avg_elapsed_seconds": avg_time,
        "by_task_type": {
            tt: {
                "success": stats["success"],
                "total": stats["total"],
                "rate": stats["success"] / stats["total"] if stats["total"] else 0.0,
            }
            for tt, stats in sorted(by_type.items())
        },
    }

def infer_task_type_from_game_file(game_file: str) -> str:
    normalized = str(game_file or "").replace("\\", "/")
    for key, value in (
        ("pick_and_place", "pick_and_place"),
        ("pick_clean_then_place", "pick_clean_then_place"),
        ("pick_heat_then_place", "pick_heat_then_place"),
        ("pick_cool_then_place", "pick_cool_then_place"),
        ("look_at_obj", "look_at_obj"),
        ("pick_two_obj", "pick_two_obj"),
        ("pick_two_obj_and_place", "pick_two_obj"),
    ):
        if key in normalized:
            return value
    return ""

def extract_task_desc(observation: str) -> str:
    match = re.search(r"Your task is to:\s*(.*?)(?:[\n\"']|$)", observation, re.DOTALL)
    if match:
        return match.group(1).strip().rstrip(".")
    return observation.split("\n")[0][:200]

def _resolve_split_dir_name(split: str) -> str:
    mapping = {
        "eval_out_of_distribution": "valid_unseen",
        "eval_in_distribution": "valid_seen",
        "valid_unseen": "valid_unseen",
        "valid_seen": "valid_seen",
        "train": "train",
    }
    return mapping.get(split, split)

def run_alfworld_eval(
    method_name: str,
    *,
    max_episodes: int = 10,
    max_steps: int = 30,
    max_trials: int = 1,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    embedding_model: str | None = None,
    embedding_base_url: str | None = None,
    embedding_api_key: str | None = None,
    output_dir: str | None = None,
    split: str = "eval_in_distribution",
) -> dict[str, Any]:
    """Convenience wrapper: run AlfWorldEvalRunner for one (method, split) cell.

    Writes ``answers_<method>_<backbone>.jsonl`` + ``summary.json`` to
    ``output_dir`` so the strict QA judge can pick them up. Methods without
    a memory backend (e.g. ``longcontext``) use a vanilla ReAct agent.
    """
    from agentmem.eval.alfworld_runner.agents import create_agent
    from agentmem.eval.alfworld_runner.memory import create_memory_backend

    out = Path(output_dir) if output_dir else Path("results/alfworld")
    out.mkdir(parents=True, exist_ok=True)
    backbone_tag = (str(llm_model or "qwen3-32b").lower().replace("/", "-"))
    summary_path = out / "summary.json"
    ans_path = out / f"answers_{method_name}_{backbone_tag}.jsonl"
    results: list[TaskResult] = []

    def write_outputs(error: str | None = None) -> dict[str, Any]:
        with ans_path.open("w", encoding="utf-8") as fh:
            for r in results:
                payload = {
                    "episode_idx": r.episode_idx,
                    "game_file": r.game_file,
                    "task_desc": r.task_desc,
                    "task_type": r.task_type,
                    "success": bool(r.success),
                    "num_steps": r.num_steps,
                    "trajectory": r.trajectory,
                    "error": r.error,
                }
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        n_success = sum(1 for r in results if r.success)
        summary = {
            "method": method_name,
            "backbone": llm_model,
            "num_tasks": len(results),
            "n_episodes": len(results),
            "num_success": n_success,
            "n_success": n_success,
            "success_rate": float(n_success / max(1, len(results))),
            "max_episodes": int(max_episodes),
            "max_steps": int(max_steps),
            "max_trials": int(max_trials),
            "answers_path": str(ans_path),
        }
        if error:
            summary["error"] = error
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    try:
        needs_memory = method_name not in {"longcontext", "vanilla", "react", "classic_react", "reflexion"}
        memory_backend = None
        if needs_memory:
            memory_backend = create_memory_backend(
                method_name,
                top_k=3,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url,
                embedding_api_key=embedding_api_key,
                save_dir=str(out / "method_state"),
            )
        agent_kind = "react_memory" if needs_memory else "react"
        agent = create_agent(
            agent_kind,
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key or "EMPTY",
            memory_backend=memory_backend,
            max_trials=max_trials,
        )

        tasks = load_alfworld_tasks(split=split, limit=max_episodes)
        env_config = ALFWorldConfig(
            max_steps=max_steps,
            use_gym=True,
            alfworld_data_path=os.environ.get("ALFWORLD_DATA", ""),
        )
        runner = AlfWorldEvalRunner(agent=agent, env_config=env_config, checkpoint_dir=str(out))
        shared_env = ALFWorldEnv(env_config)
        game_pool = [task.game_file for task in tasks]
        try:
            for idx, task in enumerate(tasks):
                results.append(
                    runner.run_task(
                        task,
                        idx,
                        shared_env=shared_env,
                        game_pool=game_pool,
                    )
                )
        finally:
            shared_env.close()
        return write_outputs()
    except (FileNotFoundError, ImportError) as exc:
        write_outputs(error=str(exc))
        raise RuntimeError(f"Fatal ALFWorld setup error: {exc}") from exc
    except Exception as exc:
        logger.exception("ALFWorld cell failed for method %s", method_name)
        return write_outputs(error=str(exc))
