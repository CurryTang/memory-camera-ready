"""ALFWorld two-phase driver: 256 train episodes populate memory, 60 test episodes evaluate.

Reads:
    results/alfworld_train_test/train_pool.json   (256 train game files)
    results/alfworld_train_test/test_pool.json    (60 test game files)

Writes (per method) under $RESULTS_ROOT/<method>/:
    train_trajectories.jsonl
    train_summary.json
    train_journal.jsonl                  # longcontext_journal only
    answers_<method>_<backbone>.jsonl
    test_summary.json

Env vars:
    RESULTS_ROOT          default: results/alfworld_train_test_qwen3-32b
    RQ1_LLM_MODEL         default: Qwen/Qwen3-32B
    RQ1_LLM_URL           default: http://localhost:30000/v1
    RQ1_EMB_MODEL         default: Qwen3-Embedding-4B
    RQ1_EMB_URL           default: http://localhost:8001/v1
    RQ1_BACKBONE_TAG      default: qwen3-32b
    RQ1_METHODS           default: longcontext_journal,simplemem,hipporag,plugmem,ama_agent,memt,memrl,lightmem
    RQ1_ALFWORLD_MAX_STEPS         default: 50
    RQ1_ALFWORLD_TEMPERATURE       default: 0.4
    RQ1_ALFWORLD_MAX_TOKENS        default: 512
    RQ1_ALFWORLD_MAX_HISTORY       default: 2
    RQ1_ALFWORLD_MAX_CHAT_HISTORY  default: 15
    RQ1_ALFWORLD_MEMORY_TOP_K      default: 3
    RQ1_ALFWORLD_MAX_JOURNAL_CALLS default: 8
    RQ1_ALFWORLD_JOURNAL_TOP_K     default: 3
    AGENTMEM_REPO         default: repository root
    ALFWORLD_DATA         must be set to the dir containing json_2.1.1/
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("AGENTMEM_REPO", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from agentmem.eval.alfworld_runner.agents import create_agent              
from agentmem.eval.alfworld_runner.journal import Journal              
from agentmem.eval.alfworld_runner.memory import create_memory_backend              
from agentmem.eval.alfworld_runner.runner import (              
    AlfWorldEvalRunner,
    AlfWorldTask,
    infer_task_type_from_game_file,
)
from agentmem.eval.alfworld_runner.env import ALFWorldConfig              

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("alfworld_train_test")

DEFAULT_METHODS = (
    "longcontext_journal_notrain,longcontext_journal,"
    "simplemem,hipporag,plugmem,ama_agent,memt,memrl,lightmem"
)

METHOD_DISPATCH: dict[str, tuple[str, str | None, bool]] = {
    "longcontext_journal": ("longcontext_journal", None, False),
    "longcontext_journal_notrain": ("longcontext_journal", None, True),
    "simplemem": ("react_memory", "simplemem", False),
    "simplemem_notrain": ("react_memory", "simplemem", True),
    "hipporag": ("react_memory", "hipporag", False),
    "hipporag_notrain": ("react_memory", "hipporag", True),
    "hipporagv2": ("react_memory", "hipporag", False),
    "plugmem": ("react_memory", "plugmem", False),
    "plugmem_notrain": ("react_memory", "plugmem", True),
    "ama_agent": ("react_memory", "ama_agent", False),
    "ama_agent_notrain": ("react_memory", "ama_agent", True),
    "memt": ("react_memory", "memt", False),
    "memt_notrain": ("react_memory", "memt", True),
    "memrl": ("react_memory", "memrl", False),
    "memrl_notrain": ("react_memory", "memrl", True),
    "lightmem": ("react_memory", "lightmem", False),
    "lightmem_notrain": ("react_memory", "lightmem", True),
    "dci_lite": ("dci_journal", None, False),
    "dci_lite_notrain": ("dci_journal", None, True),
    "dci_lite_sum": ("dci_journal_sum", None, False),
    "dci_lite_sum_notrain": ("dci_journal_sum", None, True),
    "golden_rules": ("golden_rules", None, True),
    "golden_rules_notrain": ("golden_rules", None, True),
}

EXPLICIT_RL_METHODS = {
    "memrl",
    "memrl_notrain",
}

def _load_pool(path: Path, limit: int = 0) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Game-file pool not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}; got {type(data).__name__}")
    games = [str(x) for x in data]
    if limit > 0 and len(games) > limit:
        games = games[:limit]
    return games

def _resolve_game_file(game_file: str) -> Path:
    path = Path(str(game_file))
    if path.is_absolute():
        return path
    data_root = Path(os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")).expanduser()
    return data_root / "json_2.1.1" / path

def _validate_game_files(game_files: list[str], *, label: str) -> None:
    missing = [str(_resolve_game_file(game)) for game in game_files if not _resolve_game_file(game).exists()]
    if missing:
        preview = "\n  ".join(missing[:5])
        raise FileNotFoundError(
            f"{label} pool contains {len(missing)} missing ALFWorld game files. "
            "Set ALFWORLD_DATA to the directory containing json_2.1.1/ before running.\n"
            f"First missing paths:\n  {preview}"
        )

def _is_method_done(out_dir: Path, method: str, backbone: str, expected_rows: int) -> bool:
    """Return True if `answers_<method>_<backbone>.jsonl` already has expected_rows rows."""
    answers = out_dir / f"answers_{method}_{backbone}.jsonl"
    if not answers.exists() or expected_rows <= 0:
        return False
    n = sum(1 for line in answers.read_text(encoding="utf-8").splitlines() if line.strip())
    return n >= expected_rows

def _build_tasks(game_files: list[str], split_label: str) -> list[AlfWorldTask]:
    return [
        AlfWorldTask(
            game_file=str(gf),
            task_type=infer_task_type_from_game_file(gf),
            train_eval=split_label,
        )
        for gf in game_files
    ]

def _csv_env(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]

def _validate_explicit_rl_methods(methods: list[str]) -> None:
    explicit = [method for method in methods if method in EXPLICIT_RL_METHODS]
    if not explicit:
        return
    if os.environ.get("RQ1_ALLOW_UNTRAINED_EXPLICIT_RL", "") in {"1", "true", "True"}:
        logger.warning(
            "Running explicit-RL methods without holdout enforcement because "
            "RQ1_ALLOW_UNTRAINED_EXPLICIT_RL is set: %s",
            explicit,
        )
        return
    missing: list[str] = []
    if any(method.startswith("memrl") for method in explicit):
        config_path = os.environ.get("RQ1_ALFWORLD_MEMRL_CONFIG_PATH", "")
        if not config_path:
            missing.append("RQ1_ALFWORLD_MEMRL_CONFIG_PATH")
        elif not Path(config_path).exists():
            missing.append(f"existing RQ1_ALFWORLD_MEMRL_CONFIG_PATH ({config_path})")
    if missing:
        raise RuntimeError(
            "Explicit-RL ALFWorld methods require holdout-trained configuration "
            "before paper-grade execution. Missing: "
            + ", ".join(missing)
            + ". For adapter-only smoke tests, set RQ1_ALLOW_UNTRAINED_EXPLICIT_RL=1."
        )

def _run_one_method(
    method: str,
    *,
    train_tasks: list[AlfWorldTask],
    test_tasks: list[AlfWorldTask],
    results_root: Path,
    backbone_tag: str,
    llm_model: str,
    llm_base_url: str,
    embedding_model: str,
    embedding_base_url: str,
    agent_kwargs: dict[str, Any],
    max_steps: int,
    memory_top_k: int,
    memory_config_path: str,
    journal_top_k: int,
    max_journal_calls: int,
) -> dict[str, Any]:
    if method not in METHOD_DISPATCH:
        raise ValueError(f"Unknown method '{method}'. Known: {sorted(METHOD_DISPATCH)}")
    agent_type, backend_name, skip_train = METHOD_DISPATCH[method]
    effective_train = [] if skip_train else train_tasks

    out = results_root / method
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    try:
        memory_backend = None
        journal = None
        if backend_name is not None:
            config_path = ""
            if backend_name == "memrl":
                config_path = os.environ.get("RQ1_ALFWORLD_MEMRL_CONFIG_PATH", "")
            if not config_path:
                config_path = memory_config_path

            index_dir: str | None = None
            if backend_name in {"plugmem", "hipporag", "hipporagv2", "ama_agent"}:
                index_path = out / "index"
                index_path.mkdir(parents=True, exist_ok=True)
                index_dir = str(index_path)
            memory_backend = create_memory_backend(
                backend_name,
                top_k=memory_top_k,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url,
                config_path=config_path or None,
                save_dir=index_dir,
            )
        if agent_type in {"longcontext_journal", "dci_journal", "dci_journal_sum"}:
            journal = Journal(path=out / "train_journal.jsonl")

            if journal.path is not None and journal.path.exists():
                journal.path.unlink()

        agent = create_agent(
            agent_type,
            model=llm_model,
            base_url=llm_base_url,
            memory_backend=memory_backend,
            journal=journal,
            max_journal_calls_per_step=max_journal_calls,
            journal_top_k=journal_top_k,
            **agent_kwargs,
        )

        env_config = ALFWorldConfig(max_steps=max_steps, use_gym=True)
        runner = AlfWorldEvalRunner(agent=agent, env_config=env_config)

        t0 = time.time()
        summary = runner.run_train_then_test(
            train_tasks=effective_train,
            test_tasks=test_tasks,
            output_dir=out,
            method_name=method,
            backbone_tag=backbone_tag,
            llm_model=llm_model,
            journal=journal,
        )
        elapsed = time.time() - t0
        train_total = summary["train"]["total"] or 0
        train_rate = summary["train"]["rate"] if train_total else 0.0
        train_marker = "skipped" if skip_train else f"{summary['train']['success']}/{train_total} ({100.0 * train_rate:.1f}%)"
        logger.info(
            "[%s] DONE in %.1fs | train=%s | test=%d/%d (%.1f%%)",
            method,
            elapsed,
            train_marker,
            summary["test"]["success"],
            summary["test"]["total"],
            100.0 * summary["test"]["rate"],
        )
        return summary
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

def main() -> int:
    results_root = Path(os.environ.get("RESULTS_ROOT", "results/alfworld_train_test_qwen3-32b"))
    pool_root = Path("results/alfworld_train_test")
    train_limit = int(os.environ.get("RQ1_TRAIN_LIMIT", "0"))
    test_limit = int(os.environ.get("RQ1_TEST_LIMIT", "0"))
    train_pool = _load_pool(pool_root / "train_pool.json", limit=train_limit)
    test_pool = _load_pool(pool_root / "test_pool.json", limit=test_limit)
    _validate_game_files(train_pool, label="train")
    _validate_game_files(test_pool, label="test")
    logger.info(
        "Train pool: %d games (limit=%s) · Test pool: %d games (limit=%s)",
        len(train_pool), train_limit or "none", len(test_pool), test_limit or "none",
    )
    skip_if_done = os.environ.get("RQ1_SKIP_IF_DONE", "1") not in ("0", "false", "False")

    train_tasks = _build_tasks(train_pool, "train")
    test_tasks = _build_tasks(test_pool, "valid_seen")

    backbone_tag = os.environ.get("RQ1_BACKBONE_TAG", "qwen3-32b")
    llm_model = os.environ.get("RQ1_LLM_MODEL", "Qwen/Qwen3-32B")
    llm_base_url = os.environ.get("RQ1_LLM_URL", "http://localhost:30000/v1")
    embedding_model = os.environ.get("RQ1_EMB_MODEL", "Qwen3-Embedding-4B")
    embedding_base_url = os.environ.get("RQ1_EMB_URL", "http://localhost:8001/v1")

    agent_kwargs: dict[str, Any] = {
        "temperature": float(os.environ.get("RQ1_ALFWORLD_TEMPERATURE", "0.4")),
        "max_tokens": int(os.environ.get("RQ1_ALFWORLD_MAX_TOKENS", "512")),
        "max_history": int(os.environ.get("RQ1_ALFWORLD_MAX_HISTORY", "2")),
        "max_chat_history_turns": int(os.environ.get("RQ1_ALFWORLD_MAX_CHAT_HISTORY", "15")),
    }
    max_steps = int(os.environ.get("RQ1_ALFWORLD_MAX_STEPS", "50"))
    memory_top_k = int(os.environ.get("RQ1_ALFWORLD_MEMORY_TOP_K", "3"))
    memory_config_path = os.environ.get("RQ1_ALFWORLD_MEMORY_CONFIG_PATH", "")
    journal_top_k = int(os.environ.get("RQ1_ALFWORLD_JOURNAL_TOP_K", "3"))
    max_journal_calls = int(os.environ.get("RQ1_ALFWORLD_MAX_JOURNAL_CALLS", "8"))

    methods = _csv_env("RQ1_METHODS", DEFAULT_METHODS)
    _validate_explicit_rl_methods(methods)
    logger.info("Methods to run: %s", methods)
    logger.info("Backbone: %s @ %s", llm_model, llm_base_url)
    logger.info("Output root: %s", results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    status_path = results_root / "method_status.tsv"
    if not status_path.exists():
        status_path.write_text("method\tstatus\tmessage\n", encoding="utf-8")

    review_script = REPO / "scripts" / "codex_review_alfworld_method.sh"
    score_script = REPO / "scripts" / "score_alfworld_success.py"

    def _post_method_hook(method_name: str) -> None:
        """Score this method's answers and append a per-method auto-review."""
        try:
            __import__("subprocess").run(
                [str(__import__("sys").executable), str(score_script), str(results_root)],
                check=False, timeout=60,
            )
        except Exception as exc:
            logger.warning("post-method score failed for %s: %s", method_name, exc)
        if review_script.exists():
            try:
                __import__("subprocess").run(
                    ["bash", str(review_script), method_name],
                    check=False, timeout=60,
                    env={**os.environ, "RESULTS_ROOT": str(results_root)},
                )
            except Exception as exc:
                logger.warning("post-method review failed for %s: %s", method_name, exc)

    grand_summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        logger.info("=" * 60)
        logger.info("STARTING method=%s", method)
        logger.info("=" * 60)
        if skip_if_done and _is_method_done(
            results_root / method, method, backbone_tag, expected_rows=len(test_pool)
        ):
            logger.info("[%s] SKIP — answers file already has %d rows", method, len(test_pool))
            grand_summary[method] = {"skipped": True, "reason": "answers file already complete"}
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{method}\tskipped\tanswers file already complete\n")
            _post_method_hook(method)
            continue
        try:
            summary = _run_one_method(
                method,
                train_tasks=train_tasks,
                test_tasks=test_tasks,
                results_root=results_root,
                backbone_tag=backbone_tag,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url,
                agent_kwargs=agent_kwargs,
                max_steps=max_steps,
                memory_top_k=memory_top_k,
                memory_config_path=memory_config_path,
                journal_top_k=journal_top_k,
                max_journal_calls=max_journal_calls,
            )
            grand_summary[method] = summary
            with status_path.open("a", encoding="utf-8") as handle:
                rate = summary["test"]["rate"]
                total = summary["test"]["total"]
                success = summary["test"]["success"]
                handle.write(f"{method}\tdone\ttest={success}/{total} rate={rate:.6f}\n")
            _post_method_hook(method)
        except Exception as exc:
            logger.error("Method %s FAILED: %s", method, exc)
            traceback.print_exc()
            grand_summary[method] = {"error": str(exc)}
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{method}\tfailed\t{str(exc).replace(chr(9), ' ')}\n")

    (results_root / "grand_summary.json").write_text(
        json.dumps(grand_summary, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %s", results_root / "grand_summary.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
