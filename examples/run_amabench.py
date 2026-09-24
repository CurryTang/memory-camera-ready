#!/usr/bin/env python3
"""AMAbench evaluation runner with per-episode failure recovery.

Adapted from the official AMA-Bench runner.  Key additions:
- Per-episode JSONL checkpointing: each completed episode is appended
  immediately, so a crash only loses the in-flight episode.
- Automatic resume: on restart the script reads the checkpoint file,
  skips already-completed episode_ids, and continues where it left off.
- Works with any OpenAI-compatible LLM + embedding backend.

Usage:
    pixi run python examples/run_amabench.py \\
        --method bm25 \\
        --test-file datasets/amabench/open_end_qa_set_medium.jsonl \\
        --llm-model Qwen/Qwen3-32B --llm-base-url http://localhost:8002/v1 \\
        --output-dir results/amabench

    # With embedding method:
    pixi run python examples/run_amabench.py \\
        --method embedding \\
        --test-file datasets/amabench/open_end_qa_set_medium.jsonl \\
        --llm-model Qwen/Qwen3-32B --llm-base-url http://localhost:8002/v1 \\
        --embed-model qwen/qwen3-embedding --embed-base-url https://openrouter.ai/api/v1 \\
        --embed-api-key sk-or-... \\
        --output-dir results/amabench
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _PROJECT_ROOT)

from agentmem.eval.amabench_runner.model_client import ModelClient
from agentmem.eval.amabench_runner.embedding import EmbeddingEngine
from agentmem.eval.amabench_runner.methods import get_method, BaseMethod, METHOD_REGISTRY

if sys.path[0] != _PROJECT_ROOT:
    try:
        sys.path.remove(_PROJECT_ROOT)
    except ValueError:
        pass
    sys.path.insert(0, _PROJECT_ROOT)

from utils.amabench_prompting import get_answer_format_instruction

def extract_final_answer(response: str) -> str:
    """Extract answer after ##Answer: or ###Answer: marker (with or without spaces).

    Returns the full multi-line answer up to the next answer marker (if any),
    not just the first line — trajectory-QA answers are often multi-line
    step-by-step lists.
    """
    m = re.search(r"#{2,3}\s*Answer\s*:", response, re.IGNORECASE)
    if m:
        after = response[m.end():]

        stop = re.search(r"\n#{2,3}\s*Answer\s*\d*\s*:", after)
        if stop:
            after = after[: stop.start()]
        return after.strip()
    return response.strip()

def _clean_field(value: Any, *, max_chars: int | None = None) -> str:
    if value is None:
        return ""
    text = str(value)
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "\n[TRUNCATED]"
    return text

def trajectory_to_text(trajectory: List[Dict[str, Any]]) -> str:
    parts: list[str] = []
    max_observation_chars = int(os.getenv("AMABENCH_MAX_OBSERVATION_CHARS", "40000"))
    for step in trajectory:
        idx = step.get("turn_idx", 0)
        parts.append(f"Turn {idx}:")
        action = _clean_field(step.get("action"))
        observation = _clean_field(step.get("observation"), max_chars=max_observation_chars)
        thought = _clean_field(step.get("thought") or step.get("reasoning"))
        response = _clean_field(step.get("response") or step.get("assistant_response"))
        if thought:
            parts.append(f"  Thought: {thought}")
        if action:
            parts.append(f"  Action: {action}")
        if observation:
            parts.append(f"  Observation: {observation}")
        if response:
            parts.append(f"  Response: {response}")
    return "\n".join(parts)

def _episode_task_context(episode: Dict[str, Any]) -> str:
    """Build a stable, domain-aware task block for method construction."""
    domain = _clean_field(episode.get("domain")).strip()
    task_type = _clean_field(episode.get("task_type")).strip()
    task = _clean_field(episode.get("task")).strip()
    pieces = []
    if domain:
        pieces.append(f"Domain: {domain}")
    if task_type:
        pieces.append(f"Task type: {task_type}")
    if task:
        pieces.append(f"Task: {task}")
    return "\n".join(pieces)

def _episode_memory_text(episode: Dict[str, Any]) -> str:
    """Convert AMABench episode variants into one memory transcript.

    TEXT2SQL/Spider2 rows rely heavily on the schema-exploration task header.
    Keeping that header with the trajectory prevents methods that split only
    on turns from building an empty or schema-blind store.
    """
    task_context = _episode_task_context(episode)
    traj_text = trajectory_to_text(episode.get("trajectory", []) or [])
    domain = str(episode.get("domain", "") or "").lower()
    task_type = str(episode.get("task_type", "") or "").lower()
    if "text2sql" in domain or "spider" in task_type:
        return "\n\n".join(part for part in [task_context, "# Agent Trajectory", traj_text] if part)
    return traj_text

@contextmanager
def _alarm_timeout(seconds: int | None, label: str):
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handle_timeout(signum, frame):                                
        del signum, frame
        raise TimeoutError(f"{label} exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    old_alarm = signal.alarm(0)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_alarm:
            signal.alarm(old_alarm)

def extract_batch_answers(response: str, n_questions: int) -> List[str]:
    """Extract per-question answers from a batch LLM response.

    Keeps multi-line content — some questions need step-by-step lists.
    """
    answers: list[str] = []
    for i in range(n_questions):
        next_i = i + 2
        pattern = rf"###?\s*Answer\s*{i+1}\s*:\s*(.+?)(?=###?\s*Answer\s*{next_i}\s*:|$)"
        m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if m:
            answers.append(m.group(1).strip())
        else:

            answers.append("")
    return answers

def load_checkpoint(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load already-completed episodes from a JSONL checkpoint file."""
    completed: dict[int, dict] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                eid = rec.get("episode_id")
                if eid is not None:
                    completed[eid] = rec
            except json.JSONDecodeError:
                pass
    return completed

def split_checkpoint_records(records: Dict[int, Dict[str, Any]]) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Partition checkpoint records by success vs. error state using the latest record per episode."""
    successful = {}
    failed = {}
    for episode_id, record in records.items():
        if record.get("error"):
            failed[episode_id] = record
        else:
            successful[episode_id] = record
    return successful, failed

def _resolve_api_key(base_url: Optional[str], explicit_api_key: Optional[str]) -> Optional[str]:
    """Prefer explicit keys, then provider-specific env vars for known endpoints."""
    if explicit_api_key:
        return explicit_api_key
    base = (base_url or "").lower()
    if "openrouter.ai" in base:
        return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    return os.getenv("OPENAI_API_KEY")

def _infer_ama_embedding_defaults(
    method: str,
    llm_base_url: Optional[str],
    embed_model: Optional[str],
    embed_base_url: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Infer embedding defaults for memory methods that require a dense encoder."""
    normalized = method.replace("-", "_").lower()
    if normalized not in {"ama_agent", "hipporag", "simplemem", "lightmem"}:
        return embed_model, embed_base_url

    base = (llm_base_url or "").lower()
    if "openrouter.ai" in base:
        return embed_model or "qwen/qwen3-embedding-4b", embed_base_url or llm_base_url

    if any(host in base for host in ("localhost", "127.0.0.1", "0.0.0.0")):
        return embed_model or "Qwen3-Embedding-4B", embed_base_url or "http://localhost:8003/v1"

    return embed_model, embed_base_url

def get_default_method_config(method: str, llm_base_url: Optional[str] = None) -> Optional[Path]:
    """Return the official AMA-Bench config for methods with paper-default knobs."""
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "amabench"
    mapping = {
        "ama-agent": config_dir / "ama_agent_paper.yaml",
        "ama_agent": config_dir / "ama_agent_paper.yaml",
        "simplemem": config_dir / "simplemem_paper.yaml",
        "hipporag": config_dir / "hipporag_paper.yaml",
        "lightmem": config_dir / "lightmem_paper.yaml",
    }
    path = mapping.get(method)
    if path and path.exists():
        return path
    return None

def _apply_llm_token_budget_override(method: Any, llm_max_tokens: int) -> None:
    """Keep the method's internal runtime budget aligned with the CLI budget.

    Some AMABench methods load a released/paper config that carries its own
    `max_tokens` field. We DO NOT clobber that — the per-method YAML is
    paper-faithful (e.g. ama-agent uses 16384 for the causality-graph build
    phase, while longcontext uses 4096 for per-question answers). The CLI
    `--llm-max-tokens` only governs the top-level final-answer ModelClient.
    Methods that genuinely need to follow the CLI budget should expose
    `_model_client` whose ``max_tokens`` we set here; methods with their own
    YAML setting keep it.
    """
    model_client = getattr(method, "_model_client", None)
    if model_client is not None and hasattr(model_client, "max_tokens"):
        setattr(model_client, "max_tokens", llm_max_tokens)

def append_checkpoint(path: Path, record: Dict[str, Any]) -> None:
    """Append a single completed episode record to the checkpoint JSONL."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _persist_memory_artifact(
    *,
    method: BaseMethod,
    memory: Any,
    episode_id: Any,
    domain: str,
) -> Optional[str]:
    """Persist per-episode memory/index structure when requested.

    Set ``AMABENCH_MEMORY_DUMP_DIR`` to capture structures for later analysis.
    Native on-disk indexes such as PlugMem and HippoRAG are not copied here;
    their stable storage path is recorded in a manifest. In-memory structures
    such as AMA-Agent's causal/state graph are serialized as JSON.
    """
    root_raw = os.getenv("AMABENCH_MEMORY_DUMP_DIR")
    if not root_raw:
        return None

    method_name = (
        getattr(method, "name", None)
        or getattr(getattr(method, "inner", None), "name", None)
        or method.__class__.__name__
    )
    method_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(method_name)).strip("_") or "method"
    domain_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(domain or "unknown")).strip("_") or "unknown"
    episode_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(episode_id)).strip("_") or "episode"
    target_dir = Path(root_raw) / method_name / domain_name / f"episode_{episode_name}"
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "method": method_name,
        "domain": domain,
        "episode_id": episode_id,
        "memory_type": type(memory).__name__,
        "captured_at": datetime.now().isoformat(),
    }

    sample_dir = getattr(memory, "sample_dir", None)
    if sample_dir is not None:
        manifest["sample_dir"] = str(sample_dir)

    hipporag = getattr(memory, "hipporag", None)
    hipporag_config = getattr(hipporag, "global_config", None)
    hipporag_save_dir = getattr(hipporag_config, "save_dir", None)
    if hipporag_save_dir is not None:
        manifest["hipporag_save_dir"] = str(hipporag_save_dir)

    if isinstance(memory, dict):
        memory_path = target_dir / "memory.json"
        with memory_path.open("w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2, default=str)
        manifest["memory_json"] = str(memory_path)

    manifest_path = target_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return str(manifest_path)

def process_episode(
    episode: Dict[str, Any],
    method: BaseMethod,
    client: ModelClient,
    subset: str = "openend",
    max_question_concurrency: int = 1,
) -> Dict[str, Any]:
    """Process one episode: build memory → answer all questions.

    Returns a record with both compact ``answer_list`` (for eval) and a rich
    ``trajectory`` list (for analysis).  Each trajectory entry includes the
    full retrieved context, the QA prompt sent to the LLM, the raw LLM
    response, timing, and any method-specific diagnostics.
    """
    episode_id = episode.get("episode_id", 0)
    task = episode.get("task", "")
    domain = episode.get("domain", "")
    task_type = episode.get("task_type", "")
    qa_pairs = episode.get("qa_pairs", [])

    task_context = _episode_task_context(episode)
    traj_text = _episode_memory_text(episode)

    debug_timing = os.getenv("AMABENCH_DEBUG_TIMING") == "1"
    if debug_timing:
        print(f"[amabench-debug] ep={episode_id} memory_construction start", flush=True)
    mem_t0 = time.time()
    memory = method.memory_construction(traj_text, task_context or task)
    mem_sec = time.time() - mem_t0
    memory_artifact_path = _persist_memory_artifact(
        method=method,
        memory=memory,
        episode_id=episode_id,
        domain=domain,
    )
    if debug_timing:
        print(
            f"[amabench-debug] ep={episode_id} memory_construction done "
            f"latency_sec={mem_sec:.2f}",
            flush=True,
        )

    answer_format_single = get_answer_format_instruction(subset)

    answer_list: list[str] = []
    trajectory_list: list[dict] = []

    if getattr(method, "batch_mode", False) and len(qa_pairs) > 1:

        retrieved_context = method.memory_retrieve(memory, "")
        questions = [qa.get("question", "") for qa in qa_pairs]

        if subset == "mcq":
            batch_answer_format = ("For each question, provide your answer in the format: "
                                   "###Answer N: (X) where N is the question number and X is the option letter (A, B, C, or D).")
        else:
            batch_answer_format = ("For each question, provide your answer in the format: "
                                   "###Answer N: [your answer] where N is the question number.")

        questions_text = "\n\n".join(f"Question {i+1}: {q}" for i, q in enumerate(questions))
        batch_prompt = (f"{retrieved_context}\n\n# Questions\n{questions_text}\n\n"
                        f"# Instructions\nPlease answer ALL questions above based on the provided context.\n"
                        f"{batch_answer_format}\n\nAnswers:")

        llm_t0 = time.time()
        response = client.query(batch_prompt, temperature=0.0)
        llm_sec = time.time() - llm_t0
        batch_usage = dict(getattr(client, "last_usage", {}) or {})

        n_qa_for_split = max(len(qa_pairs), 1)
        per_q_prompt_tokens = int(batch_usage.get("prompt_tokens", 0)) // n_qa_for_split
        per_q_completion_tokens = int(batch_usage.get("completion_tokens", 0)) // n_qa_for_split

        batch_answers = extract_batch_answers(response, len(questions))
        answer_list = batch_answers

        for qi, qa in enumerate(qa_pairs):
            qa_traj: dict = {
                "question_idx": qi,
                "question": qa.get("question", ""),
                "qa_type": qa.get("type", ""),
                "gold": qa.get("answer", ""),
                "retrieved_context": retrieved_context,
                "retrieved_chars": len(retrieved_context),
                "mode": "batch_qa",
                "prediction": batch_answers[qi] if qi < len(batch_answers) else "",
                "llm_sec": round(llm_sec / n_qa_for_split, 3),
                "prompt_tokens": per_q_prompt_tokens,
                "completion_tokens": per_q_completion_tokens,
            }
            trajectory_list.append(qa_traj)
    else:

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _retrieve_one(qi: int, question: str) -> tuple[int, str, float]:
            if debug_timing:
                print(
                    f"[amabench-debug] ep={episode_id} q={qi} retrieve start",
                    flush=True,
                )
            t0 = time.time()
            retrieved = method.memory_retrieve(memory, question)
            sec = time.time() - t0
            if debug_timing:
                print(
                    f"[amabench-debug] ep={episode_id} q={qi} retrieve done "
                    f"latency_sec={sec:.2f} retrieved_chars={len(retrieved)}",
                    flush=True,
                )
            return qi, retrieved, sec

        retrievals: dict[int, tuple[str, float]] = {}
        n_qa = len(qa_pairs)
        worker_count = max(1, min(max_question_concurrency, n_qa))
        if worker_count == 1:
            for qi, qa in enumerate(qa_pairs):
                _, retrieved, sec = _retrieve_one(qi, qa.get("question", ""))
                retrievals[qi] = (retrieved, sec)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                futures = {
                    ex.submit(_retrieve_one, qi, qa.get("question", "")): qi
                    for qi, qa in enumerate(qa_pairs)
                }
                for future in as_completed(futures):
                    qi, retrieved, sec = future.result()
                    retrievals[qi] = (retrieved, sec)

        for qi, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            qa_type = qa.get("type", "")
            gold = qa.get("answer", "")
            retrieved, retrieve_sec = retrievals.get(qi, ("", 0.0))

            qa_traj: dict = {
                "question_idx": qi,
                "question": question,
                "qa_type": qa_type,
                "gold": gold,
                "retrieved_context": retrieved,
                "retrieved_chars": len(retrieved),
                "retrieve_sec": round(retrieve_sec, 3),
            }

            if retrieved.startswith("###Answer: "):
                answer = retrieved[len("###Answer: "):].strip()
                answer_list.append(answer)
                qa_traj["mode"] = "direct_answer"
                qa_traj["prediction"] = answer
            else:

                task_header = ""
                if task or domain:
                    parts = []
                    if domain:
                        parts.append(f"Domain: {domain}")
                    if task:
                        parts.append(f"Task: {task}")
                    task_header = " | ".join(parts) + "\n\n"
                prompt = (
                    f"{task_header}"
                    f"The following is retrieved memory context from an agent's trajectory.\n"
                    f"Use it to answer the question. If the context is insufficient, "
                    f"answer based on what is available.\n\n"
                    f"{retrieved}\n\n# Question\n{question}\n\n{answer_format_single}\n\nAnswer:"
                )
                if debug_timing:
                    print(
                        f"[amabench-debug] ep={episode_id} q={qi} qa start "
                        f"prompt_chars={len(prompt)}",
                        flush=True,
                    )
                llm_t0 = time.time()
                response = client.query(prompt, temperature=0.0)
                llm_sec = time.time() - llm_t0
                qa_usage = dict(getattr(client, "last_usage", {}) or {})
                if debug_timing:
                    print(
                        f"[amabench-debug] ep={episode_id} q={qi} qa done "
                        f"latency_sec={llm_sec:.2f}",
                        flush=True,
                    )
                answer = extract_final_answer(response)
                answer_list.append(answer)
                qa_traj["mode"] = "retrieval_qa"
                qa_traj["qa_prompt"] = prompt
                qa_traj["llm_response"] = response
                qa_traj["prediction"] = answer
                qa_traj["llm_sec"] = round(llm_sec, 3)
                qa_traj["prompt_tokens"] = int(qa_usage.get("prompt_tokens", 0))
                qa_traj["completion_tokens"] = int(qa_usage.get("completion_tokens", 0))

            if hasattr(method, "last_trajectory"):
                qa_traj.update(method.last_trajectory() or {})
            trajectory_list.append(qa_traj)

    t_query_input = sum(int(t.get("prompt_tokens", 0) or 0) for t in trajectory_list)
    t_query_output = sum(int(t.get("completion_tokens", 0) or 0) for t in trajectory_list)
    w_llm_query = sum(float(t.get("llm_sec", 0.0) or 0.0) for t in trajectory_list)
    w_tool_retrieve = sum(
        float(t.get("retrieve_sec", 0.0) or 0.0) for t in trajectory_list
    )

    counters = getattr(method, "counters", None)
    counters_dict = (
        counters.to_dict()
        if counters is not None and hasattr(counters, "to_dict")
        else {}
    )
    t_build_input = int(counters_dict.get("build_input_tokens", 0) or 0)
    t_build_output = int(counters_dict.get("build_output_tokens", 0) or 0)

    w_llm_build = float(counters_dict.get("w_llm_build", 0.0) or 0.0)
    w_tool_build = max(float(mem_sec) - w_llm_build, 0.0)

    efficiency = {
        "t_build_input": t_build_input,
        "t_build_output": t_build_output,
        "t_query_input": t_query_input,
        "t_query_output": t_query_output,
        "w_llm": round(w_llm_build + w_llm_query, 3),
        "w_tool": round(w_tool_build + w_tool_retrieve, 3),
        "gpu_util_mean": None,
    }
    efficiency_notes = []
    if not counters_dict:
        efficiency_notes.append(
            "no method.counters surface — t_build_input/output may be 0"
        )

    return {
        "episode_id": episode_id,
        "domain": domain,
        "task_type": task_type,
        "num_turns": len(episode.get("trajectory", []) or []),
        "num_questions": len(qa_pairs),
        "memory_construction_sec": round(mem_sec, 3),
        "memory_artifact": memory_artifact_path,
        "answer_list": answer_list,
        "trajectory": trajectory_list,
        "efficiency": efficiency,
        "efficiency_notes": efficiency_notes,
        "method_counters": counters_dict,
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AMAbench runner with failure recovery")

    p.add_argument(
        "--method",
        default="bm25",
        choices=sorted(METHOD_REGISTRY),
        help="Memory method",
    )
    p.add_argument("--method-config", default=None, help="Path to method config YAML/JSON")
    p.add_argument(
        "--metered",
        action="store_true",
        help=(
            "Route supported methods (simplemem, hipporag, plugmem, memt) through "
            "the unified agentmem.methods registry so EfficiencyCounters are "
            "populated alongside the legacy trajectory dump."
        ),
    )
    p.add_argument("--subset", default="openend", choices=["mcq", "openend"])

    p.add_argument("--llm-model", required=True, help="Model name for the LLM endpoint")
    p.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL")
    p.add_argument("--llm-api-key", default=None, help="API key (default: OPENAI_API_KEY or EMPTY)")
    p.add_argument("--llm-max-tokens", type=int, default=16384)

    p.add_argument("--embed-model", default=None, help="Embedding model name")
    p.add_argument("--embed-base-url", default=None, help="Embedding API base URL")
    p.add_argument("--embed-api-key", default=None, help="Embedding API key")
    p.add_argument("--embed-batch-size", type=int, default=8)

    p.add_argument("--test-file", required=True, help="Path to test JSONL")
    p.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional path to a subsample manifest JSON with an 'episode_ids' "
            "list. Filters --test-file to only those episode_ids before any "
            "--max-episodes sampling. Manifests live under "
            "agentmem/eval/amabench_runner/manifests/."
        ),
    )
    p.add_argument(
        "--backbone",
        default="qwen3-32b",
        help=(
            "Short backbone tag embedded in the default output filename "
            "(answers_<method>_<backbone>.jsonl). Override only if running a "
            "different backbone."
        ),
    )
    p.add_argument("--output-dir", default="results/amabench", help="Output directory")
    p.add_argument("--run-tag", default=None, help="Run tag (default: timestamp)")
    p.add_argument(
        "--max-question-concurrency",
        type=int,
        default=1,
        help="Max concurrent retrieval workers per episode (default: 1, matching the official runner)",
    )

    p.add_argument("--max-episodes", type=int, default=None,
                    help="Max episodes to process (random sample if < total)")
    p.add_argument("--random-seed", type=int, default=42,
                    help="Random seed for episode sampling (default: 42)")

    p.add_argument("--checkpoint-file", default=None,
                    help="Checkpoint JSONL path (default: auto from output-dir)")
    p.add_argument(
        "--episode-timeout-seconds",
        type=int,
        default=int(os.getenv("AMABENCH_EPISODE_TIMEOUT_SECONDS", "0") or 0),
        help="Hard timeout for one episode; timed-out episodes are checkpointed as failures.",
    )
    p.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=int(os.getenv("AMABENCH_MAX_RUNTIME_SECONDS", "0") or 0),
        help="Stop starting new episodes after this wall-clock budget and write a partial summary.",
    )

    p.add_argument("--judge-model", default=None,
                    help="LLM judge model (e.g. Qwen/Qwen3-32B). If set, auto-judges after eval.")
    p.add_argument("--judge-base-url", default=None,
                    help="Base URL for self-hosted judge (e.g. http://localhost:8000/v1)")
    p.add_argument("--judge-api-key", default=None,
                    help="API key for judge model (default: OPENAI_API_KEY)")
    p.add_argument(
        "--judge-mode",
        default="strict",
        choices=["strict", "amabench", "locomo", "hotpotqa"],
        help=(
            "Judge prompt mode. 'strict' is the headline cross-benchmark "
            "judge (matches LoCoMo + HotpotQA strict numbers). 'amabench' "
            "is the upstream-aligned yes/no prompt — emit it as a secondary "
            "footnote number, NOT as the headline (upstream measured a 9pt "
            "leniency bias on Qwen3-32B with this prompt)."
        ),
    )
    p.add_argument("--judge-concurrent", type=int, default=20,
                    help="Max concurrent judge API calls (default: 20)")

    p.add_argument(
        "--hf-repo-id",
        default=os.getenv("HF_REPO_ID"),
        help="HF dataset repo for automatic result upload",
    )
    p.add_argument(
        "--hf-path-prefix",
        default=os.getenv("HF_PATH_PREFIX", "amabench_results"),
        help="Path prefix inside the HF dataset repo",
    )

    return p.parse_args()

def main() -> None:
    args = parse_args()

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    backbone_tag = (args.backbone or "").strip() or "qwen3-32b"
    ckpt_path = Path(args.checkpoint_file) if args.checkpoint_file else (
        output_dir / f"answers_{args.method}_{backbone_tag}.jsonl"
    )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    completed = load_checkpoint(ckpt_path)
    successful_records, failed_records = split_checkpoint_records(completed)
    if completed:
        print(
            f"[resume] Found {len(completed)} latest checkpoint records in {ckpt_path} "
            f"({len(successful_records)} successful, {len(failed_records)} failed)"
        )

    llm_api_key = args.llm_api_key
    llm_base_url = args.llm_base_url
    if llm_base_url is None and os.getenv("OPENROUTER_API_KEY"):
        llm_base_url = "https://openrouter.ai/api/v1"
    llm_api_key = _resolve_api_key(llm_base_url, llm_api_key)
    args.embed_model, args.embed_base_url = _infer_ama_embedding_defaults(
        args.method,
        llm_base_url,
        args.embed_model,
        args.embed_base_url,
    )
    client = ModelClient(
        model=args.llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
        max_tokens=args.llm_max_tokens,
    )
    print(f"LLM: {args.llm_model} @ {llm_base_url or 'default'}")

    embedding_engine = None
    if args.method == "embedding":
        if not args.embed_model or not args.embed_base_url:
            raise ValueError("--embed-model and --embed-base-url required for embedding method")
        embedding_engine = EmbeddingEngine(
            model_name=args.embed_model,
            base_url=args.embed_base_url,
            api_key=_resolve_api_key(args.embed_base_url, args.embed_api_key) or "EMPTY",
            batch_size=args.embed_batch_size,
        )
        print(f"Embedding: {args.embed_model} @ {args.embed_base_url}")

    method_kwargs: dict[str, Any] = {}
    if args.method_config:
        method_kwargs["config_path"] = args.method_config
    else:
        default_config = get_default_method_config(args.method, llm_base_url=llm_base_url)
        if default_config is not None:
            method_kwargs["config_path"] = str(default_config)
            print(f"[config] Using official default config: {default_config}")
    if embedding_engine is not None:
        method_kwargs["embedding_engine"] = embedding_engine

    method_kwargs["llm_model"] = args.llm_model
    method_kwargs["llm_base_url"] = llm_base_url
    method_kwargs["llm_api_key"] = llm_api_key
    method_kwargs["llm_max_tokens"] = args.llm_max_tokens
    if args.embed_model:
        method_kwargs["embedding_model"] = args.embed_model
    if args.embed_base_url:
        method_kwargs["embedding_base_url"] = args.embed_base_url
    if args.embed_api_key:
        method_kwargs["embedding_api_key"] = args.embed_api_key
    elif args.embed_base_url:
        resolved_embed_api_key = _resolve_api_key(args.embed_base_url, None)
        if resolved_embed_api_key:
            method_kwargs["embedding_api_key"] = resolved_embed_api_key

    method = get_method(args.method, metered=args.metered, **method_kwargs)
    _apply_llm_token_budget_override(method, args.llm_max_tokens)
    print(f"Method: {args.method}")

    episodes: list[dict] = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.startswith("["):
        episodes = json.loads(raw)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    print(f"Dataset: {len(episodes)} episodes from {args.test_file}")

    if args.manifest:
        manifest_path = Path(args.manifest)
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        keep_ids = set(manifest.get("episode_ids") or [])
        if not keep_ids:
            raise ValueError(
                f"Manifest {manifest_path} has no 'episode_ids' field"
            )
        before = len(episodes)
        episodes = [ep for ep in episodes if ep.get("episode_id") in keep_ids]
        missing = keep_ids - {ep.get("episode_id") for ep in episodes}
        if missing:
            raise ValueError(
                f"Manifest references {len(missing)} episode_ids absent from "
                f"{args.test_file}: sample={sorted(missing)[:5]}"
            )
        print(
            f"Manifest {manifest_path.name}: filtered {before} -> {len(episodes)} "
            f"episodes (domain={manifest.get('domain')}, "
            f"task_type={manifest.get('task_type')})"
        )

    if args.max_episodes is not None and args.max_episodes < len(episodes):
        rng = random.Random(args.random_seed)
        episodes = rng.sample(episodes, args.max_episodes)
        print(f"Sampled {len(episodes)} episodes (seed={args.random_seed})")

    remaining = [ep for ep in episodes if ep.get("episode_id") not in successful_records]
    print(
        f"To process: {len(remaining)} episodes "
        f"({len(successful_records)} already successful, {len(failed_records)} scheduled for retry)"
    )

    if not remaining:
        print("All episodes completed. Nothing to do.")
        _write_summary(output_dir, args, episodes, ckpt_path)
        return

    run_config = {
        "method": args.method,
        "backbone": backbone_tag,
        "llm_model": args.llm_model,
        "llm_base_url": llm_base_url,
        "embed_model": args.embed_model,
        "embed_base_url": args.embed_base_url,
        "test_file": args.test_file,
        "manifest": args.manifest,
        "subset": args.subset,
        "max_episodes": args.max_episodes,
        "random_seed": args.random_seed,
        "run_tag": run_tag,
        "started_at": datetime.now().isoformat(),
        "episode_ids": [ep.get("episode_id") for ep in episodes],
        "episode_timeout_seconds": args.episode_timeout_seconds,
        "max_runtime_seconds": args.max_runtime_seconds,
    }
    config_path = output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    t0 = time.time()
    n_done = len(successful_records)
    n_total = len(episodes)

    for i, episode in enumerate(remaining):
        if args.max_runtime_seconds and (time.time() - t0) >= args.max_runtime_seconds:
            print(
                f"[timeout] Reached max runtime {args.max_runtime_seconds}s; "
                "writing partial summary.",
                flush=True,
            )
            break
        eid = episode.get("episode_id", "?")
        n_q = len(episode.get("qa_pairs", []))
        ep_start = time.time()

        try:
            with _alarm_timeout(args.episode_timeout_seconds, f"episode {eid}"):
                result = process_episode(
                    episode,
                    method,
                    client,
                    subset=args.subset,
                    max_question_concurrency=args.max_question_concurrency,
                )
            append_checkpoint(ckpt_path, result)
            n_done += 1
            elapsed = time.time() - ep_start
            total_elapsed = time.time() - t0
            rate = (i + 1) / total_elapsed if total_elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{n_done}/{n_total}] ep {eid} ({n_q}q) "
                f"done in {elapsed:.1f}s | ETA {eta/60:.0f}m",
                flush=True,
            )
        except Exception as exc:
            print(f"  [{n_done}/{n_total}] ep {eid} FAILED: {exc!r}", flush=True)

            fail_record = {
                "episode_id": eid,
                "answer_list": [],
                "error": str(exc),
            }
            append_checkpoint(ckpt_path, fail_record)

    total_time = time.time() - t0
    print(f"\nDone. {n_done}/{n_total} episodes in {total_time/60:.1f} min")
    print(f"Checkpoint: {ckpt_path}")

    _write_summary(output_dir, args, episodes, ckpt_path)

def _write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    episodes: list[dict],
    ckpt_path: Path,
) -> None:
    """Compute metrics and write summary JSON. Optionally runs LLM judge."""
    from agentmem.eval.amabench_metrics import per_question_metrics, summarize_amabench_rows

    completed = load_checkpoint(ckpt_path)
    successful_records, failed_records = split_checkpoint_records(completed)

    rows: list[dict] = []
    missing_predictions = 0
    for ep in episodes:
        eid = ep.get("episode_id")
        result = successful_records.get(eid)
        if result is None:
            continue
        answers = result.get("answer_list", [])
        qa_pairs = ep.get("qa_pairs", [])
        domain = ep.get("domain", "unknown")
        task_description = ep.get("task", "")

        missing_predictions += max(0, len(qa_pairs) - len(answers))

        for qa_index, (pred, qa) in enumerate(zip(answers, qa_pairs)):
            gold = str(qa.get("answer", ""))
            m = per_question_metrics(predicted=pred, golden=gold, question=qa.get("question", ""))
            rows.append({
                "episode_id": eid,
                "qa_index": qa_index,
                "question_uuid": qa.get("question_uuid", ""),
                "question": qa.get("question", ""),
                "qa_type": qa.get("type", "unknown"),
                "domain": domain,
                "task_description": task_description,
                "gold": gold,
                "prediction": pred,
                "method": args.method,
                "metrics": m,
            })

    judge_model = getattr(args, "judge_model", None)
    batch_result = None
    if judge_model and rows:
        from agentmem.eval.llm_judge import LLMJudge
        judge = LLMJudge(
            model=judge_model,
            api_key=_resolve_api_key(
                getattr(args, "judge_base_url", None),
                getattr(args, "judge_api_key", None),
            ),
            base_url=getattr(args, "judge_base_url", None),
            mode=getattr(args, "judge_mode", "amabench"),
            max_concurrent=getattr(args, "judge_concurrent", 20),
        )
        print(f"\nRunning LLM judge ({judge_model}) on {len(rows)} questions...")
        batch_result = judge.judge_batch(rows)
        for row, result in zip(rows, batch_result.results):
            row["llm_judge_correct"] = result.correct
            row["llm_judge_score"] = result.score
            row["llm_judge_reason"] = result.reason
            row["llm_judge_raw_response"] = result.raw_response
            row["llm_judge_parse_error"] = result.error is not None
            if result.error:
                row["llm_judge_error"] = result.error
            row["metrics"]["llm_judge"] = result.score
        print(f"LLM judge accuracy: {batch_result.accuracy*100:.1f}%")

    primary_metric = "llm_judge" if (judge_model and rows and "llm_judge" in rows[0].get("metrics", {})) else "f1_score"
    summary = summarize_amabench_rows(rows, group_key="qa_type", primary_metric=primary_metric)
    report = {
        "method": args.method,
        "llm_model": args.llm_model,
        "embed_model": args.embed_model,
        "judge_model": judge_model,
        "primary_metric": primary_metric,
        "test_file": args.test_file,
        "total_episodes": len(episodes),
        "completed_episodes": len(successful_records),
        "failed_episodes": len(failed_records),
        "total_questions": len(rows),
        "missing_predictions": missing_predictions,
        "by_qa_type": summary.get("groups", {}),
        "overall": summary.get("average", {}),
    }

    summary_path = output_dir / f"summary_{args.method}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Summary: {summary_path}")

    rows_path = output_dir / f"rows_{args.method}.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _maybe_publish_results(
        output_dir=output_dir,
        args=args,
        report=report,
        rows=rows,
        rows_path=rows_path,
        batch_result=batch_result,
    )

    metric_col = primary_metric.upper()[:8]
    print(f"\n{'QA Type':<20} {'Count':>6} {metric_col:>9}")
    print("-" * 38)
    for qtype, stats in sorted(report.get("by_qa_type", {}).items()):
        n = stats.get("num_questions", 0)
        v = (stats.get("metrics") or {}).get(primary_metric, 0) or 0
        print(f"{qtype:<20} {n:>6} {v:>9.4f}")
    overall = report.get("overall", {})
    print("-" * 38)
    print(f"{'AVERAGE':<20} {'':>6} {overall.get(primary_metric, 0) or 0:>9.4f}")

def _maybe_publish_results(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    rows_path: Path,
    batch_result: Any,
) -> None:
    """Publish AMABench outputs to Hugging Face when credentials are present."""
    try:
        from scripts.judge_publish import publish_to_hf, write_publish_metadata
    except Exception as exc:
        print(f"[publish] judge_publish unavailable: {exc}", flush=True)
        return

    valid_count = len([row for row in rows if not row.get("llm_judge_parse_error")])
    correct = sum(1 for row in rows if row.get("llm_judge_correct"))
    parse_errors = sum(1 for row in rows if row.get("llm_judge_parse_error"))

    publish_report = {
        "task_name": args.method,
        "total": len(rows),
        "valid_count": valid_count,
        "correct": correct,
        "overall_accuracy": (batch_result.accuracy if batch_result is not None else 0.0),
        "parse_errors": parse_errors,
        "judged_rows_file": str(rows_path),
        "primary_metric": report.get("primary_metric"),
        "overall": report.get("overall", {}),
    }

    metadata = {
        "benchmark": "amabench",
        "method": args.method,
        "judge_model": getattr(args, "judge_model", None),
        "rows_path": str(rows_path),
        "summary_path": str(output_dir / f"summary_{args.method}.json"),
        "hf_repo_id": getattr(args, "hf_repo_id", None),
        "hf_path_prefix": getattr(args, "hf_path_prefix", None),
    }
    write_publish_metadata(output_dir, metadata)

    hf_repo_id = getattr(args, "hf_repo_id", None)
    if hf_repo_id:
        publish_to_hf(
            benchmark="amabench",
            output_dir=output_dir,
            hf_repo_id=hf_repo_id,
            path_prefix=getattr(args, "hf_path_prefix", "amabench_results"),
        )

if __name__ == "__main__":
    main()
