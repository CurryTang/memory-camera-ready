"""CLI for ALFWorld prompt-agent evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from agentmem.eval.alfworld_runner.agents import create_agent
from agentmem.eval.alfworld_runner.memory import create_memory_backend
from agentmem.eval.alfworld_runner.runner import AlfWorldEvalRunner, load_alfworld_tasks
from agentmem.eval.alfworld_runner.env import ALFWorldConfig

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ALFWorld prompt agents")
    parser.add_argument(
        "--agent",
        choices=[
            "vanilla",
            "react",
            "classic_react",
            "react_memory",
            "reflexion",
            "reflexion_memory",
        ],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--split", default="eval_in_distribution")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-history", type=int, default=2)
    parser.add_argument("--max-chat-history-turns", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--react-few-shot-path", default=None)
    parser.add_argument("--react-num-examples", type=int, default=2)
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--max-reflections", type=int, default=3)
    parser.add_argument("--reflection-max-tokens", type=int, default=256)

    parser.add_argument(
        "--memory-backend",
        choices=[
            "hipporagv2",
            "plugmem",
            "simplemem",
            "lightmem",
            "memt",
            "memrl",
            "dci_lite",
            "dci_lite_sum",
            "ama_agent",
        ],
        default=None,
    )
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--memory-config-path", default=None)
    parser.add_argument("--memory-llm-model", default=None)
    parser.add_argument("--memory-llm-base-url", default=None)
    parser.add_argument("--memory-llm-api-key", default=None)
    parser.add_argument("--memory-embedding-model", default=None)
    parser.add_argument("--memory-embedding-base-url", default=None)
    parser.add_argument("--memory-embedding-api-key", default=None)
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    memory_backend = None
    needs_memory = args.agent in {"react_memory", "reflexion_memory"}
    if needs_memory and not args.memory_backend:
        parser.error(f"--memory-backend is required for {args.agent}")
    if args.memory_backend:
        memory_backend = create_memory_backend(
            args.memory_backend,
            top_k=args.memory_top_k,
            llm_model=args.memory_llm_model or args.model,
            llm_base_url=args.memory_llm_base_url or args.base_url,
            llm_api_key=args.memory_llm_api_key or args.api_key,
            embedding_model=args.memory_embedding_model,
            embedding_base_url=args.memory_embedding_base_url,
            embedding_api_key=args.memory_embedding_api_key,
            config_path=args.memory_config_path,
        )

    agent = create_agent(
        args.agent,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_history=args.max_history,
        max_chat_history_turns=args.max_chat_history_turns,
        memory_backend=memory_backend,
        react_few_shot_path=args.react_few_shot_path,
        react_num_examples=args.react_num_examples,
        max_trials=args.max_trials,
        reflection_max_tokens=args.reflection_max_tokens,
        max_reflections=args.max_reflections,
    )

    tasks = load_alfworld_tasks(
        split=args.split,
        limit=args.num_episodes,
        seed=args.seed,
    )
    runner = AlfWorldEvalRunner(
        agent=agent,
        env_config=ALFWorldConfig(max_steps=args.max_steps, use_gym=True),
        checkpoint_dir=args.checkpoint_dir,
    )
    report = runner.run(tasks)

    payload = {
        "agent": args.agent,
        "model": args.model,
        "memory_backend": args.memory_backend,
        "num_tasks": report.num_tasks,
        "n_episodes": report.num_tasks,
        "num_success": report.num_success,
        "n_success": report.num_success,
        "success_rate": report.success_rate,
        "avg_steps": report.avg_steps,
        "avg_time_seconds": report.avg_time_seconds,
        "tasks": [result.__dict__ for result in report.tasks],
    }

    output = args.output or f"results/alfworld_{args.agent}.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.getLogger(__name__).info("Saved report to %s", output)
    logging.getLogger(__name__).info("%s", report.summary())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
