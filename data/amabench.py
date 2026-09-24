"""
AMABench dataset adapter.

AMA-Bench evaluates long-horizon memory for agentic applications.
Each episode is an agent trajectory (action-observation turns) with
QA pairs across 4 types: A (Recall), B (Causal Inference),
C (State Updating), D (State Abstraction).

Domains: TEXT2SQL, EMBODIED_AI, WEB, SOFTWARE, OPENWORLD_QA, Game

Data source: https://huggingface.co/datasets/AMA-bench/AMA-bench
Paper: https://arxiv.org/abs/2602.22769
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from data.base import BaseDataset, DialogueRecord, DownloadConfig, QAPair
from data.factory import register_dataset

logger = logging.getLogger(__name__)

_QA_TYPE_TO_CATEGORY = {"A": 1, "B": 2, "C": 3, "D": 4}

@register_dataset("amabench")
class AMABenchDataset(BaseDataset):
    """
    Dataset adapter for AMA-Bench.

    Data format (JSONL, one episode per line):
        {
            "episode_id": int,
            "task": str,
            "task_type": str,
            "domain": str,           # TEXT2SQL | EMBODIED_AI | WEB | ...
            "success": bool,
            "num_turns": int,
            "total_tokens": int,
            "trajectory": [{"turn_idx": int, "action": str?, "observation": str?}, ...],
            "qa_pairs": [{"question": str, "answer": str, "type": str, "question_uuid": str}, ...]
        }
    """

    NAME = "amabench"
    CATEGORY = "agent-memory"
    INTERFACE_STYLE = "trajectory_qa"
    MEMORY_AXES = (
        "recall",
        "causal_inference",
        "state_updating",
        "state_abstraction",
    )
    DEFAULT_VARIANT = "v1"
    DOWNLOAD_CONFIGS: dict[str, DownloadConfig] = {}

    DOMAINS_RQ1 = ("TEXT2SQL", "EMBODIED_AI")

    def load_samples(self, path: str | Path) -> list[dict[str, Any]]:
        """Load episodes from a JSONL file (or directory containing one)."""
        p = Path(path)
        if p.is_dir():
            jsonl_files = list(p.glob("**/*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(f"No .jsonl files found in {p}")
            p = jsonl_files[0]

        samples = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        logger.info("Loaded %d AMBench episodes from %s", len(samples), p)
        return samples

    def load_domain(
        self, path: str | Path, domain: str,
    ) -> list[dict[str, Any]]:
        """Load only episodes from a specific domain."""
        return [s for s in self.load_samples(path) if s.get("domain") == domain]

    def iter_dialogues(self, sample: dict[str, Any]) -> Iterable[DialogueRecord]:
        """Yield trajectory turns as DialogueRecord for memory ingestion.

        Each (action, observation) pair becomes a DialogueRecord so that
        retrieval configs can index the trajectory content.
        """
        for turn in sample.get("trajectory", []):
            turn_idx = turn.get("turn_idx", 0)
            action = turn.get("action")
            observation = turn.get("observation")

            parts = [f"[Turn {turn_idx}]"]
            if action:
                parts.append(f"Action: {action}")
            if observation:
                parts.append(f"Observation: {observation}")
            content = " | ".join(parts)

            yield DialogueRecord(
                speaker="agent",
                content=content,
                timestamp=None,
            )

    def iter_qa_pairs(self, sample: dict[str, Any]) -> Iterable[QAPair]:
        """Yield QA pairs from an AMBench episode.

        Maps AMBench QA types (A/B/C/D) to integer categories (1/2/3/4).
        """
        for qa in sample.get("qa_pairs", []):
            qa_type = qa.get("type", "A")
            yield QAPair(
                question=str(qa.get("question", "")),
                answer=qa.get("answer"),
                category=_QA_TYPE_TO_CATEGORY.get(qa_type),
                adversarial_answer=None,
                evidence=(),
            )
