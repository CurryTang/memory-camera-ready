"""
LOCOMO dataset adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from data.base import BaseDataset, DialogueRecord, DownloadConfig, QAPair
from data.factory import register_dataset
from data.utils.io import load_json

@register_dataset("locomo")
class LoCoMoDataset(BaseDataset):
    """
    Dataset adapter for LoCoMo evaluation data.

    Format references:
    - https://github.com/snap-research/locomo/blob/main/data/locomo10.json
    - https://github.com/aiming-lab/SimpleMem/blob/main/test_locomo10.py
    """

    NAME = "locomo"
    CATEGORY = "personalized-memory"
    INTERFACE_STYLE = "session_dialogue_qa"
    MEMORY_AXES = ("user_state_tracking", "temporal_reasoning", "adversarial_abstention")
    DEFAULT_VARIANT = "locomo10"
    DOWNLOAD_CONFIGS = {
        "locomo10": DownloadConfig(
            url=(
                "https://raw.githubusercontent.com/snap-research/"
                "locomo/main/data/locomo10.json"
            ),
            filename="locomo10.json",
            description="Official LoCoMo release with 10 long conversations.",
        ),
        "original": DownloadConfig(
            url=(
                "https://raw.githubusercontent.com/snap-research/"
                "locomo/main/data/locomo10.json"
            ),
            filename="locomo10.json",
            description="Alias of the official LoCoMo release for original-paper setup.",
        ),
    }

    def load_samples(self, path: str | Path) -> list[dict[str, Any]]:
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError("LOCOMO dataset must be a JSON list of samples.")
        return data

    def iter_dialogues(self, sample: dict[str, Any]) -> Iterable[DialogueRecord]:
        conversation = sample.get("conversation", {})
        if not isinstance(conversation, dict):
            return

        session_keys = sorted(
            [
                key
                for key, value in conversation.items()
                if key.startswith("session_") and isinstance(value, list)
            ],
            key=lambda k: int(k.split("_")[1]),
        )

        for session_key in session_keys:
            timestamp = conversation.get(f"{session_key}_date_time")
            turns = conversation.get(session_key, [])
            for turn in turns:
                content = self._normalize_turn_text(turn)
                if not content:
                    continue
                speaker = turn.get("speaker", "Unknown")
                yield DialogueRecord(
                    speaker=speaker,
                    content=content,
                    timestamp=timestamp,
                )

    def iter_qa_pairs(self, sample: dict[str, Any]) -> Iterable[QAPair]:
        qa_list = sample.get("qa", [])
        if not isinstance(qa_list, list):
            return

        for qa in qa_list:
            evidence_list = qa.get("evidence") or []
            if not isinstance(evidence_list, list):
                evidence_list = []

            yield QAPair(
                question=qa.get("question", ""),
                answer=qa.get("answer"),
                category=qa.get("category"),
                adversarial_answer=qa.get("adversarial_answer"),
                evidence=tuple(str(item) for item in evidence_list),
            )

    @staticmethod
    def _normalize_turn_text(turn: dict[str, Any]) -> str:
        text = str(turn.get("text", "") or "")
        caption = turn.get("blip_caption")
        has_image = "img_url" in turn and caption
        if has_image:
            text = f"[Image: {caption}] {text}".strip()
        return text.strip()
