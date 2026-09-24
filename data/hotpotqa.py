"""
HotpotQA dataset adapter.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from data.base import BaseDataset, DialogueRecord, DownloadConfig, QAPair
from data.factory import register_dataset
from data.utils.io import load_json

_DOCUMENT_SPLIT_REGEX = re.compile(r"(?:^|\n)Document\s+\d+\s*:\s*\n?", re.IGNORECASE)

@register_dataset("hotpotqa")
class HotpotQADataset(BaseDataset):
    """
    Dataset adapter for HotpotQA-style data.

    Supported sample layouts:
    - Official/standard HotpotQA:
      {"question", "answer", "supporting_facts", "context": [[title, [sentences...]], ...]}
    - Long-context synthesized format used by some memory benchmarks:
      {"input", "answers", "context": "Document 1:\\n...Document 2:\\n..."}
    """

    NAME = "hotpotqa"
    CATEGORY = "rag-era"
    INTERFACE_STYLE = "retrieval_qa"
    MEMORY_AXES = ("retrieval_precision", "multi_hop_reasoning")
    DEFAULT_VARIANT = "distractor_dev"
    DOWNLOAD_CONFIGS = {
        "distractor_dev": DownloadConfig(
            url=(
                "http://curtis.ml.cmu.edu/datasets/hotpot/"
                "hotpot_dev_distractor_v1.json"
            ),
            filename="hotpot_dev_distractor_v1.json",
            description=(
                "Official HotpotQA distractor dev split "
                "(7,405 samples).  Required by AC-3."
            ),
        ),
        "hipporag": DownloadConfig(
            url=(
                "https://raw.githubusercontent.com/OSU-NLP-Group/"
                "HippoRAG/main/reproduce/dataset/hotpotqa.json"
            ),
            filename="hotpotqa.json",
            description=(
                "HotpotQA split used by HippoRAG reproduce setup "
                "(1,000 samples with context and supporting_facts)."
            ),
        ),
        "default": DownloadConfig(
            url=(
                "http://curtis.ml.cmu.edu/datasets/hotpot/"
                "hotpot_dev_distractor_v1.json"
            ),
            filename="hotpot_dev_distractor_v1.json",
            description="Alias for official distractor dev split.",
        ),
    }

    def load_samples(self, path: str | Path) -> list[dict[str, Any]]:
        data = load_json(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "samples", "records"):
                nested = data.get(key)
                if isinstance(nested, list):
                    return nested
        raise ValueError(
            "HotpotQA dataset must be a JSON list or a dict containing "
            "a list under one of: data/samples/records."
        )

    def iter_dialogues(self, sample: dict[str, Any]) -> Iterable[DialogueRecord]:
        for title, sentences in self._iter_context_documents(sample):
            speaker = title or "Document"
            for sentence in sentences:
                content = str(sentence or "").strip()
                if not content:
                    continue
                yield DialogueRecord(speaker=speaker, content=content, timestamp=None)

    def iter_qa_pairs(self, sample: dict[str, Any]) -> Iterable[QAPair]:
        question = str(sample.get("question") or sample.get("input") or "").strip()
        answer = self._extract_answer(sample)
        evidence = tuple(self._extract_evidence(sample))

        if not question and answer is None:
            return

        yield QAPair(
            question=question,
            answer=answer,
            category=None,
            evidence=evidence,
        )

    @classmethod
    def _iter_context_documents(cls, sample: dict[str, Any]) -> Iterable[tuple[str, list[str]]]:
        context = sample.get("context")

        if isinstance(context, list):
            for index, entry in enumerate(context, start=1):
                title, sentences = cls._parse_structured_context_entry(entry, index=index)
                if sentences:
                    yield title, sentences
            return

        if isinstance(context, str):
            for index, raw_doc in enumerate(cls._split_serialized_documents(context), start=1):
                lines = [line.strip() for line in raw_doc.splitlines() if line.strip()]
                if not lines:
                    continue
                if len(lines) == 1:
                    title = f"Document {index}"
                    sentences = [lines[0]]
                else:
                    title = lines[0]
                    sentences = [" ".join(lines[1:]).strip()]
                yield title, sentences

    @staticmethod
    def _parse_structured_context_entry(entry: Any, *, index: int) -> tuple[str, list[str]]:
        default_title = f"Document {index}"

        if isinstance(entry, dict):
            title = str(entry.get("title") or default_title).strip() or default_title
            sentences = HotpotQADataset._normalize_sentences(
                entry.get("sentences")
                or entry.get("text")
                or entry.get("context")
                or entry.get("content")
                or []
            )
            return title, sentences

        if isinstance(entry, (list, tuple)):
            if len(entry) >= 2:
                title = str(entry[0] or default_title).strip() or default_title
                sentences = HotpotQADataset._normalize_sentences(entry[1])
                return title, sentences
            if len(entry) == 1:
                title = default_title
                sentences = HotpotQADataset._normalize_sentences(entry[0])
                return title, sentences
            return default_title, []

        if isinstance(entry, str):
            text = entry.strip()
            return default_title, [text] if text else []

        return default_title, []

    @staticmethod
    def _normalize_sentences(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, tuple):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return []

    @staticmethod
    def _split_serialized_documents(raw_context: str) -> list[str]:
        text = raw_context.strip()
        if not text:
            return []

        chunks = [chunk.strip() for chunk in _DOCUMENT_SPLIT_REGEX.split(text) if chunk.strip()]
        if chunks:
            return chunks
        return [text]

    @staticmethod
    def _extract_answer(sample: dict[str, Any]) -> str | None:
        answer = sample.get("answer")
        if answer is not None:
            return str(answer)

        answers = sample.get("answers")
        if isinstance(answers, list) and answers:
            return str(answers[0])
        if isinstance(answers, str) and answers.strip():
            return answers.strip()
        return None

    @staticmethod
    def _extract_evidence(sample: dict[str, Any]) -> list[str]:
        evidence: list[str] = []
        raw_support = sample.get("supporting_facts") or sample.get("evidence") or []

        if not isinstance(raw_support, list):
            return evidence

        for item in raw_support:
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    title = str(item[0]).strip()
                    sentence_idx = str(item[1]).strip()
                    if title and sentence_idx:
                        evidence.append(f"{title}:{sentence_idx}")
                        continue
                for part in item:
                    text = str(part).strip()
                    if text:
                        evidence.append(text)
                continue

            text = str(item).strip()
            if not text:
                continue
            if ";" in text:
                evidence.extend(ref.strip() for ref in text.split(";") if ref.strip())
            else:
                evidence.append(text)

        return evidence
