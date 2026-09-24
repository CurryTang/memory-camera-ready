"""Long-context method — feeds full trajectory to the LLM.

Paper uses batch question mode: all questions for an episode are answered
in a single LLM call. The runner detects this via the ``batch_mode`` flag
and calls ``answer_all_questions_batch()`` instead of per-question QA.
"""

from __future__ import annotations

import os
import re
from typing import Any

from agentmem.eval.amabench_runner.methods.base import BaseMethod

class LongContextMemory:
    def __init__(self, full_text: str):
        self.full_text = full_text

class LongContextMethod(BaseMethod):

    batch_mode = False

    def __init__(
        self,

        max_model_length: int = 32768,
        max_response_tokens: int = 4096,
        chars_per_token: int = 4,
        config_path: str = None,
        **_kw,
    ):
        if config_path:
            cfg = self._load_config(config_path)
            max_model_length = cfg.get("max_model_length", max_model_length)
            max_response_tokens = cfg.get("max_response_tokens", max_response_tokens)
            chars_per_token = cfg.get("chars_per_token", chars_per_token)
        self.max_input_tokens = max_model_length - max_response_tokens
        self.chars_per_token = chars_per_token
        self.overflow_mode = os.environ.get("LONGCONTEXT_OVERFLOW_MODE", "search").lower()
        self.search_top_k = int(os.environ.get("LONGCONTEXT_SEARCH_TOP_K", "12"))
        self.search_chunk_words = int(os.environ.get("LONGCONTEXT_SEARCH_CHUNK_WORDS", "768"))
        self.search_overlap_words = int(os.environ.get("LONGCONTEXT_SEARCH_OVERLAP_WORDS", "64"))

    def memory_construction(self, traj_text: str, task: str = "") -> LongContextMemory:
        full_text = f"# Task\n{task}\n\n# Agent Trajectory\n{traj_text}" if task else traj_text
        return LongContextMemory(full_text)

    def memory_retrieve(self, memory: LongContextMemory, question: str) -> str:
        if self._estimate_tokens(memory.full_text) > self.max_input_tokens and self.overflow_mode == "search":
            return self._search_overflow_context(memory.full_text, question)
        return self._truncate(memory.full_text)

    def _truncate(self, text: str) -> str:
        est_tokens = self._estimate_tokens(text)
        if est_tokens <= self.max_input_tokens:
            return text
        target = int(self.max_input_tokens * self.chars_per_token)
        half = target // 2
        return text[:half] + "\n\n... [middle section truncated] ...\n\n" + text[-half:]

    def _estimate_tokens(self, text: str) -> float:
        return len(text) / max(1, self.chars_per_token)

    def _search_overflow_context(self, text: str, question: str) -> str:
        chunks = self._chunk_text(text)
        ranked = self._rank_chunks(chunks, question)
        selected = ranked[: max(1, self.search_top_k)]
        budget_chars = int(self.max_input_tokens * self.chars_per_token)
        header = (
            "[Long-context overflow handled with lexical search over bounded chunks.]\n"
            f"Question: {question}\n\n"
        )
        parts = [header]
        used = len(header)
        for rank, (idx, chunk, score) in enumerate(selected, start=1):
            block = f"\n[Search result {rank}; chunk={idx}; score={score:.3f}]\n{chunk}\n"
            if used + len(block) > budget_chars:
                remaining = max(0, budget_chars - used)
                if remaining > 200:
                    parts.append(block[:remaining])
                break
            parts.append(block)
            used += len(block)
        return "".join(parts)

    def _chunk_text(self, text: str) -> list[tuple[int, str]]:
        words = text.split()
        if not words:
            return []
        chunk_words = max(128, self.search_chunk_words)
        overlap = max(0, min(self.search_overlap_words, chunk_words - 1))
        step = chunk_words - overlap
        chunks: list[tuple[int, str]] = []
        for idx, start in enumerate(range(0, len(words), step), start=1):
            chunks.append((idx, " ".join(words[start : start + chunk_words])))
            if start + chunk_words >= len(words):
                break
        return chunks

    def _rank_chunks(self, chunks: list[tuple[int, str]], question: str) -> list[tuple[int, str, float]]:
        query_terms = re.findall(r"[A-Za-z0-9_]+", question.lower())
        query_counts: dict[str, int] = {}
        for term in query_terms:
            if len(term) <= 2:
                continue
            query_counts[term] = query_counts.get(term, 0) + 1
        if not query_counts:
            return [(idx, chunk, 0.0) for idx, chunk in chunks]

        ranked: list[tuple[int, str, float]] = []
        for idx, chunk in chunks:
            lower = chunk.lower()
            score = 0.0
            for term, weight in query_counts.items():
                if term in lower:
                    score += weight * (1.0 + min(5, lower.count(term)) / 5.0)
            ranked.append((idx, chunk, score))
        return sorted(ranked, key=lambda item: (item[2], -item[0]), reverse=True)
