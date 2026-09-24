"""Embedding engine — OpenAI-compatible API only (no local models).

Works with sglang --is-embedding, vllm embedding, OpenRouter, etc.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from openai import OpenAI

class EmbeddingEngine:
    """Encode texts via an OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "EMPTY",
        batch_size: int = 8,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        if not texts:
            return np.array([])

        embeddings: list = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def __call__(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
