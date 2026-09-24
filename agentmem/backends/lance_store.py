"""
LanceDB-backed memory store.

Lighter alternative to ChromaDB:
- Embedded, no server process required (file-based via Lance format)
- Default embedding: BAAI/bge-small-en-v1.5 (33M params, 384-dim)
  vs ChromaCollectionStore's BAAI/bge-m3 (570M params, 1024-dim)

Install: pip install lancedb sentence-transformers
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

import numpy as np

from agentmem.backends.base import BaseMemoryStore, MemoryRecord, SearchResult

class LanceVectorStore(BaseMemoryStore):
    """
    LanceDB-backed vector memory store with sentence-transformer embeddings.

    Args:
        path: Directory path for the LanceDB database files.
        table_name: Name of the LanceDB table to use.
        embedding_model: sentence-transformers model name. Defaults to
            BAAI/bge-small-en-v1.5 (33M params, 384-dim).
    """

    def __init__(
        self,
        path: str = "./lance_db",
        table_name: str = "memories",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        try:
            import lancedb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "LanceVectorStore requires `lancedb` and `sentence-transformers`. "
                "Install with: pip install lancedb sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(embedding_model)
        self._dim: int = self._model.get_sentence_embedding_dimension()
        self._table_name = table_name

        self._db = lancedb.connect(path)
        self._schema = self._build_schema()
        self._table = self._open_or_create_table()

    def add(self, record: MemoryRecord) -> str:
        vec = self._embed(record.content)
        row = {
            "id": record.id,
            "content": record.content,
            "meta_json": json.dumps(record.metadata or {}, ensure_ascii=False),
            "vector": vec.tolist(),
        }

        safe = record.id.replace("'", "''")
        try:
            self._table.delete(f"id = '{safe}'")
        except Exception:
            pass
        self._table.add([row])
        return record.id

    def get(self, id: str) -> Optional[MemoryRecord]:
        safe = id.replace("'", "''")
        try:
            rows = (
                self._table.search()
                .where(f"id = '{safe}'", prefilter=True)
                .limit(1)
                .to_list()
            )
        except Exception:

            import pandas as pd

            df = self._table.to_pandas()
            rows = df[df["id"] == id].to_dict("records")

        if not rows:
            return None
        row = rows[0]
        return MemoryRecord(
            id=str(row["id"]),
            content=str(row["content"]),
            metadata=json.loads(row.get("meta_json") or "{}"),
        )

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        q = (query or "").strip()
        if not q or k <= 0 or self.count() == 0:
            return []

        vec = self._embed(q)
        n = min(int(k), self.count())
        try:
            rows = self._table.search(vec).limit(n).to_list()
        except Exception:
            return []

        results: list[SearchResult] = []
        for row in rows:
            dist = float(row.get("_distance", 1.0))
            score = 1.0 / (1.0 + max(dist, 0.0))
            results.append(
                SearchResult(
                    record=MemoryRecord(
                        id=str(row["id"]),
                        content=str(row["content"]),
                        metadata=json.loads(row.get("meta_json") or "{}"),
                    ),
                    score=score,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def delete(self, id: str) -> None:
        safe = id.replace("'", "''")
        try:
            self._table.delete(f"id = '{safe}'")
        except Exception:
            pass

    def clear(self) -> None:
        try:
            self._db.drop_table(self._table_name)
        except Exception:
            pass
        self._table = self._open_or_create_table()

    def count(self) -> int:
        try:
            return int(self._table.count_rows())
        except Exception:
            try:
                return len(self._table.to_pandas())
            except Exception:
                return 0

    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
        batch_size: int = 256,
    ) -> list[str]:
        """
        Batch-embed and insert a list of text passages efficiently.

        Args:
            texts: Passage strings to index.
            metadatas: Optional per-passage metadata dicts.
            ids: Optional pre-assigned IDs; auto-generated if None.
            batch_size: Embedding batch size.

        Returns:
            List of assigned IDs.
        """
        if not texts:
            return []

        n = len(texts)
        metadatas = metadatas or [{} for _ in range(n)]
        ids = ids or [str(uuid.uuid4()) for _ in range(n)]

        assigned: list[str] = []
        for start in range(0, n, batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_metas = metadatas[start : start + batch_size]
            batch_ids = ids[start : start + batch_size]

            vecs = self._model.encode(batch_texts, show_progress_bar=False)
            rows = [
                {
                    "id": bid,
                    "content": t,
                    "meta_json": json.dumps(m or {}, ensure_ascii=False),
                    "vector": v.tolist(),
                }
                for bid, t, m, v in zip(batch_ids, batch_texts, batch_metas, vecs)
            ]
            self._table.add(rows)
            assigned.extend(batch_ids)

        return assigned

    def _embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, show_progress_bar=False)

    def _build_schema(self):
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("meta_json", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self._dim)),
            ]
        )

    def _open_or_create_table(self):
        if self._table_name in self._db.table_names():
            return self._db.open_table(self._table_name)
        return self._db.create_table(self._table_name, schema=self._schema)
