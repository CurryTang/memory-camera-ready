"""
Open-Viking relational memory backend.

This backend keeps AgentMem's BaseMemoryStore API while exposing table-level
helpers needed by relational memory pipelines.

Design goals:
- Work with an injected Open-Viking client when available.
- Provide an in-process fallback client for local development/testing.
- Preserve BaseMemoryStore semantics for existing operations.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Protocol

from agentmem.backends.base import BaseMemoryStore, MemoryRecord, SearchResult

class OpenVikingClient(Protocol):
    """Protocol for Open-Viking-like clients."""

    def upsert(self, *, namespace: str, table: str, row: dict[str, Any]) -> str:
        ...

    def get(self, *, namespace: str, table: str, row_id: str) -> Optional[dict[str, Any]]:
        ...

    def delete(self, *, namespace: str, table: str, row_id: str) -> None:
        ...

    def clear(self, *, namespace: str, table: Optional[str] = None) -> None:
        ...

    def count(self, *, namespace: str, table: str) -> int:
        ...

    def search(
        self,
        *,
        namespace: str,
        table: str,
        query: str,
        k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        ...

    def list_rows(self, *, namespace: str, table: str) -> list[dict[str, Any]]:
        ...

@dataclass(frozen=True)
class OpenVikingTableRef:
    namespace: str
    table: str

class InMemoryOpenVikingClient:
    """
    Lightweight stand-in for Open-Viking.

    Data layout:
      data[namespace][table][row_id] = row_dict
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    def _ensure(self, namespace: str, table: str) -> dict[str, dict[str, Any]]:
        ns = self._data.setdefault(str(namespace), {})
        return ns.setdefault(str(table), {})

    def upsert(self, *, namespace: str, table: str, row: dict[str, Any]) -> str:
        rows = self._ensure(namespace, table)
        row_id = str(row.get("id") or "")
        if not row_id:
            raise ValueError("row['id'] is required for upsert.")
        payload = dict(row)
        payload["id"] = row_id
        payload.setdefault("metadata", {})
        rows[row_id] = payload
        return row_id

    def get(self, *, namespace: str, table: str, row_id: str) -> Optional[dict[str, Any]]:
        rows = self._data.get(str(namespace), {}).get(str(table), {})
        row = rows.get(str(row_id))
        return dict(row) if row is not None else None

    def delete(self, *, namespace: str, table: str, row_id: str) -> None:
        rows = self._data.get(str(namespace), {}).get(str(table), {})
        rows.pop(str(row_id), None)

    def clear(self, *, namespace: str, table: Optional[str] = None) -> None:
        ns_key = str(namespace)
        if ns_key not in self._data:
            return
        if table is None:
            self._data[ns_key] = {}
            return
        self._data[ns_key].pop(str(table), None)

    def count(self, *, namespace: str, table: str) -> int:
        return len(self._data.get(str(namespace), {}).get(str(table), {}))

    def list_rows(self, *, namespace: str, table: str) -> list[dict[str, Any]]:
        rows = self._data.get(str(namespace), {}).get(str(table), {})
        return [dict(v) for v in rows.values()]

    def search(
        self,
        *,
        namespace: str,
        table: str,
        query: str,
        k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        rows = self._data.get(str(namespace), {}).get(str(table), {})
        if not query.strip() and not filters:
            out = [dict(v) for v in rows.values()]
            return out[: max(int(k), 0)]

        q = query.lower().strip()
        q_terms = _tokenize_terms(q) if q else set()
        top_k = max(int(k), 0)

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows.values():
            if filters and not _matches_filters(row=row, filters=filters):
                continue

            content = str(row.get("content") or "")
            metadata = row.get("metadata") or {}
            columns = metadata.get("columns") or {}
            blob = " ".join(
                [
                    content,
                    json.dumps(columns, ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                ]
            ).lower()

            if not q:
                score = 1.0
            else:
                overlap = len(q_terms & _tokenize_terms(blob))
                contains = 1.0 if q in blob else 0.0
                if overlap == 0 and contains == 0.0:
                    continue
                score = float(overlap) + contains

                ts = _extract_timestamp(metadata)
                if ts:
                    score += _recency_bonus(ts)

            scored.append((score, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, payload in scored[:top_k]:
            payload["_score"] = float(score)
            out.append(payload)
        return out

class OpenVikingMemoryStore(BaseMemoryStore):
    """
    Open-Viking-backed memory store with relational-table helpers.

    BaseMemoryStore methods operate on `default_table` to stay compatible with
    existing AgentMem operations.
    """

    def __init__(
        self,
        *,
        namespace: str = "agentmem",
        default_table: str = "memories",
        client: Optional[OpenVikingClient] = None,
    ) -> None:
        self.namespace = str(namespace)
        self.default_table = str(default_table)
        self._client: OpenVikingClient = client or self._maybe_build_external_client() or InMemoryOpenVikingClient()
        self._id_to_table: dict[str, str] = {}

    def _maybe_build_external_client(self) -> Optional[OpenVikingClient]:
        """
        Best-effort Open-Viking SDK wiring.

        If the SDK is unavailable (or env is not configured), return None and
        fall back to the in-memory adapter.
        """
        endpoint = os.getenv("OPEN_VIKING_ENDPOINT")
        api_key = os.getenv("OPEN_VIKING_API_KEY")

        try:
            import open_viking                
        except Exception:
            return None

        if not endpoint:
            return None

        for ctor_name in ("Client", "OpenVikingClient"):
            ctor = getattr(open_viking, ctor_name, None)
            if ctor is None:
                continue
            try:
                if api_key:
                    return ctor(endpoint=endpoint, api_key=api_key)
                return ctor(endpoint=endpoint)
            except Exception:
                continue
        return None

    def add(self, record: MemoryRecord) -> str:
        return self.add_to_table(self.default_table, record)

    def get(self, id: str) -> Optional[MemoryRecord]:
        row_id = str(id)
        table = self._id_to_table.get(row_id)
        if table is not None:
            row = self._client.get(namespace=self.namespace, table=table, row_id=row_id)
            return _row_to_record(row) if row else None

        row = self._client.get(namespace=self.namespace, table=self.default_table, row_id=row_id)
        if row:
            self._id_to_table[row_id] = self.default_table
            return _row_to_record(row)

        for table_name in self.available_tables():
            row = self._client.get(namespace=self.namespace, table=table_name, row_id=row_id)
            if row:
                self._id_to_table[row_id] = table_name
                return _row_to_record(row)
        return None

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        return self.search_table(self.default_table, query=query, k=k)

    def delete(self, id: str) -> None:
        row_id = str(id)
        table = self._id_to_table.get(row_id, self.default_table)
        self._client.delete(namespace=self.namespace, table=table, row_id=row_id)
        self._id_to_table.pop(row_id, None)

    def clear(self) -> None:
        self._client.clear(namespace=self.namespace)
        self._id_to_table = {}

    def count(self) -> int:
        return int(self._client.count(namespace=self.namespace, table=self.default_table))

    def add_to_table(self, table: str, record: MemoryRecord) -> str:
        table_name = str(table)
        payload = {
            "id": str(record.id),
            "content": str(record.content),
            "metadata": dict(record.metadata or {}),
        }
        out_id = self._client.upsert(namespace=self.namespace, table=table_name, row=payload)
        self._id_to_table[str(out_id)] = table_name
        return str(out_id)

    def upsert_row(
        self,
        *,
        table: str,
        row_id: str,
        columns: dict[str, Any],
        content: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        meta = dict(metadata or {})
        meta["columns"] = dict(columns)
        record = MemoryRecord(
            id=str(row_id),
            content=str(content or _row_content_from_columns(columns)),
            metadata=meta,
        )
        return self.add_to_table(table, record)

    def get_row(self, *, table: str, row_id: str) -> Optional[MemoryRecord]:
        row = self._client.get(namespace=self.namespace, table=str(table), row_id=str(row_id))
        if row is None:
            return None
        self._id_to_table[str(row_id)] = str(table)
        return _row_to_record(row)

    def list_rows(self, *, table: str) -> list[MemoryRecord]:
        out: list[MemoryRecord] = []
        rows = self._client.list_rows(namespace=self.namespace, table=str(table))
        for row in rows:
            rec = _row_to_record(row)
            out.append(rec)
            self._id_to_table[rec.id] = str(table)
        return out

    def search_table(
        self,
        table: str,
        *,
        query: str,
        k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        rows = self._client.search(
            namespace=self.namespace,
            table=str(table),
            query=str(query),
            k=max(int(k), 0),
            filters=dict(filters or {}),
        )

        results: list[SearchResult] = []
        for row in rows:
            rec = _row_to_record(row)
            self._id_to_table[rec.id] = str(table)
            score = float(row.get("_score", 0.0))
            results.append(SearchResult(record=rec, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def count_table(self, table: str) -> int:
        return int(self._client.count(namespace=self.namespace, table=str(table)))

    def clear_table(self, table: str) -> None:
        self._client.clear(namespace=self.namespace, table=str(table))
        table_name = str(table)
        self._id_to_table = {rid: t for rid, t in self._id_to_table.items() if t != table_name}

    def available_tables(self) -> list[str]:
        if isinstance(self._client, InMemoryOpenVikingClient):
            ns = self._client._data.get(self.namespace, {})                                              
            return sorted(ns.keys())
        known = sorted(set(self._id_to_table.values()))
        if self.default_table not in known:
            known.append(self.default_table)
        return known

def _row_to_record(row: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        id=str(row.get("id") or ""),
        content=str(row.get("content") or ""),
        metadata=dict(row.get("metadata") or {}),
    )

def _row_content_from_columns(columns: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in columns.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        parts.append(f"{key}: {text}")
    return " | ".join(parts)

def _tokenize_terms(text: str) -> set[str]:
    tokens = {tok for tok in re.findall(r"[a-z0-9_]+", str(text).lower()) if tok}
    expanded = set(tokens)
    for tok in tokens:
        if len(tok) > 3 and tok.endswith("s"):
            expanded.add(tok[:-1])
    return expanded

def _matches_filters(*, row: dict[str, Any], filters: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    columns = metadata.get("columns") or {}
    for key, expected in filters.items():
        if key in columns:
            actual = columns.get(key)
        else:
            actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True

def _extract_timestamp(metadata: dict[str, Any]) -> Optional[str]:
    for key in ("timestamp", "time", "updated_at", "created_at"):
        value = metadata.get(key)
        if value:
            return str(value)
    columns = metadata.get("columns") or {}
    for key in ("timestamp", "time"):
        value = columns.get(key)
        if value:
            return str(value)
    return None

def _recency_bonus(timestamp: str) -> float:
    """
    Soft recency prior in [0, 1], for lexical search tie-breaking.
    """
    ts = str(timestamp).strip()
    if not ts:
        return 0.0

    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(ts[:10], fmt)
            age_days = max((datetime.utcnow() - dt).days, 0)
            return max(0.0, 1.0 - min(age_days / 3650.0, 1.0))
        except Exception:
            continue

    normalized = re.sub(r"Z$", "+00:00", ts)
    try:
        dt = datetime.fromisoformat(normalized)
    except Exception:
        return 0.0
    age_days = max((datetime.utcnow() - dt.replace(tzinfo=None)).days, 0)
    return max(0.0, 1.0 - min(age_days / 3650.0, 1.0))

__all__ = [
    "OpenVikingClient",
    "OpenVikingMemoryStore",
    "OpenVikingTableRef",
    "InMemoryOpenVikingClient",
]
