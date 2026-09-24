"""Shared helpers for upstream memory-system adapters."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

def count_tokens(text: Any, token_counter: Any | None = None) -> int:
    if token_counter is not None:
        try:
            return int(token_counter(str(text or "")))
        except Exception:
            return 0
    return len(str(text or "").split())

def split_trajectory_text(
    text: str,
    *,
    chunk_chars: int = 2400,
    overlap_chars: int = 200,
    max_chunks: int | None = None,
) -> list[str]:
    """Split a flat trajectory into reasonably stable memory-note chunks."""
    content = str(text or "").strip()
    if not content:
        return []

    blocks = [
        block.strip()
        for block in re.split(r"\n(?=(?:#\s*)?(?:Session|Turn|Step)\s+\w+[:\s])", content)
        if block.strip()
    ]
    if len(blocks) <= 1:
        blocks = [content]

    chunks: list[str] = []
    current = ""
    for block in blocks:
        if not current:
            current = block
        elif len(current) + len(block) + 2 <= chunk_chars:
            current += "\n\n" + block
        else:
            chunks.extend(_hard_wrap(current, chunk_chars=chunk_chars, overlap_chars=overlap_chars))
            current = block
    if current:
        chunks.extend(_hard_wrap(current, chunk_chars=chunk_chars, overlap_chars=overlap_chars))

    if max_chunks is not None and max_chunks > 0:
        return chunks[:max_chunks]
    return chunks

def _hard_wrap(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    step = max(1, chunk_chars - min(overlap_chars, chunk_chars - 1))
    return [text[i : i + chunk_chars].strip() for i in range(0, len(text), step) if text[i : i + chunk_chars].strip()]

def format_mapping_items(items: Iterable[Mapping[str, Any]], *, title: str = "Retrieved Memories") -> str:
    lines = [f"# {title}"]
    count = 0
    for index, item in enumerate(items, start=1):
        content = str(item.get("content") or item.get("knowledge") or item.get("text") or "").strip()
        if not content:
            continue
        count += 1
        metadata = []
        for key in ("timestamp", "context", "category", "tags", "keywords"):
            value = item.get(key)
            if value:
                metadata.append(f"{key}={value}")
        suffix = f" ({'; '.join(metadata)})" if metadata else ""
        lines.append(f"[{index}]{suffix}\n{content}")
    if count == 0:
        return ""
    return "\n\n".join(lines)
