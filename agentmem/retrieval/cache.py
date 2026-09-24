"""
IndexDiskCache — transparent disk-based cache for built Index objects.

Avoids redundant index building when multiple configs share the same
IndexBuilder (e.g. C1/C2 both use PlainTextIndex; C4/C5 both use
LLMCompressor). The caller is responsible for computing a stable cache key
that captures the builder type, its params, and the input documents.

Usage::

    cache = IndexDiskCache(cache_dir="results/index_cache")
    key = cache.make_key("PlainTextIndex", params={}, documents=docs)
    index = cache.get_or_build(key, builder_fn=lambda: plain_index.build(docs))
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

from agentmem.retrieval.base import Document, Index

logger = logging.getLogger(__name__)

def _docs_fingerprint(documents: list[Document]) -> str:
    """Stable hash of document ids + content (order-independent)."""
    h = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: d.id):
        h.update(doc.id.encode())
        h.update(doc.content.encode())
    return h.hexdigest()[:16]

def _params_fingerprint(params: dict[str, Any]) -> str:
    """Stable hash of a flat params dict (string values only)."""
    h = hashlib.sha256()
    for k in sorted(params.keys()):
        h.update(str(k).encode())
        h.update(str(params[k]).encode())
    return h.hexdigest()[:12]

class IndexDiskCache:
    """
    Disk-based cache for Index objects.

    Cache files are stored as ``<cache_dir>/<key>.pkl``.  Hits avoid
    calling the (potentially expensive) builder function entirely.

    Args:
        cache_dir: Directory for cache files. Created on first use.
        enabled: Set to False to disable caching (all calls go to builder).
    """

    def __init__(
        self,
        cache_dir: str | Path = "results/index_cache",
        enabled: bool = True,
    ) -> None:
        self._dir = Path(cache_dir)
        self._enabled = enabled
        self._hits = 0
        self._misses = 0

    def make_key(
        self,
        builder_name: str,
        params: dict[str, Any],
        documents: list[Document],
    ) -> str:
        """Compute a stable cache key from builder name, params, and documents.

        Args:
            builder_name: Class name of the IndexBuilder (e.g. "LLMCompressor").
            params: Dict of constructor-level params that affect the output
                (e.g. {"model": "gpt-4o-mini", "density_threshold": 0.3}).
                Do NOT include api_key or base_url — they don't affect content.
            documents: The exact documents that will be passed to build().

        Returns:
            A short alphanumeric key suitable for use as a filename.
        """
        pf = _params_fingerprint(params)
        df = _docs_fingerprint(documents)

        safe_name = "".join(c if c.isalnum() else "_" for c in builder_name)
        return f"{safe_name}_{pf}_{df}"

    def get_or_build(
        self,
        key: str,
        builder_fn: Callable[[], Index],
    ) -> Index:
        """Return cached Index if available, else call builder_fn and cache result.

        Args:
            key: Cache key from make_key().
            builder_fn: Zero-argument callable that produces the Index.

        Returns:
            Index from cache or freshly built.
        """
        if not self._enabled:
            return builder_fn()

        self._dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._dir / f"{key}.pkl"

        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    index: Index = pickle.load(f)
                self._hits += 1
                logger.info("IndexDiskCache HIT  %s (%d units)", key, len(index.units))
                return index
            except Exception as exc:
                logger.warning(
                    "IndexDiskCache: failed to load %s (%s). Rebuilding.", key, exc
                )

        index = builder_fn()
        self._misses += 1
        logger.info("IndexDiskCache MISS %s (%d units). Saving.", key, len(index.units))
        try:
            with cache_path.open("wb") as f:
                pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.warning("IndexDiskCache: failed to save %s (%s).", key, exc)

        return index

    @property
    def stats(self) -> dict[str, int]:
        """Return hit/miss counts since creation."""
        return {"hits": self._hits, "misses": self._misses}
