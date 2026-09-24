"""UnifiedIndex — multi-path retrieval from a shared memory store.

Implements four retrieval paths inspired by SimpleMem's three-path architecture
plus a lightweight graph path:

1. **Semantic** (R_sem): Dense embedding cosine similarity
2. **Lexical** (R_lex): BM25 keyword matching
3. **Symbolic** (R_sym): Structured metadata filtering (entities, timestamps)
4. **Graph** (R_graph): Lightweight KG + PersonalizedPageRank (HippoRAG-inspired)

The agent can select one path or combine multiple via union.
Paths 1-3 are cheap at both index and query time.
Path 4 requires one-time KG construction (cached via IndexDiskCache).

Also supports the original C1-C9 configs for oracle labeling / evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from agentmem.retrieval.base import (
    Compositor,
    Document,
    Index,
    IndexUnit,
    Retriever,
    RetrievalConfig,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

STRATEGY_PATHS = ["semantic", "lexical", "symbolic", "graph"]

STRATEGY_PATH_DESCRIPTIONS: Dict[str, str] = {
    "semantic": "Dense embedding similarity — best for paraphrased or conceptual queries",
    "lexical": "BM25 keyword matching — best for exact term matches and factual lookups",
    "symbolic": "Metadata filtering by entities/timestamps — best for temporal or entity-constrained queries",
    "graph": "Lightweight KG + PageRank traversal — best for multi-hop and entity-relation queries",
}

STRATEGY_PATH_COST_TIERS: Dict[str, int] = {
    "semantic": 1,                          
    "lexical": 1,                       
    "symbolic": 1,                         
    "graph": 2,                                                           
}

DEFAULT_CONFIGS = ["C1", "C3", "C4", "C5", "C7", "C8"]

STRATEGY_COST_TIERS: Dict[str, int] = {
    "C1": 1, "C2": 1, "C3": 2, "C9": 2,
    "C4": 3, "C5": 3,
    "C6": 4, "C7": 4, "C8": 4,

    **STRATEGY_PATH_COST_TIERS,
}

STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    "C1": "BM25 keyword retrieval — fast, good for exact term matches and single-hop factual lookups",
    "C3": "Dense embedding retrieval — semantic similarity, good for paraphrased or conceptual queries",
    "C4": "LLM-compressed memory + dense retrieval — captures temporal and summary information well",
    "C5": "Hybrid (dense + BM25 fusion) — balanced, combines keyword precision with semantic recall",
    "C7": "Multi-hop graph traversal — proposition extraction + concept graph, good for multi-hop reasoning",
    "C8": "Knowledge graph + PageRank — entity-centric graph retrieval, best for entity-relation queries",
    **STRATEGY_PATH_DESCRIPTIONS,
}

_TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"                           
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}(?:,?\s+\d{4})?\b"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*(?:\s+\d{4})?\b"
    r"|\b(?:yesterday|today|tomorrow|last\s+(?:week|month|year)|this\s+(?:morning|afternoon|evening))\b",
    re.IGNORECASE,
)

_PERSON_PATTERN = re.compile(
    r"\b(?:User|Speaker|Person|Alice|Bob|Charlie|David|Eve|Frank|Grace|"
    r"He|She|They|I|You|We)\b"
    r"|(?<=[.!?]\s)[A-Z][a-z]+(?:\s[A-Z][a-z]+)+"                                
)

def _extract_entities_lightweight(text: str) -> Dict[str, Set[str]]:
    """Extract entities and timestamps from text using regex (no LLM/spaCy)."""
    entities: Dict[str, Set[str]] = {"persons": set(), "timestamps": set(), "other": set()}

    for m in _TIMESTAMP_PATTERN.finditer(text):
        entities["timestamps"].add(m.group().strip())

    for m in _PERSON_PATTERN.finditer(text):
        name = m.group().strip()
        if name and len(name) > 1:
            entities["persons"].add(name)

    words = text.split()
    for i, w in enumerate(words):
        if (
            len(w) > 2
            and w[0].isupper()
            and not w.isupper()
            and i > 0
            and not words[i - 1].endswith((".", "!", "?"))
        ):
            entities["other"].add(w.strip(".,;:!?\"'()"))

    return entities

def _build_lightweight_kg(
    units: List[IndexUnit],
) -> Tuple[Dict[str, Set[str]], Dict[str, List[int]]]:
    """Build a lightweight entity co-occurrence graph from index units.

    Returns:
        entity_to_units: entity -> set of unit indices it appears in
        unit_entities: unit_idx -> list of entities in that unit
    """
    entity_to_units: Dict[str, Set[int]] = {}
    unit_entities: Dict[int, List[str]] = {}

    for idx, unit in enumerate(units):
        ents = _extract_entities_lightweight(unit.content)
        all_ents = ents["persons"] | ents["timestamps"] | ents["other"]
        unit_entities[idx] = list(all_ents)
        for ent in all_ents:
            ent_lower = ent.lower()
            entity_to_units.setdefault(ent_lower, set()).add(idx)

    return entity_to_units, unit_entities

@dataclass
class FourPathIndex:
    """Shared memory store supporting 4 retrieval paths from the same data.

    All paths operate on the same set of IndexUnits (chunked from documents).
    Paths 1-3 are cheap. Path 4 builds a lightweight entity graph (no LLM).
    """

    units: List[IndexUnit] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)

    _bm25_retriever: Optional[Any] = field(default=None, repr=False)
    _dense_retriever: Optional[Any] = field(default=None, repr=False)
    _bm25_index: Optional[Index] = field(default=None, repr=False)
    _dense_index: Optional[Index] = field(default=None, repr=False)

    _entity_to_units: Optional[Dict[str, Set[int]]] = field(default=None, repr=False)
    _unit_entities: Optional[Dict[int, List[str]]] = field(default=None, repr=False)

    _unit_metadata: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)

    build_times: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        documents: List[Document],
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[str] = None,
    ) -> "FourPathIndex":
        """Build all 4 paths from documents.

        Args:
            documents: Raw documents to index.
            chunk_size: Token count per chunk.
            chunk_overlap: Overlap between chunks.
            embedding_model: Model for dense embeddings.
            cache_dir: If set, cache the KG index to disk.
        """
        from agentmem.retrieval.indexing.chunker import FixedSizeChunker
        from agentmem.retrieval.retrievers.bm25 import BM25Retriever
        from agentmem.retrieval.retrievers.dense import DenseRetriever

        result = cls(documents=list(documents))

        t0 = time.monotonic()
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        chunk_index = chunker.build(documents)
        result.units = chunk_index.units
        result.build_times["chunking"] = time.monotonic() - t0

        t0 = time.monotonic()
        result._dense_retriever = DenseRetriever(embedding_model=embedding_model)
        result._dense_index = chunk_index
        result.build_times["semantic"] = time.monotonic() - t0

        t0 = time.monotonic()
        result._bm25_retriever = BM25Retriever(remove_stopwords=True)
        result._bm25_index = chunk_index
        result.build_times["lexical"] = time.monotonic() - t0

        t0 = time.monotonic()
        result._unit_metadata = []
        for unit in result.units:
            ents = _extract_entities_lightweight(unit.content)
            result._unit_metadata.append({
                "persons": list(ents["persons"]),
                "timestamps": list(ents["timestamps"]),
                "entities": list(ents["other"]),
            })
        result.build_times["symbolic"] = time.monotonic() - t0

        t0 = time.monotonic()
        result._entity_to_units, result._unit_entities = _build_lightweight_kg(result.units)
        result.build_times["graph"] = time.monotonic() - t0

        total = sum(v for v in result.build_times.values() if v > 0)
        logger.info(
            "FourPathIndex built: %d units, %.2fs total (sem=%.2f, lex=%.2f, sym=%.2f, graph=%.2f)",
            len(result.units), total,
            result.build_times.get("semantic", 0),
            result.build_times.get("lexical", 0),
            result.build_times.get("symbolic", 0),
            result.build_times.get("graph", 0),
        )
        return result

    def retrieve(
        self,
        path: str,
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Retrieve using a specific path.

        Args:
            path: One of "semantic", "lexical", "symbolic", "graph".
            query: The search query.
            k: Number of results to return.
        """
        path = path.lower()
        if path == "semantic":
            return self._retrieve_semantic(query, k)
        elif path == "lexical":
            return self._retrieve_lexical(query, k)
        elif path == "symbolic":
            return self._retrieve_symbolic(query, k, **kwargs)
        elif path == "graph":
            return self._retrieve_graph(query, k)
        else:
            raise ValueError(f"Unknown path '{path}'. Use: {STRATEGY_PATHS}")

    def retrieve_multi(
        self,
        paths: List[str],
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Retrieve using multiple paths and merge via union (SimpleMem-style).

        Deduplicates by content, keeps the highest score per unique content.
        """
        seen: Dict[str, RetrievalResult] = {}
        for path in paths:
            results = self.retrieve(path, query, k=k, **kwargs)
            for r in results:
                key = r.content[:200]                                 
                if key not in seen or r.score > seen[key].score:
                    seen[key] = r
        merged = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        return merged[:k]

    def retrieve_all(
        self,
        query: str,
        k: int = 10,
    ) -> Dict[str, List[RetrievalResult]]:
        """Query all 4 paths (for oracle labeling)."""
        results: Dict[str, List[RetrievalResult]] = {}
        for path in STRATEGY_PATHS:
            try:
                results[path] = self.retrieve(path, query, k=k)
            except Exception as exc:
                logger.warning("retrieve_all: %s failed: %s", path, exc)
        return results

    def _retrieve_semantic(self, query: str, k: int) -> List[RetrievalResult]:
        assert self._dense_retriever is not None and self._dense_index is not None
        return self._dense_retriever.retrieve(query, self._dense_index, k=k)

    def _retrieve_lexical(self, query: str, k: int) -> List[RetrievalResult]:
        assert self._bm25_retriever is not None and self._bm25_index is not None
        return self._bm25_retriever.retrieve(query, self._bm25_index, k=k)

    def _retrieve_symbolic(
        self, query: str, k: int, **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Filter units by entity/timestamp overlap with query."""
        if self._unit_metadata is None:
            return []

        query_ents = _extract_entities_lightweight(query)
        query_all = {e.lower() for e in query_ents["persons"] | query_ents["timestamps"] | query_ents["other"]}

        query_words = {w.lower().strip(".,;:!?\"'()") for w in query.split() if len(w) > 2}
        query_all |= query_words

        scored: List[Tuple[int, float]] = []
        for idx, meta in enumerate(self._unit_metadata):
            unit_ents = {e.lower() for e in meta["persons"] + meta["timestamps"] + meta["entities"]}
            overlap = len(query_all & unit_ents)
            if overlap > 0:
                score = overlap / max(len(query_all), 1)
                scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scored[:k]:
            results.append(RetrievalResult(
                doc_id=self.units[idx].source_doc_id or "",
                content=self.units[idx].content,
                score=score,
                metadata=self._unit_metadata[idx],
            ))
        return results

    def _retrieve_graph(self, query: str, k: int) -> List[RetrievalResult]:
        """Multi-hop graph retrieval via entity co-occurrence.

        1. Extract entities from query
        2. Find units containing those entities (seed set)
        3. Expand 1-hop: find entities in seed units -> find more units
        4. Score by entity overlap depth
        """
        if self._entity_to_units is None or self._unit_entities is None:
            return []

        query_ents = _extract_entities_lightweight(query)
        seed_entities = {
            e.lower()
            for e in query_ents["persons"] | query_ents["timestamps"] | query_ents["other"]
        }

        query_words = {w.lower().strip(".,;:!?\"'()") for w in query.split() if len(w) > 2}
        seed_entities |= query_words

        unit_scores: Dict[int, float] = {}
        hop0_units: Set[int] = set()
        for ent in seed_entities:
            for uid in self._entity_to_units.get(ent, set()):
                unit_scores[uid] = unit_scores.get(uid, 0) + 1.0
                hop0_units.add(uid)

        hop1_entities: Set[str] = set()
        for uid in hop0_units:
            for ent in self._unit_entities.get(uid, []):
                hop1_entities.add(ent.lower())
        hop1_entities -= seed_entities                     

        for ent in hop1_entities:
            for uid in self._entity_to_units.get(ent, set()):
                if uid not in hop0_units:
                    unit_scores[uid] = unit_scores.get(uid, 0) + 0.5              

        if not unit_scores:
            return []

        max_score = max(unit_scores.values())
        scored = sorted(unit_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored[:k]:
            results.append(RetrievalResult(
                doc_id=self.units[idx].source_doc_id or "",
                content=self.units[idx].content,
                score=score / max_score if max_score > 0 else 0,
                metadata={"hop_score": score},
            ))
        return results

    @property
    def available_paths(self) -> List[str]:
        return list(STRATEGY_PATHS)

    def format_paths_for_prompt(self) -> str:
        """Format available paths for the agent's system prompt."""
        lines = []
        for path in STRATEGY_PATHS:
            desc = STRATEGY_PATH_DESCRIPTIONS[path]
            tier = STRATEGY_PATH_COST_TIERS[path]
            cost_label = {1: "cheap", 2: "medium"}[tier]
            lines.append(f"- **{path}** ({cost_label}): {desc}")
        return "\n".join(lines)

    @staticmethod
    def cost_tier(path: str) -> int:
        return STRATEGY_PATH_COST_TIERS.get(path.lower(), 1)

@dataclass
class UnifiedIndex:
    """Pre-built C1-C9 sub-indexes, queryable by config name.

    Used for oracle labeling and evaluation. For RL training, prefer
    FourPathIndex which is much cheaper to build.
    """

    sub_indexes: Dict[str, Index] = field(default_factory=dict)
    sub_retrievers: Dict[str, Retriever] = field(default_factory=dict)
    sub_compositors: Dict[str, Optional[Compositor]] = field(default_factory=dict)
    sub_configs: Dict[str, RetrievalConfig] = field(default_factory=dict)
    documents: List[Document] = field(default_factory=list)
    build_times: Dict[str, float] = field(default_factory=dict)
    config_names: List[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        documents: List[Document],
        config_names: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        **shared_params: Any,
    ) -> "UnifiedIndex":
        """Build all sub-indexes from the same document set.

        Args:
            documents: Raw documents to index.
            config_names: Which configs to build (default: DEFAULT_CONFIGS).
            cache_dir: If set, use IndexDiskCache to avoid redundant builds.
            **shared_params: Forwarded to build_config() (e.g., llm_model, api_key).
        """
        from agentmem.retrieval.configs import build_config

        if config_names is None:
            config_names = list(DEFAULT_CONFIGS)

        cache = None
        if cache_dir:
            from agentmem.retrieval.cache import IndexDiskCache
            cache = IndexDiskCache(cache_dir=cache_dir)

        result = cls(
            documents=list(documents),
            config_names=list(config_names),
        )

        for name in config_names:
            t0 = time.monotonic()
            try:
                config = build_config(name, **shared_params)

                if cache is not None:
                    cache_key = cache.make_key(
                        type(config.index_builder).__name__,
                        {k: str(v) for k, v in config.params.items() if k != "description"},
                        documents,
                    )
                    index = cache.get_or_build(
                        cache_key, lambda: config.index_builder.build(documents)
                    )
                else:
                    index = config.index_builder.build(documents)

                result.sub_indexes[name] = index
                result.sub_configs[name] = config
                if config.retriever is not None:
                    result.sub_retrievers[name] = config.retriever
                if config.compositor is not None:
                    result.sub_compositors[name] = config.compositor
                result.build_times[name] = time.monotonic() - t0
                logger.info("Built %s index in %.2fs", name, result.build_times[name])
            except Exception as exc:
                logger.warning("Failed to build %s index: %s", name, exc)
                result.build_times[name] = -1.0

        return result

    def retrieve(
        self, config_name: str, query: str, k: int = 10,
    ) -> List[RetrievalResult]:
        """Query a specific sub-index by config name."""
        name = config_name.upper()
        if name not in self.sub_indexes:
            raise ValueError(f"Config '{name}' not built. Available: {list(self.sub_indexes)}")
        index = self.sub_indexes[name]
        compositor = self.sub_compositors.get(name)
        if compositor is not None:
            return compositor.compose(query, index, k=k)
        retriever = self.sub_retrievers.get(name)
        if retriever is not None:
            return retriever.retrieve(query, index, k=k)
        raise ValueError(f"Config '{name}' has no retriever or compositor")

    def retrieve_all(
        self, query: str, k: int = 10,
    ) -> Dict[str, List[RetrievalResult]]:
        """Query ALL sub-indexes (for oracle labeling)."""
        results: Dict[str, List[RetrievalResult]] = {}
        for name in self.sub_indexes:
            try:
                results[name] = self.retrieve(name, query, k=k)
            except Exception as exc:
                logger.warning("retrieve_all: %s failed: %s", name, exc)
        return results

    @property
    def available_strategies(self) -> List[str]:
        return [n for n in self.config_names if n in self.sub_indexes]

    def format_strategies_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.available_strategies):
            desc = STRATEGY_DESCRIPTIONS.get(name, self.sub_configs[name].params.get("description", ""))
            tier = STRATEGY_COST_TIERS.get(name, 2)
            cost_label = {1: "cheap", 2: "medium", 3: "expensive", 4: "very expensive"}[tier]
            lines.append(f"- **{name}** ({cost_label}): {desc}")
        return "\n".join(lines)

    @staticmethod
    def cost_tier(config_name: str) -> int:
        return STRATEGY_COST_TIERS.get(config_name.upper(), 2)

    @staticmethod
    def cache_key(documents: List[Document], config_names: List[str]) -> str:
        content_hash = hashlib.sha256()
        for doc in sorted(documents, key=lambda d: d.id):
            content_hash.update(doc.id.encode())
            content_hash.update(doc.content.encode())
        content_hash.update(",".join(sorted(config_names)).encode())
        return content_hash.hexdigest()[:16]

def build_all_configs(**shared_params: Any) -> Dict[str, RetrievalConfig]:
    """Build all default retrieval configs (convenience wrapper)."""
    from agentmem.retrieval.configs import build_config

    configs = {}
    for name in DEFAULT_CONFIGS:
        try:
            configs[name] = build_config(name, **shared_params)
        except Exception as exc:
            logger.warning("build_all_configs: %s failed: %s", name, exc)
    return configs
