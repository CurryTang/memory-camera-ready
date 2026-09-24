"""
build_config factory — creates RetrievalConfig instances for C1-C8.

Each configuration is a composition of IndexBuilder + Retriever + optional Compositor.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from agentmem.retrieval.base import RetrievalConfig
from agentmem.retrieval.indexing.plain import PlainTextIndex
from agentmem.retrieval.indexing.chunker import FixedSizeChunker
from agentmem.retrieval.retrievers.bm25 import BM25Retriever
from agentmem.retrieval.retrievers.substring import SubstringMatcher
from agentmem.retrieval.retrievers.dense import DenseRetriever

def build_config(name: str, **params: Any) -> RetrievalConfig:
    """
    Build a RetrievalConfig by name.

    Supported names: C1, C2, C3, C4, C5, C6, C7, C8
    (case-insensitive).

    Extra **params are merged into the config params dict and forwarded
    to constructors where applicable.
    """
    name_upper = name.upper()

    if name_upper == "C1":
        return RetrievalConfig(
            index_builder=PlainTextIndex(),
            retriever=BM25Retriever(
                remove_stopwords=params.get("remove_stopwords", True)
            ),
            compositor=None,
            params={"description": "Plain text, BM25 retrieval", **params},
        )

    if name_upper == "C2":
        return RetrievalConfig(
            index_builder=PlainTextIndex(),
            retriever=SubstringMatcher(),
            compositor=None,
            params={"description": "Plain text, keyword substring match", **params},
        )

    if name_upper == "C3":
        chunk_size = int(params.get("chunk_size", 512))
        overlap = int(params.get("overlap", 64))
        embedding_model = params.get("embedding_model", "BAAI/bge-small-en-v1.5")
        return RetrievalConfig(
            index_builder=FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
            retriever=DenseRetriever(embedding_model=embedding_model),
            compositor=None,
            params={
                "description": "Fixed-size chunks, dense embedding retrieval",
                **params,
            },
        )

    if name_upper == "C4":
        from agentmem.retrieval.indexing.compressor import LLMCompressor                
        return RetrievalConfig(
            index_builder=LLMCompressor(**{k: v for k, v in params.items() if k in LLMCompressor._init_params()}),
            retriever=DenseRetriever(
                embedding_model=params.get("embedding_model", "BAAI/bge-small-en-v1.5")
            ),
            compositor=None,
            params={"description": "LLM-compressed memory, dense retrieval", **params},
        )

    if name_upper == "C5":
        from agentmem.retrieval.indexing.compressor import LLMCompressor                
        from agentmem.retrieval.fusion.score_fusion import ScoreFusion
        embedding_model = params.get("embedding_model", "BAAI/bge-small-en-v1.5")
        dense = DenseRetriever(embedding_model=embedding_model)
        bm25 = BM25Retriever()
        compositor = ScoreFusion(
            retrievers=[(dense, 0.5), (bm25, 0.5)],
            mode=params.get("fusion_mode", "rrf"),
        )
        return RetrievalConfig(
            index_builder=LLMCompressor(**{k: v for k, v in params.items() if k in LLMCompressor._init_params()}),
            retriever=None,
            compositor=compositor,
            params={"description": "LLM-compressed, multi-view (dense+BM25) with RRF", **params},
        )

    if name_upper == "C6":
        from agentmem.retrieval.indexing.causal_graph import CausalGraphBuilder                
        from agentmem.retrieval.indexing.state_memory import StateMemoryBuilder                
        from agentmem.retrieval.retrievers.graph_reasoner import LLMGraphReasoner                

        llm_model = params.get("llm_model", "gpt-4o-mini")
        llm_api_key = params.get("llm_api_key")

        class _C6DualIndexBuilder:
            """Builds CausalGraph + StateMemory indexes in parallel; exposes both."""
            def __init__(self, model: str, api_key: Any) -> None:
                self._causal = CausalGraphBuilder(model=model, api_key=api_key)
                self._state = StateMemoryBuilder(model=model, api_key=api_key)
                self.state_index = None                     

            def build(self, documents: list) -> Any:
                self.state_index = self._state.build(documents)
                return self._causal.build(documents)

        dual_builder = _C6DualIndexBuilder(model=llm_model, api_key=llm_api_key)
        return RetrievalConfig(
            index_builder=dual_builder,
            retriever=LLMGraphReasoner(
                model=llm_model,
                api_key=llm_api_key,
            ),
            compositor=None,
            params={
                "description": (
                    "Causal graph (CausalGraphBuilder) + State memory (StateMemoryBuilder) "
                    "+ LLM graph reasoner (AMA-Agent Option A)"
                ),
                "builders": ["CausalGraphBuilder", "StateMemoryBuilder"],
                "retrievers": ["LLMGraphReasoner"],
                **params,
            },
        )

    if name_upper == "C7":
        from agentmem.retrieval.indexing.proposition import PropositionExtractor                
        from agentmem.retrieval.indexing.concept_graph import ConceptGraphBuilder                
        from agentmem.retrieval.retrievers.multi_hop import MultiHopTraverser                

        class _C7PipelineIndexBuilder:
            """Chains PropositionExtractor → ConceptGraphBuilder."""
            def __init__(self, model: str, api_key: Any) -> None:
                self._prop = PropositionExtractor(model=model, api_key=api_key)
                self._concept = ConceptGraphBuilder()

            def build(self, documents: list) -> Any:
                prop_index = self._prop.build(documents)
                return self._concept.build_from_index(prop_index)

        llm_model = params.get("llm_model", "gpt-4o-mini")
        llm_api_key = params.get("llm_api_key")
        max_hops = int(params.get("max_hops", 2))
        return RetrievalConfig(
            index_builder=_C7PipelineIndexBuilder(model=llm_model, api_key=llm_api_key),
            retriever=MultiHopTraverser(max_hops=max_hops),
            compositor=None,
            params={
                "description": "Proposition extraction + concept graph, multi-hop traversal",
                "builders": ["PropositionExtractor", "ConceptGraphBuilder"],
                "retrievers": ["MultiHopTraverser"],
                **params,
            },
        )

    if name_upper == "C8":
        from agentmem.retrieval.indexing.knowledge_graph import KnowledgeGraphBuilder                
        from agentmem.retrieval.retrievers.pagerank import PersonalizedPageRank                

        llm_model = params.get("llm_model", "gpt-4o-mini")
        llm_api_key = params.get("llm_api_key")
        return RetrievalConfig(
            index_builder=KnowledgeGraphBuilder(model=llm_model, api_key=llm_api_key),
            retriever=PersonalizedPageRank(),
            compositor=None,
            params={
                "description": "KG (OpenIE triples) + PersonalizedPageRank (HippoRAG2-inspired)",
                "builders": ["KnowledgeGraphBuilder"],
                "retrievers": ["PersonalizedPageRank"],
                **params,
            },
        )

    if name_upper == "C9":
        from agentmem.retrieval.retrievers.colbert import ColBERTRetriever                

        model_name = params.get("model_name", "colbert-ir/colbertv2.0")
        chunk_size = int(params.get("chunk_size", 256))
        overlap = int(params.get("overlap", 32))
        return RetrievalConfig(
            index_builder=FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
            retriever=ColBERTRetriever(
                model_name=model_name,
                max_doc_length=params.get("max_doc_length", 256),
                max_query_length=params.get("max_query_length", 32),
                batch_size=params.get("batch_size", 32),
            ),
            compositor=None,
            params={
                "description": (
                    "Fixed-size chunks + ColBERT late interaction (MaxSim token-level scoring)"
                ),
                **params,
            },
        )

    raise ValueError(
        f"Unknown retrieval config name: '{name}'. "
        "Supported: C1, C2, C3, C4, C5, C6, C7, C8, C9."
    )
