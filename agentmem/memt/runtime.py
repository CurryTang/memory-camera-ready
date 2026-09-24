"""
Mem-T runtime adapter - thin wrapper around the vendored Mem-T codebase.

The original code is vendored at:
  vendor/memt/

This module provides:
  - MemTConfig: config dataclass matching original SystemConfig
  - MemTRuntimeEngine: wires up original MemoryBuilder + MemoryRetriever
  - MemTMemoryBank: thin handle to the ChromaDB state

All heavy lifting is done by the original code via direct imports.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_VENDOR_MEMT = Path(__file__).resolve().parents[2] / "vendor" / "memt"
_MEMT_MODULE_ORDER = (
    "config",
    "llm_stats",
    "utils",
    "trajectory_logger",
    "tools_base",
    "llm_api",
    "vector_db",
    "memory_formation",
    "memory_update",
    "memory_builder",
    "memory_retrieval",
    "dataset",
)

def _load_vendor_module(module_name: str) -> Any:
    module_path = _VENDOR_MEMT / f"{module_name}.py"
    if not module_path.exists():
        raise ImportError(
            f"Cannot import original Mem-T code: missing {module_path}. "
            "Ensure vendor/memt is present."
        )

    existing = sys.modules.get(module_name)
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == module_path:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import original Mem-T code: cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _load_vendor_modules() -> None:
    if not _VENDOR_MEMT.exists():
        raise ImportError(
            f"Cannot import original Mem-T code: vendor directory not found at {_VENDOR_MEMT}"
        )
    for module_name in _MEMT_MODULE_ORDER:
        _load_vendor_module(module_name)

_original_cache: dict | None = None

def _get_original_modules() -> dict:
    """Import original Mem-T modules lazily.

    Requires: chromadb, loguru, vllm (for local), openai.
    Install: pip install chromadb loguru
    """
    global _original_cache
    if _original_cache is not None:
        return _original_cache

    try:
        _load_vendor_modules()
        from config import SystemConfig, VectorDBConfig, MemoryConfig, LLMConfig
        from vector_db import VectorDBFactory
        from memory_formation import MemoryFormation
        from memory_update import MemoryUpdate
        from memory_builder import MemoryBuilder
        from memory_retrieval import MemoryRetriever
        from trajectory_logger import get_collector
    except ImportError as e:
        raise ImportError(
            f"Cannot import original Mem-T code: {e}\n"
            f"Ensure dependencies are installed: pip install chromadb loguru\n"
            f"Original code expected at: {_VENDOR_MEMT}"
        ) from e

    _original_cache = {
        "SystemConfig": SystemConfig,
        "VectorDBConfig": VectorDBConfig,
        "MemoryConfig": MemoryConfig,
        "LLMConfig": LLMConfig,
        "VectorDBFactory": VectorDBFactory,
        "MemoryFormation": MemoryFormation,
        "MemoryUpdate": MemoryUpdate,
        "MemoryBuilder": MemoryBuilder,
        "MemoryRetriever": MemoryRetriever,
        "get_collector": get_collector,
    }
    return _original_cache

@dataclass
class MemTConfig:
    """Adapter config that maps to original SystemConfig fields."""

    db_backend: str = "chroma"
    db_type: str = "http"                               
    db_path: str = "./database/"
    db_host: str = "localhost"
    db_port: int = 8070
    from_scratch: bool = False

    k_turns: int = 4
    update_retrieval_topk: int = 3
    retrieval_topk: int = 5
    max_tool_steps: int = 6
    max_context_tokens: int = 4096

    use_local_llm: bool = False
    local_model: str = "EdwinYue/Mem-T-4B"
    local_model_path: str = "models/EdwinYue/Mem-T-4B"
    strong_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 65536

    data_name: str = "locomo"
    dataset_path: str = ""
    mode: str = "test"

    use_parallel: bool = False
    num_workers: int = 1
    seed: int = 42

    traj_dir: str = "./traj/memt_run"
    log_path: str = "./logs/memt_run.log"

    def to_system_config(self) -> Any:
        """Convert to original SystemConfig."""
        mods = _get_original_modules()
        cfg = mods["SystemConfig"]()
        cfg.vector_db.backend = self.db_backend
        cfg.vector_db.db_type = self.db_type
        cfg.vector_db.path = self.db_path
        cfg.vector_db.host = self.db_host
        cfg.vector_db.port = self.db_port
        cfg.vector_db.from_scratch = self.from_scratch
        cfg.memory.summary_context_turns = self.k_turns
        cfg.memory.update_retrieval_topk = self.update_retrieval_topk
        cfg.memory.retrieval_topk = self.retrieval_topk
        cfg.memory.max_tool_steps = self.max_tool_steps
        cfg.memory.max_context_tokens = self.max_context_tokens
        cfg.llm.local_model = self.local_model
        cfg.llm.local_model_path = self.local_model_path
        cfg.llm.strong_model = self.strong_model
        cfg.llm.temperature = self.temperature
        cfg.llm.max_tokens = self.max_tokens
        cfg.USE_LOCAL_LLM = self.use_local_llm
        cfg.USE_PARALLEL = self.use_parallel
        cfg.NUM_WORKERS = self.num_workers
        cfg.seed = self.seed
        cfg.data_name = self.data_name
        cfg.dataset_path = self.dataset_path
        cfg.mode = self.mode
        cfg.traj_dir = self.traj_dir
        cfg.log_path = self.log_path
        return cfg

class MemTMemoryBank:
    """Handle to the ChromaDB-backed memory bank created by original code."""

    COLLECTIONS = ("turns", "facts", "experiences", "personas", "summary")

    def __init__(self, vector_db: Any = None, sample_id: str = ""):
        self.vector_db = vector_db
        self.sample_id = sample_id

    def collection_name(self, base: str) -> str:
        return f"{self.sample_id}_{base}"

    def search(self, collection_base: str, query: str, top_k: int = 5) -> dict:
        if self.vector_db is None:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        col_name = self.collection_name(collection_base)
        return self.vector_db.search(
            col_name, query_texts=[query], top_k=top_k,
            include=["documents", "metadatas"],
        )

    def get_counts(self) -> dict[str, int]:
        if self.vector_db is None:
            return {}
        counts = {}
        for base in self.COLLECTIONS:
            try:
                col_name = self.collection_name(base)
                if hasattr(self.vector_db, "_get_collection"):
                    counts[base] = int(self.vector_db._get_collection(col_name).count())
                else:
                    result = self.vector_db.get(col_name)
                    counts[base] = len(result.get("ids", []))
            except Exception:
                counts[base] = 0
        return counts

MemTCollections = MemTMemoryBank.COLLECTIONS

class MemTRuntimeEngine:
    """
    Thin adapter that wires up original Mem-T components.

    Usage:
        engine = MemTRuntimeEngine(config=MemTConfig(...))
        engine.build_from_dataset("locomo", "datasets/locomo/locomo10.json")
        result = engine.retrieve_and_answer(question, sample_id)
    """

    def __init__(self, config: Optional[MemTConfig] = None) -> None:
        self.config = config or MemTConfig()
        self._mods = _get_original_modules()
        self._sys_config = self.config.to_system_config()

        self._vector_db = self._mods["VectorDBFactory"].create_db(
            self._sys_config.vector_db
        )

        self._llm = self._create_llm_client()

        self._formation = self._mods["MemoryFormation"](llm_executor=self._llm)
        self._update = self._mods["MemoryUpdate"](
            llm_executor=self._llm, vector_db=self._vector_db,
        )
        self._builder = self._mods["MemoryBuilder"](
            vector_db=self._vector_db,
            formation=self._formation,
            update=self._update,
            config=self._sys_config,
        )
        self._retriever = self._mods["MemoryRetriever"](
            llm_executor=self._llm,
            vector_db=self._vector_db,
            config=self._sys_config,
        )
        self._collector = self._mods["get_collector"](self.config.traj_dir)

    def _create_llm_client(self) -> Any:
        if self.config.use_local_llm:
            from llm_api import VLLMClient
            model_path = self.config.local_model_path
            if not os.path.exists(model_path):
                model_path = self.config.local_model
            return VLLMClient(model_name_or_path=model_path)
        else:
            from llm_api import OpenAIAPIClient
            return OpenAIAPIClient(model=self.config.strong_model)

    def load_dataset(
        self, data_name: str = "locomo", dataset_path: str = "",
    ) -> tuple[list, list, list]:
        """Load and split dataset using original code. Returns (train, valid, test)."""
        from dataset import load_hotpotqa_dataset, load_locomo_dataset, train_valid_test_split

        if data_name == "locomo":
            chat_data = load_locomo_dataset(dataset_path)
            return [chat_data[0]], [chat_data[1]], chat_data[2:]
        elif data_name == "hotpotqa":
            chat_data = load_hotpotqa_dataset(dataset_path)
            return train_valid_test_split(chat_data, seed=self.config.seed)
        else:
            raise ValueError(f"Unsupported dataset: {data_name}")

    def build_from_sample(self, sample: dict[str, Any]) -> None:
        """Build memory database for one sample using original pipeline."""
        self._builder.build_from_sample(sample)

    def build_from_samples(self, samples: list[dict[str, Any]]) -> None:
        """Build memory database for multiple samples."""
        for sample in samples:
            self.build_from_sample(sample)

    def retrieve_and_answer(
        self, question: str, sample_id: str, category: str = "",
    ) -> dict[str, Any]:
        """Run multi-step retrieval and return answer + traces."""
        return self._retriever.retrieve_and_answer(
            question, sample_id=sample_id, category=category,
        )

    def run_eval(
        self,
        data_name: str = "locomo",
        dataset_path: str = "",
        mode: str = "test",
        build_db: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run the full Mem-T eval pipeline (build + retrieve + answer).
        Mirrors original main.py logic.

        Returns list of {sample_id, qa_id, question, pred, gold, traces, category}.
        """
        from trajectory_logger import QATrajectoryLog

        train, valid, test = self.load_dataset(data_name, dataset_path)
        if mode == "train":
            data = train
        elif mode == "valid":
            data = valid
        else:
            data = test

        results = []
        for i, sample in enumerate(data):
            sample_id = (
                sample["qa"][0].get("sample_id", f"sample_{i}")
                if sample.get("qa") else f"sample_{i}"
            )

            if build_db:
                self.build_from_sample(sample)

            for j, qa in enumerate(sample.get("qa", [])):
                question = qa["question"]
                gold = qa["answer"]
                evidence = qa.get("evidence", [])
                category = qa.get("category", "")

                result = self.retrieve_and_answer(question, sample_id, str(category))
                pred = result["answer"]
                traces = result["traces"]

                self._collector.log_qa_step(QATrajectoryLog(
                    sample_id=sample_id,
                    qa_id=f"{sample_id}_{j}",
                    question=question,
                    pred=pred,
                    gold=gold,
                    evidence=evidence,
                    traces=traces,
                    category=category,
                ))

                results.append({
                    "sample_id": sample_id,
                    "qa_id": f"{sample_id}_{j}",
                    "question": question,
                    "pred": pred,
                    "gold": gold,
                    "evidence": evidence,
                    "traces": traces,
                    "category": category,
                })

        return results

    def get_bank(self, sample_id: str) -> MemTMemoryBank:
        """Get a bank handle for a specific sample."""
        return MemTMemoryBank(vector_db=self._vector_db, sample_id=sample_id)

    @property
    def vector_db(self) -> Any:
        return self._vector_db

    @property
    def builder(self) -> Any:
        return self._builder

    @property
    def retriever(self) -> Any:
        return self._retriever
