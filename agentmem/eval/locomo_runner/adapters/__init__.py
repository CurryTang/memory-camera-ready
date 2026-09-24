"""Adapter classes for LoCoMo evaluation.

All adapters follow the observe/finalize/ask pattern and use the same
_build_locomo_answer_prompt for consistent evaluation.
"""

from agentmem.eval.locomo_runner.adapters.ama_agent import AMAAgentLoCoMoAdapter
from agentmem.eval.locomo_runner.adapters.base import _BaselineLoCoMoAdapter
from agentmem.eval.locomo_runner.adapters.baselines import (
    C1LoCoMoAdapter,
    C2LoCoMoAdapter,
    C3LoCoMoAdapter,
    C4LoCoMoAdapter,
    C5LoCoMoAdapter,
    C6LoCoMoAdapter,
    C7LoCoMoAdapter,
    C8LoCoMoAdapter,
    C9LoCoMoAdapter,
    _make_compress_provider,
)
from agentmem.eval.locomo_runner.adapters.longcontext import LongContextLoCoMoAdapter
from agentmem.eval.locomo_runner.adapters.hipporagv2 import HippoRAGv2LoCoMoAdapter
from agentmem.eval.locomo_runner.adapters.memrl import MemRLLoCoMoAdapter
from agentmem.eval.locomo_runner.adapters.plugmem import PlugMemLoCoMoAdapter

__all__ = [
    "AMAAgentLoCoMoAdapter",
    "_BaselineLoCoMoAdapter",
    "C1LoCoMoAdapter",
    "C2LoCoMoAdapter",
    "C3LoCoMoAdapter",
    "C4LoCoMoAdapter",
    "C5LoCoMoAdapter",
    "C6LoCoMoAdapter",
    "C7LoCoMoAdapter",
    "C8LoCoMoAdapter",
    "C9LoCoMoAdapter",
    "LongContextLoCoMoAdapter",
    "HippoRAGv2LoCoMoAdapter",
    "MemRLLoCoMoAdapter",
    "PlugMemLoCoMoAdapter",
    "_make_compress_provider",
]
