"""
Faithful port of the official AMA-Agent method from AMA-Bench.

Source: datasets/amabench/code/src/method/ama_agent.py + ama_agent_core/
Paper: https://arxiv.org/abs/2602.22769

This package replicates the official algorithm exactly:
  1. construct_state_memory() — chunk-level LLM compression with **STATE_MEMORY** marker
  2. memory_retrieve() — two-stage LLM retrieval (state sufficiency → chunk relevance scoring)
"""

from agentmem.retrieval.ama_agent.construct import construct_state_memory
from agentmem.retrieval.ama_agent.retrieve import memory_retrieve

__all__ = ["construct_state_memory", "memory_retrieve"]
