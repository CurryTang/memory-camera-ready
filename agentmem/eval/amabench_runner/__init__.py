"""AMAbench runner — adapted from the official AMA-Bench evaluation code.

Key additions over the official runner:
- Per-episode JSONL checkpointing with automatic resume on restart.
- OpenAI-compatible API for both LLM and embedding (works with sglang, vllm,
  OpenRouter, etc.).
- Integrated into the agentmem project structure.
"""
