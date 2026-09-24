"""Prompt templates for the MemoryArena agent loop."""

from __future__ import annotations

_DOMAIN_HINTS = {
    "bundled_shopping": (
        "Solve the webshop bundle one product step at a time. Preserve prior "
        "purchases and compatibility notes across sessions. For each subtask, "
        "select the target product and include its ASIN in the final answer."
    ),
    "progressive_search": (
        "Accumulate constraints across sessions. Each subtask may introduce a "
        "new condition, and the final answer must satisfy all conditions from "
        "the previous search steps. In the paper-faithful setting, use the "
        "search environment for evidence and preserve cited evidence in the "
        "trace. End with `Exact Answer: <answer>`."
    ),
    "group_travel_planner": (
        "Revise the shared group trip plan while preserving compatible choices "
        "from earlier travelers. Return only compact JSON for the full updated "
        "daily plan, using the fields from the background plan."
    ),
}

def build_prompt(
    *,
    question: str,
    overall_question: str | None = None,
    background: str,
    memory_context: str,
    domain: str,
) -> str:
    hint = _DOMAIN_HINTS.get(domain, "Solve the current subtask using memory if helpful.")
    bg = background.strip() or "None"
    mem = memory_context.strip() or "None (this is the first session of the task)."
    overall = (overall_question or "").strip()
    overall_block = (
        "Original full question / task goal:\n"
        f"{overall}\n\n"
        if overall and overall != question.strip()
        else ""
    )
    return (
        "You are solving a multi-session MemoryArena task.\n\n"
        f"Domain: {domain}\n"
        f"Domain instructions: {hint}\n\n"
        f"{overall_block}"
        "Background:\n"
        f"{bg}\n\n"
        "Memory from previous sessions:\n"
        f"{mem}\n\n"
        "Current subtask:\n"
        f"{question}\n\n"
        "Instructions:\n"
        "0. Do not include hidden reasoning, visible chain-of-thought, `<think>` blocks, or explanation outside the requested final answer.\n"
        "1. Use memory only when relevant.\n"
        "2. Solve the current subtask using evidence that satisfies all stated constraints.\n"
        "3. Prefer directly supported evidence over unsupported assumptions.\n"
        "4. For group travel planning, return the complete revised plan as JSON and preserve unchanged slots from the existing plan.\n"
        "5. For web-search tasks, end with exactly one concise line: Exact Answer: <answer>."
        "\n6. For bundled shopping, include exactly one ASIN for the selected current product."
    )
