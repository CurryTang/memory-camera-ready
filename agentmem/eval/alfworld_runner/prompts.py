"""Prompt builders for ALFWorld prompt-based agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from examples.agent_rl.prompts import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE_WITH_MEMORY,
)
from examples.agent_rl.trainable_agents import _format_action_history

_TASK_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
    "pick_two_obj_and_place": "puttwo",
    "look_at_obj_in_light": "examine",
}

def _format_admissible_actions(admissible_commands: list[str]) -> str:
    filtered = [str(cmd).strip() for cmd in admissible_commands if str(cmd).strip() and str(cmd).strip().lower() != "help"]
    return "\n ".join(f"'{cmd}'" for cmd in filtered)

def build_vanilla_prompt(
    observation: str,
    task_desc: str,
    admissible_commands: list[str],
    history: list[tuple[str, str]],
    memory_prompt: str = "",
    few_shot_prompt: str = "",
    max_history: int = 5,
) -> str:
    """Action-only baseline prompt without explicit ReAct reasoning."""
    admissible_actions = _format_admissible_actions(admissible_commands)
    lines = []
    if few_shot_prompt:
        lines.append(few_shot_prompt.strip())
    lines.append("You are an ALFWorld agent solving a household task.")
    if task_desc:
        lines.append(f"Task: {task_desc}")
    if memory_prompt:
        lines.append(memory_prompt.strip())
    if history:
        history_text, history_len = _format_action_history(history, max_recent=max_history)
        lines.append(f"Recent history ({history_len} step(s)):\n{history_text}")
    lines.append(f"Current observation: {observation}")
    lines.append(f"Admissible actions: [{admissible_actions}]")
    lines.append("Reply with exactly one admissible action and no extra commentary.")
    return "\n\n".join(line for line in lines if line)

def build_react_prompt(
    observation: str,
    task_desc: str,
    admissible_commands: list[str],
    history: list[tuple[str, str]],
    memory_prompt: str = "",
    few_shot_prompt: str = "",
    max_history: int = 5,
) -> str:
    """ReAct-style prompt aligned with the existing ALFWorld training templates."""
    prompt = _build_react_core_prompt(
        observation=observation,
        task_desc=task_desc,
        admissible_commands=admissible_commands,
        history=history,
        memory_prompt=memory_prompt,
        max_history=max_history,
    )
    if not few_shot_prompt:
        return prompt
    return f"{few_shot_prompt.rstrip()}\n\nHere is the current task.\n{prompt}"

def build_reflexion_prompt(
    observation: str,
    task_desc: str,
    admissible_commands: list[str],
    history: list[tuple[str, str]],
    reflections: list[str],
    memory_prompt: str = "",
    few_shot_prompt: str = "",
    max_history: int = 5,
) -> str:
    prompt = _build_react_core_prompt(
        observation=observation,
        task_desc=task_desc,
        admissible_commands=admissible_commands,
        history=history,
        memory_prompt=memory_prompt,
        max_history=max_history,
    )
    prefix_parts: list[str] = []
    if few_shot_prompt:
        prefix_parts.append(few_shot_prompt.rstrip())
    if reflections:
        reflection_text = "\n".join(
            f"Trial {idx + 1}: {text.strip()}"
            for idx, text in enumerate(reflections)
            if str(text or "").strip()
        )
        if reflection_text:
            prefix_parts.append(
                "Reflections from earlier failed attempts on the same task:\n"
                f"{reflection_text}\n"
                "Use these reflections to avoid repeating mistakes."
            )
    if not prefix_parts:
        return prompt
    return "\n\n".join(prefix_parts + ["Here is the current task.", prompt])

def _build_react_core_prompt(
    observation: str,
    task_desc: str,
    admissible_commands: list[str],
    history: list[tuple[str, str]],
    memory_prompt: str,
    max_history: int,
) -> str:
    admissible_actions = _format_admissible_actions(admissible_commands)
    if not history:
        if not task_desc and not memory_prompt:
            return ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=observation,
                admissible_actions=admissible_actions,
            )

        template = ALFWORLD_TEMPLATE_WITH_MEMORY if memory_prompt else ALFWORLD_TEMPLATE
        kwargs = {
            "task_description": task_desc,
            "step_count": 0,
            "history_length": 0,
            "action_history": "None yet.",
            "current_step": 1,
            "current_observation": observation,
            "admissible_actions": admissible_actions,
        }
        if memory_prompt:
            kwargs["retrieved_memories"] = memory_prompt
        return template.format(**kwargs)

    history_str, history_len = _format_action_history(history, max_recent=max_history)
    template = ALFWORLD_TEMPLATE_WITH_MEMORY if memory_prompt else ALFWORLD_TEMPLATE
    kwargs = {
        "task_description": task_desc,
        "step_count": len(history),
        "history_length": history_len,
        "action_history": history_str,
        "current_step": len(history) + 1,
        "current_observation": observation,
        "admissible_actions": admissible_actions,
    }
    if memory_prompt:
        kwargs["retrieved_memories"] = memory_prompt
    return template.format(**kwargs)

class ReActFewShotLibrary:
    """Loads optional few-shot exemplars in the ReAct ALFWorld JSON format."""

    def __init__(self, exemplars: dict[str, str]):
        self._exemplars = dict(exemplars)

    @classmethod
    def from_path(cls, path: Optional[str | Path]) -> "ReActFewShotLibrary | None":
        if not path:
            return None
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict JSON in {path}")
        return cls({str(k): str(v) for k, v in data.items()})

    def render(self, task_type: str, num_examples: int = 2) -> str:
        prefix = _TASK_PREFIXES.get(str(task_type or "").strip().lower())
        if not prefix:
            return ""

        examples: list[str] = []
        for idx in range(max(0, num_examples) - 1, -1, -1):
            key = f"react_{prefix}_{idx}"
            text = self._exemplars.get(key, "").strip()
            if text:
                examples.append(text)
        if not examples:
            return ""
        return "Interact with a household to solve a task. Here are some examples.\n" + "\n".join(examples)
