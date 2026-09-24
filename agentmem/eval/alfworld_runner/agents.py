"""Prompt-based ALFWorld agent implementations."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agentmem.eval.alfworld_runner.journal import (
    Journal,
    parse_journal_search,
    render_journal_results,
)
from agentmem.eval.alfworld_runner.memory import BaseAlfWorldMemoryBackend
from agentmem.eval.alfworld_runner.prompts import (
    ReActFewShotLibrary,
    build_reflexion_prompt,
    build_react_prompt,
    build_vanilla_prompt,
)
from agentmem.providers.base import Message
from agentmem.providers.openai_compat import OpenAICompatibleProvider
from examples.agent_rl.trainable_agents import parse_action

@dataclass
class AgentStepRecord:
    prompt: str
    raw_response: str
    action: str
    observation: str

_CLASSIC_REACT_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

def _default_classic_react_prompt_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "vendor"
        / "reflexion"
        / "alfworld_runs"
        / "prompts"
        / "alfworld_3prompts.json"
    )

def _process_classic_observation(observation: str) -> str:
    text = str(observation or "")
    if text.startswith("You arrive at loc "):
        return text[text.find(". ") + 2 :]
    return text

class ClassicEnvironmentHistory:
    """Minimal history formatter matching the public Reflexion/ReAct ALFWorld loop."""

    def __init__(self, base_query: str, start_info: str, memory: list[str]) -> None:
        query = base_query
        if memory:
            query += "\n\nYour memory for the task below:"
            for idx, item in enumerate(memory):
                query += f"\nTrial {idx}:\n{item.strip()}"
        query += f"\nHere is the task:\n{start_info}"
        self._base_query = query
        self._history: list[dict[str, str]] = []
        self._last_action = ""
        self._is_exhausted = False

    def add(self, label: str, value: str) -> None:
        self._history.append({"label": label, "value": value})
        if label == "action":
            if value == self._last_action:
                self._is_exhausted = True
            else:
                self._last_action = value

    def check_is_exhausted(self) -> bool:
        return self._is_exhausted

    def __str__(self) -> str:
        text = self._base_query + "\n"
        for idx, item in enumerate(self._history):
            if item["label"] == "action":
                text += f'> {item["value"]}'
            else:
                text += item["value"]
            if idx != len(self._history) - 1:
                text += "\n"
        return text

class ClassicReActFewShotLibrary:
    """Loads the original ALFWorld ReAct exemplars from Reflexion/ReAct."""

    def __init__(self, exemplars: dict[str, str]) -> None:
        self._exemplars = dict(exemplars)

    @classmethod
    def from_path(cls, path: Optional[str | Path]) -> "ClassicReActFewShotLibrary":
        prompt_path = Path(path) if path else _default_classic_react_prompt_path()
        data = json.loads(prompt_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict JSON in {prompt_path}")
        return cls({str(k): str(v) for k, v in data.items()})

    def render(self, task_type: str) -> str:
        prefix = _CLASSIC_REACT_PREFIXES.get(str(task_type or "").strip().lower())
        if not prefix:
            return "Interact with a household to solve a task."
        example_1 = self._exemplars.get(f"react_{prefix}_1", "")
        example_0 = self._exemplars.get(f"react_{prefix}_0", "")
        return "Interact with a household to solve a task. Here are two examples.\n" + example_1 + example_0

class BaseAlfWorldAgent:
    """Shared LLM-driven ALFWorld agent."""

    mode = "base"

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_history: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 256,
        memory_backend: Optional[BaseAlfWorldMemoryBackend] = None,
        react_few_shot_path: Optional[str] = None,
        react_num_examples: int = 2,
        max_chat_history_turns: int = 15,
        max_trials: int = 1,
        reflection_max_tokens: int = 256,
    ) -> None:
        self._provider = OpenAICompatibleProvider(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
            model=model,
            base_url=base_url,
            default_max_tokens=max_tokens,
            disable_thinking=True,
        )
        self.max_history = max(1, int(max_history))
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.memory_backend = memory_backend
        self.react_num_examples = max(0, int(react_num_examples))
        self.max_chat_history_turns = max(1, int(max_chat_history_turns))
        self._max_trials = max(1, int(max_trials))
        self.reflection_max_tokens = max(32, int(reflection_max_tokens))
        self.few_shot_library = ReActFewShotLibrary.from_path(react_few_shot_path)

        self._task_desc = ""
        self._task_type = ""
        self._memory_prompt = ""
        self._history: list[tuple[str, str]] = []
        self._records: list[AgentStepRecord] = []
        self._messages: list[Message] = []
        self._trial_idx = 0

    @property
    def max_trials(self) -> int:
        return self._max_trials

    def begin_task(self, task_info: dict[str, Any]) -> None:
        del task_info
        self._trial_idx = 0

    def reset(self, initial_observation: str, task_info: dict[str, Any]) -> None:
        del initial_observation
        self._task_desc = str(task_info.get("task_desc") or "").strip()
        self._task_type = str(task_info.get("task_type") or "").strip()
        self._trial_idx = int(task_info.get("trial_idx", self._trial_idx))
        self._history = []
        self._records = []
        self._messages = []
        self._memory_prompt = ""
        if self.memory_backend is not None and self._task_desc:
            self._memory_prompt = self.memory_backend.get_memory_prompt(
                task_desc=self._task_desc,
                task_type=self._task_type,
            )

    def act(self, observation: str, admissible_commands: list[str]) -> str:
        prompt = self._build_prompt(observation, admissible_commands)
        if not self._messages:
            self._messages = [Message(role="user", content=prompt)]
        else:
            self._messages.append(Message(role="user", content=prompt))
            self._truncate_messages()
        response = self._provider.chat(
            self._messages,
            temperature=self.temperature,
        )
        raw_response = str(response.content or "").strip()
        self._messages.append(Message(role="assistant", content=raw_response))
        action = self._parse_action(raw_response, admissible_commands)
        self._records.append(
            AgentStepRecord(
                prompt=prompt,
                raw_response=raw_response,
                action=action,
                observation=observation,
            )
        )
        return action

    def observe(self, action: str, observation: str) -> None:
        self._history.append((observation, action))

    def finish_episode(
        self,
        *,
        episode_idx: int,
        task_info: dict[str, Any],
        trajectory: list[dict[str, Any]],
        success: bool,
    ) -> None:
        if self.memory_backend is None:
            return
        self.memory_backend.update_from_episode(
            episode_idx=episode_idx,
            task_desc=str(task_info.get("task_desc") or self._task_desc),
            task_type=str(task_info.get("task_type") or self._task_type),
            success=success,
            trajectory=trajectory,
        )

    def get_debug_trace(self) -> list[dict[str, Any]]:
        return [
            {
                "prompt": record.prompt,
                "raw_response": record.raw_response,
                "action": record.action,
                "observation": record.observation,
            }
            for record in self._records
        ]

    def should_stop_episode(self) -> bool:
        return False

    def finish_trial(
        self,
        *,
        task_info: dict[str, Any],
        trajectory: list[dict[str, Any]],
        success: bool,
        trial_idx: int,
        error: Optional[str] = None,
    ) -> None:
        del task_info, trajectory, success, trial_idx, error

    def _build_prompt(self, observation: str, admissible_commands: list[str]) -> str:
        raise NotImplementedError

    def _few_shot_prompt(self) -> str:
        if self.few_shot_library is None:
            return ""
        return self.few_shot_library.render(
            self._task_type,
            num_examples=self.react_num_examples,
        )

    def _parse_action(self, raw_response: str, admissible_commands: list[str]) -> str:
        action = parse_action(raw_response).strip()
        normalized = self._normalize_action(action, admissible_commands)
        return normalized or "look"

    def _normalize_action(self, action: str, admissible_commands: list[str]) -> str:
        candidate = str(action or "").strip()
        if not candidate:
            return ""

        commands = [str(cmd).strip() for cmd in admissible_commands if str(cmd).strip()]
        if not commands:
            return candidate

        candidate = candidate.strip("`").strip().rstrip(".")
        candidate = re.sub(r"^\[action\]\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*\[/action\]\s*$", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"^\s*action\s*[:\-]\s*", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip().strip("'\"").rstrip(".")

        for cmd in commands:
            if candidate == cmd:
                return cmd
        lowered = candidate.lower()
        for cmd in commands:
            if lowered == cmd.lower():
                return cmd

        substring_matches = [cmd for cmd in commands if cmd.lower() in lowered]
        if substring_matches:
            return max(substring_matches, key=len)

        return candidate

    _MAX_INPUT_CHARS = 80_000

    def _truncate_messages(self) -> None:
        limit = 1 + self.max_chat_history_turns * 2
        if len(self._messages) > limit:
            self._messages = self._messages[:1] + self._messages[-(limit - 1):]

        total_chars = sum(len(str(m.content or "")) for m in self._messages)
        while total_chars > self._MAX_INPUT_CHARS and len(self._messages) > 3:

            removed = self._messages.pop(1)
            total_chars -= len(str(removed.content or ""))
            if len(self._messages) > 1:
                removed2 = self._messages.pop(1)
                total_chars -= len(str(removed2.content or ""))

class VanillaAlfWorldAgent(BaseAlfWorldAgent):
    mode = "vanilla"

    def _build_prompt(self, observation: str, admissible_commands: list[str]) -> str:
        return build_vanilla_prompt(
            observation=observation,
            task_desc=self._task_desc,
            admissible_commands=admissible_commands,
            history=self._history,
            memory_prompt=self._memory_prompt,
            few_shot_prompt="",
            max_history=self.max_history,
        )

class ReActAlfWorldAgent(BaseAlfWorldAgent):
    mode = "react"

    def _build_prompt(self, observation: str, admissible_commands: list[str]) -> str:
        return build_react_prompt(
            observation=observation,
            task_desc=self._task_desc,
            admissible_commands=admissible_commands,
            history=self._history,
            memory_prompt=self._memory_prompt,
            few_shot_prompt=self._few_shot_prompt(),
            max_history=self.max_history,
        )

class MemoryAugmentedReActAlfWorldAgent(ReActAlfWorldAgent):
    mode = "react_memory"

_GOLDEN_RULES_PATH = Path(__file__).resolve().parent / "golden_rules.txt"

class GoldenRulesReActAlfWorldAgent(ReActAlfWorldAgent):
    """ReAct agent that injects a static rule sheet through the memory prompt."""

    mode = "golden_rules"

    def __init__(
        self,
        *,
        golden_rules_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("memory_backend", None)
        super().__init__(memory_backend=None, **kwargs)
        path = Path(golden_rules_path) if golden_rules_path else _GOLDEN_RULES_PATH
        self._golden_rules_text = path.read_text(encoding="utf-8").strip()

    def reset(self, initial_observation: str, task_info: dict[str, Any]) -> None:
        super().reset(initial_observation, task_info)

        self._memory_prompt = self._golden_rules_text

class ReflexionAlfWorldAgent(ReActAlfWorldAgent):
    """Multi-trial ReAct agent with verbal self-reflection between retries."""

    mode = "reflexion"

    def __init__(self, *, max_reflections: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_reflections = max(1, int(max_reflections))
        self._reflections: list[str] = []

    def begin_task(self, task_info: dict[str, Any]) -> None:
        super().begin_task(task_info)
        self._reflections = []

    def _build_prompt(self, observation: str, admissible_commands: list[str]) -> str:
        return build_reflexion_prompt(
            observation=observation,
            task_desc=self._task_desc,
            admissible_commands=admissible_commands,
            history=self._history,
            reflections=self._reflections,
            memory_prompt=self._memory_prompt,
            few_shot_prompt=self._few_shot_prompt(),
            max_history=self.max_history,
        )

    def finish_trial(
        self,
        *,
        task_info: dict[str, Any],
        trajectory: list[dict[str, Any]],
        success: bool,
        trial_idx: int,
        error: Optional[str] = None,
    ) -> None:
        if success:
            return
        reflection = self._summarize_reflection(
            task_desc=str(task_info.get("task_desc") or self._task_desc),
            task_type=str(task_info.get("task_type") or self._task_type),
            trajectory=trajectory,
            trial_idx=trial_idx,
            error=error,
        )
        if not reflection:
            return
        self._reflections.append(reflection)
        self._reflections = self._reflections[-self.max_reflections :]

    def _summarize_reflection(
        self,
        *,
        task_desc: str,
        task_type: str,
        trajectory: list[dict[str, Any]],
        trial_idx: int,
        error: Optional[str],
    ) -> str:
        trace_lines: list[str] = []
        for idx, step in enumerate(trajectory[-10:]):
            action = str(step.get("action") or "").strip()
            observation = str(step.get("observation") or "").strip()
            if action:
                trace_lines.append(f"Step {idx}: action={action}")
            if observation:
                trace_lines.append(f"Step {idx}: observation={observation}")
        if not trace_lines:
            trace_lines.append("No useful trajectory was recorded.")

        prompt = "\n\n".join(
            [
                "Write a concise reflection for a future retry of the same ALFWorld task.",
                f"Trial: {trial_idx + 1}",
                f"Task type: {task_type}",
                f"Task: {task_desc}",
                f"Failure: {error or 'The task was not completed within the step budget.'}",
                "Recent trajectory:\n" + "\n".join(trace_lines),
                "Output 2-4 short sentences with concrete mistakes to avoid and the next strategy to try.",
            ]
        )
        response = self._provider.chat(
            [Message(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=self.reflection_max_tokens,
        )
        return str(response.content or "").strip()

class MemoryAugmentedReflexionAlfWorldAgent(ReflexionAlfWorldAgent):
    mode = "reflexion_memory"

class LongContextJournalAgent(ReActAlfWorldAgent):
    """Long-context baseline: ReAct + a flat journal of past trajectories with grep tool.

    The agent receives a description of `journal_search: <query>` at the start of every
    episode. During each environment step, the LLM may issue up to
    `max_journal_calls_per_step` journal queries before it must emit an environment
    action. Each journal call is one extra LLM round-trip; the result is appended to
    the chat history as a synthetic user message and the loop re-prompts.

    The journal itself is the deliberately-vanilla long-context proxy: append-only,
    no embedding, no graph, no encoding. Queries are case-insensitive substring grep
    over `task_desc + trajectory_text`. See `agentmem/eval/alfworld_runner/journal.py`.
    """

    mode = "longcontext_journal"

    SEARCH_TOOL_NAME = "journal_search"

    def __init__(
        self,
        *,
        journal: Journal,
        max_journal_calls_per_step: int = 8,
        journal_top_k: int = 3,
        journal_max_chars_per_entry: int = 2000,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("memory_backend", None)
        super().__init__(memory_backend=None, **kwargs)
        self.journal = journal
        self.max_journal_calls_per_step = max(0, int(max_journal_calls_per_step))
        self.journal_top_k = max(1, int(journal_top_k))
        self.journal_max_chars_per_entry = max(200, int(journal_max_chars_per_entry))
        self._journal_calls_this_step = 0

    def reset(self, initial_observation: str, task_info: dict[str, Any]) -> None:
        super().reset(initial_observation, task_info)
        self._journal_calls_this_step = 0

    def _journal_tool_block(self) -> str:
        tool = self.SEARCH_TOOL_NAME
        return (
            "You also have access to a journal of past ALFWorld trajectories you have "
            "lived through. To consult it, emit on its own line:\n"
            f"    {tool}: <plain-text query>\n"
            "The system will return up to "
            f"{self.journal_top_k} matching past trajectory snippets, most-recent first. "
            f"You may issue at most {self.max_journal_calls_per_step} journal queries before each "
            "environment action. Use the journal to recall what worked or failed in similar "
            "rooms, then output a single concrete environment action grounded in the current "
            "observation."
        )

    def _build_prompt(self, observation: str, admissible_commands: list[str]) -> str:
        base = super()._build_prompt(observation, admissible_commands)

        if not self._messages:
            return self._journal_tool_block() + "\n\n" + base
        return base

    def act(self, observation: str, admissible_commands: list[str]) -> str:
        prompt = self._build_prompt(observation, admissible_commands)
        if not self._messages:
            self._messages = [Message(role="user", content=prompt)]
        else:
            self._messages.append(Message(role="user", content=prompt))
            self._truncate_messages()

        self._journal_calls_this_step = 0
        last_raw_response = ""

        for _ in range(self.max_journal_calls_per_step + 1):
            response = self._provider.chat(
                self._messages,
                temperature=self.temperature,
            )
            raw_response = str(response.content or "").strip()
            last_raw_response = raw_response
            self._messages.append(Message(role="assistant", content=raw_response))

            if self._journal_calls_this_step >= self.max_journal_calls_per_step:

                action = self._parse_action(raw_response, admissible_commands)
                self._records.append(
                    AgentStepRecord(
                        prompt=prompt,
                        raw_response=raw_response,
                        action=action,
                        observation=observation,
                    )
                )
                return action

            query = parse_journal_search(raw_response, tool_name=self.SEARCH_TOOL_NAME)
            if query is None:

                action = self._parse_action(raw_response, admissible_commands)
                self._records.append(
                    AgentStepRecord(
                        prompt=prompt,
                        raw_response=raw_response,
                        action=action,
                        observation=observation,
                    )
                )
                return action

            self._journal_calls_this_step += 1
            matches = self.journal.grep(
                query,
                k=self.journal_top_k,
                max_chars_per_entry=self.journal_max_chars_per_entry,
            )
            tool_response = render_journal_results(matches, tool_name=self.SEARCH_TOOL_NAME)
            if self._journal_calls_this_step >= self.max_journal_calls_per_step:
                tool_response += (
                    f"\n\n[{self.SEARCH_TOOL_NAME}] Budget exhausted "
                    f"({self._journal_calls_this_step}/{self.max_journal_calls_per_step}). "
                    "Output an environment action now."
                )
            self._messages.append(Message(role="user", content=tool_response))
            self._truncate_messages()

        action = self._parse_action(last_raw_response, admissible_commands)
        self._records.append(
            AgentStepRecord(
                prompt=prompt,
                raw_response=last_raw_response,
                action=action,
                observation=observation,
            )
        )
        return action

class DCIJournalAgent(LongContextJournalAgent):
    """DCI-Lite ALFWorld agent: raw episode corpus + bounded LLM-driven search.

    Same runtime as ``LongContextJournalAgent`` (append-only journal + grep tool),
    with two prompt-level changes:

    1. The tool is exposed as ``dci_search:`` (Direct-Corpus Interaction).
    2. The system block reframes the corpus as raw past trajectories that the
       agent must lexically search and adapt, not paraphrase or copy verbatim.

    The framing matches the DCI-Agent-Lite paper's principle: keep raw records,
    let the controller plan bounded searches over them, never abstract the
    trajectory at write time. The agent's task is to identify a successful
    receptacle search order or action template from past episodes and apply it
    to the current room (where object ids and counts will differ).
    """

    mode = "dci_lite"
    SEARCH_TOOL_NAME = "dci_search"

    def _journal_tool_block(self) -> str:
        tool = self.SEARCH_TOOL_NAME
        return (
            "DIRECT-CORPUS MEMORY (DCI). You have an append-only memory of past "
            "ALFWorld episodes. Each entry is the *raw* action/observation trace "
            "of one episode (no summary, no graph). Episodes from successful runs "
            "and from failures are both stored.\n\n"
            "Tool: to search the corpus, emit on its own line:\n"
            f"    {tool}: <plain-text query>\n"
            f"You may issue at most {self.max_journal_calls_per_step} searches before "
            "each environment action. Each search returns up to "
            f"{self.journal_top_k} matching past trajectories, most-recent first.\n\n"
            "How to use this corpus well:\n"
            "  (a) Plan a search whose query identifies the closest analogous task "
            "(target object + receptacle, e.g. \"clean tomato microwave\", "
            "\"cool egg fridge\", \"two pencil drawer\").\n"
            "  (b) From the returned traces, extract: the *order* in which "
            "receptacles were checked, the *exact action template* that succeeded "
            "(e.g. take X from Y / clean X with Z / heat X with M / put X in Y), "
            "and any failure→recovery patterns.\n"
            "  (c) Adapt the strategy to your *current* room: the object ids and "
            "receptacle counts will differ. Do NOT copy past entity ids verbatim. "
            "Always verify against the current observation and admissible actions.\n"
            "  (d) If a search returns no useful match, refine the query (try a "
            "broader term, or just the action verb plus the object class) before "
            "giving up.\n\n"
            "Then output ONE environment action grounded in the current observation."
        )

class DCIJournalSummarizeAgent(DCIJournalAgent):
    """DCI-Lite+Sum ALFWorld agent: same as DCIJournalAgent but the prompt asks
    the model to abstract a one-line *strategy* from each retrieved trace before
    acting, biasing it toward transferable patterns rather than verbatim replay.

    The corpus itself is still raw (matching the AMABench DCI-Lite+Sum design,
    which adds a context-level=4 summarization step on retrieved evidence
    rather than at write time).
    """

    mode = "dci_lite_sum"
    SEARCH_TOOL_NAME = "dci_search"

    def _journal_tool_block(self) -> str:
        base = super()._journal_tool_block()
        return base + (
            "\n\nBefore acting, internally distill each retrieved trace into a "
            "one-line strategy of the form: \"target X in receptacle-class Y → "
            "search order [Y1, Y2, ...] → action template [...]\". Use that "
            "abstracted strategy to choose the next environment action; do not "
            "copy the trace's exact tokens unless the receptacle id is the same."
        )

class ClassicReActAlfWorldAgent(BaseAlfWorldAgent):
    """Original ALFWorld ReAct prompting loop adapted from Reflexion/ReAct."""

    mode = "classic_react"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._classic_library = ClassicReActFewShotLibrary.from_path(kwargs.get("react_few_shot_path"))
        self._classic_history: Optional[ClassicEnvironmentHistory] = None
        self._classic_exhausted = False
        self._classic_retry_count = 6
        self._classic_retry_temperature_step = 0.2
        self._classic_memory_limit = 3

    def reset(self, initial_observation: str, task_info: dict[str, Any]) -> None:
        self._task_desc = str(task_info.get("task_desc") or "").strip()
        self._task_type = str(task_info.get("task_type") or "").strip()
        self._history = []
        self._records = []
        self._messages = []
        self._memory_prompt = ""
        self._classic_exhausted = False

        memory_items: list[str] = []
        if self.memory_backend is not None and self._task_desc:
            self._memory_prompt = self.memory_backend.get_memory_prompt(
                task_desc=self._task_desc,
                task_type=self._task_type,
            )
            if self._memory_prompt.strip():
                memory_items = [self._memory_prompt.strip()]

        base_prompt = self._classic_library.render(self._task_type)
        self._classic_history = ClassicEnvironmentHistory(
            base_prompt=base_prompt,
            start_info=_process_classic_observation(initial_observation),
            memory=memory_items[-self._classic_memory_limit :],
        )

    def act(self, observation: str, admissible_commands: list[str]) -> str:
        del observation
        if self._classic_history is None:
            raise RuntimeError("ClassicReActAlfWorldAgent.reset() must be called before act().")

        prompt = f"{self._classic_history}>"
        raw_response = ""
        for attempt in range(self._classic_retry_count):
            temperature = self.temperature + attempt * self._classic_retry_temperature_step
            response = self._provider.chat(
                [Message(role="user", content=prompt)],
                temperature=temperature,
                max_tokens=self.max_tokens,
                stop=["\n"],
            )
            raw_response = str(response.content or "").strip()
            if len(raw_response.strip()) >= 5:
                break

        action = self._parse_classic_action(raw_response, admissible_commands)
        self._records.append(
            AgentStepRecord(
                prompt=prompt,
                raw_response=raw_response,
                action=action,
                observation="",
            )
        )
        return action

    def observe(self, action: str, observation: str) -> None:
        processed_observation = _process_classic_observation(observation)
        if action.lower().startswith("think:"):
            processed_observation = "OK."
        self._history.append((processed_observation, action))
        if self._classic_history is not None:
            self._classic_history.add("action", action)
            self._classic_history.add("observation", processed_observation)
            self._classic_exhausted = self._classic_history.check_is_exhausted()

    def should_stop_episode(self) -> bool:
        return self._classic_exhausted

    def _build_prompt(self, observation: str, admissible_commands: list[str]) -> str:
        del observation, admissible_commands
        if self._classic_history is None:
            return ""
        return f"{self._classic_history}>"

    def _parse_classic_action(self, raw_response: str, admissible_commands: list[str]) -> str:
        candidate = str(raw_response or "").strip()
        candidate = candidate.lstrip(">").strip().strip("`")
        if not candidate:
            return "look"
        if candidate.lower().startswith("think:"):
            return candidate
        normalized = self._normalize_action(candidate, admissible_commands)
        return normalized or candidate

def create_agent(
    agent_type: str,
    *,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_history: int = 5,
    memory_backend: Optional[BaseAlfWorldMemoryBackend] = None,
    react_few_shot_path: Optional[str] = None,
    react_num_examples: int = 2,
    max_chat_history_turns: int = 15,
    max_trials: int = 1,
    reflection_max_tokens: int = 256,
    max_reflections: int = 3,
    journal: Optional[Journal] = None,
    max_journal_calls_per_step: int = 8,
    journal_top_k: int = 3,
) -> BaseAlfWorldAgent:
    key = str(agent_type or "").strip().lower().replace("-", "_")
    kwargs = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_history": max_history,
        "memory_backend": memory_backend,
        "react_few_shot_path": react_few_shot_path,
        "react_num_examples": react_num_examples,
        "max_chat_history_turns": max_chat_history_turns,
        "max_trials": max_trials,
        "reflection_max_tokens": reflection_max_tokens,
    }
    if key == "vanilla":
        return VanillaAlfWorldAgent(**kwargs)
    if key == "react":
        return ReActAlfWorldAgent(**kwargs)
    if key in {"classic_react", "react_classic", "public_react"}:
        return ClassicReActAlfWorldAgent(**kwargs)
    if key in {"react_memory", "react_with_memory"}:
        if memory_backend is None:
            raise ValueError("react_memory requires a memory backend")
        return MemoryAugmentedReActAlfWorldAgent(**kwargs)
    if key in {"golden_rules", "rules", "react_rules"}:
        kwargs.pop("memory_backend", None)
        return GoldenRulesReActAlfWorldAgent(**kwargs)
    if key == "reflexion":
        return ReflexionAlfWorldAgent(max_reflections=max_reflections, **kwargs)
    if key in {"reflexion_memory", "reflexion_with_memory"}:
        if memory_backend is None:
            raise ValueError("reflexion_memory requires a memory backend")
        return MemoryAugmentedReflexionAlfWorldAgent(max_reflections=max_reflections, **kwargs)
    if key in {"longcontext_journal", "journal", "longcontext"}:
        if journal is None:
            raise ValueError("longcontext_journal requires a Journal instance")
        kwargs.pop("memory_backend", None)
        return LongContextJournalAgent(
            journal=journal,
            max_journal_calls_per_step=max_journal_calls_per_step,
            journal_top_k=journal_top_k,
            **kwargs,
        )
    if key in {"dci_journal", "dci_lite", "dci_lite_journal"}:
        if journal is None:
            raise ValueError("dci_journal requires a Journal instance")
        kwargs.pop("memory_backend", None)
        return DCIJournalAgent(
            journal=journal,
            max_journal_calls_per_step=max_journal_calls_per_step,
            journal_top_k=journal_top_k,
            **kwargs,
        )
    if key in {"dci_journal_sum", "dci_lite_sum", "dci_lite_sum_journal"}:
        if journal is None:
            raise ValueError("dci_journal_sum requires a Journal instance")
        kwargs.pop("memory_backend", None)
        return DCIJournalSummarizeAgent(
            journal=journal,
            max_journal_calls_per_step=max_journal_calls_per_step,
            journal_top_k=journal_top_k,
            **kwargs,
        )
    raise ValueError(f"Unknown ALFWorld agent type: {agent_type}")
