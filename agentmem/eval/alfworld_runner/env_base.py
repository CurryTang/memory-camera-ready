"""Base environment interface for agentic task runners.

Provides the small interface used by the ALFWorld evaluation runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class EnvConfig:
    """Base configuration shared across all environments."""

    max_steps: int = 50
    max_retries: int = 3
    timeout_seconds: float = 120.0

@dataclass
class EnvStep:
    """A single environment step result."""

    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    success: bool = False

class AgentEnv(ABC):
    """Abstract environment for agentic evaluation.

    All environments follow the same lifecycle:
        env = MyEnv(config)
        env.reset(task_data)
        while not done:
            step = env.step(agent_action)
            done = step.done
        trajectory = env.get_trajectory()

    The format helpers convert observations and actions to chat messages.
    """

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()
        self._trajectory: list[dict[str, Any]] = []
        self._step_count: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0

    @abstractmethod
    def reset(self, task_data: dict[str, Any]) -> str:
        """Reset environment with task data. Returns initial observation."""
        ...

    @abstractmethod
    def step(self, action: str) -> EnvStep:
        """Execute an action. Returns observation, reward, done, info."""
        ...

    @property
    def done(self) -> bool:
        return self._done

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def step_count(self) -> int:
        return self._step_count

    def format_observation(self, observation: str) -> dict[str, str]:
        """Convert observation to a chat message dict for SLIME rollout."""
        return {"role": "user", "content": observation}

    def format_action(self, action: str) -> dict[str, str]:
        """Convert action to a chat message dict."""
        return {"role": "assistant", "content": action}

    def get_trajectory(self) -> list[dict[str, Any]]:
        """Return the full trajectory of (observation, action, reward) tuples."""
        return list(self._trajectory)

    def get_task_type(self) -> str:
        """Return the task type/category for skill organization."""
        return "generic"

    def get_success(self) -> bool:
        """Whether the task was completed successfully."""
        return False

    def close(self) -> None:
        """Clean up resources."""
        pass
