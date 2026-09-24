"""ALFWorld environment wrapper for evaluation.

ALFWorld is a text-based household task environment with 6 task types:
Pick, Look, Clean, Heat, Cool, Pick2.

This wrapper supports both the alfworld gym interface and a simplified offline
mode using pre-collected trajectories.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agentmem.eval.alfworld_runner.env_base import AgentEnv, EnvConfig, EnvStep

_TASK_TYPE_PATTERNS = {
    "pick_and_place": "Pick",
    "look_at_obj": "Look",
    "pick_clean_then_place": "Clean",
    "pick_heat_then_place": "Heat",
    "pick_cool_then_place": "Cool",
    "pick_two_obj": "Pick2",
}

def _resolve_alfred_tw_env_class() -> Any:
    """Resolve AlfredTWEnv across alfworld package layouts."""
    import alfworld.agents.environment as env_module

    env_cls = getattr(env_module, "AlfredTWEnv", None)
    if env_cls is not None:
        return env_cls

    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    return AlfredTWEnv

def _load_alfworld_config(config_or_path: Any) -> Any:
    """Load an alfworld config dict from a path or pass dicts through."""
    if isinstance(config_or_path, dict):
        return config_or_path
    if not config_or_path:
        bundled = (
            Path(__file__).resolve().parents[3]
            / "resource"
            / "AgentGym_RL"
            / "AgentGym"
            / "agentenv-alfworld"
            / "configs"
            / "base_config.yaml"
        )
        if bundled.exists():
            config_or_path = bundled
        else:
            data_root = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
            return _build_default_alfworld_config(data_root)

    import yaml

    with open(str(config_or_path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def _build_default_alfworld_config(data_root: str) -> dict[str, Any]:
    json_root = os.path.join(str(data_root), "json_2.1.1")
    train_path = os.path.join(json_root, "train")
    seen_path = os.path.join(json_root, "valid_seen")
    unseen_path = os.path.join(json_root, "valid_unseen")
    return {
        "env": {
            "goal_desc_human_anns_prob": 0.0,
            "task_types": [1, 2, 3, 4, 5, 6],
            "domain_randomization": False,
            "expert_type": "handcoded",

            "alfred_tw": {
                "data_path": train_path,
                "valid_seen_data_path": seen_path,
                "valid_unseen_data_path": unseen_path,
            },
        },
        "dataset": {
            "data_path": train_path,
            "eval_id_data_path": seen_path,
            "eval_ood_data_path": unseen_path,
            "num_train_games": 0,
            "num_eval_games": 0,
        },
        "general": {
            "training_method": "dqn",
        },
        "rl": {
            "training": {
                "max_nb_steps_per_episode": 50,
            },
        },
    }

def _infer_alfworld_train_eval(game_file: str) -> str:
    normalized = str(game_file or "").replace("\\", "/")
    split_aliases = {
        "eval_in_distribution": "eval_in_distribution",
        "eval_out_of_distribution": "eval_out_of_distribution",
        "valid_seen": "eval_in_distribution",
        "valid_unseen": "eval_out_of_distribution",
    }
    if normalized in split_aliases:
        return split_aliases[normalized]
    if "valid_unseen/" in normalized or normalized.startswith("valid_unseen"):
        return "eval_out_of_distribution"
    if "valid_seen/" in normalized or normalized.startswith("valid_seen"):
        return "eval_in_distribution"
    return "train"

def _resolve_alfworld_game_file(game_file: str, data_root: str) -> str:
    game_file = str(game_file or "")
    if not game_file:
        return game_file
    if os.path.isabs(game_file):
        return game_file
    return os.path.join(str(data_root), "json_2.1.1", game_file)

def _unwrap_batch_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0]
    return value

@dataclass
class ALFWorldConfig(EnvConfig):
    """ALFWorld-specific configuration."""

    max_steps: int = 50
    use_gym: bool = False
    alfworld_data_path: str = ""
    alfworld_config_path: str = ""
    reward_success: float = 1.0
    reward_failure: float = 0.0
    reward_step_penalty: float = 0.0

class ALFWorldEnv(AgentEnv):
    """ALFWorld environment wrapper.

    Supports two modes:
    1. **Gym mode** (use_gym=True): Connects to alfworld.agents.environment
    2. **Offline mode** (default): Works with pre-extracted task descriptions
       for training without alfworld dependency.
    """

    def __init__(self, config: Optional[ALFWorldConfig] = None) -> None:
        super().__init__(config or ALFWorldConfig())
        self._config: ALFWorldConfig = self.config                            
        self._gym_env: Any = None
        self._task_type: str = "generic"
        self._task_desc: str = ""
        self._admissible_commands: list[str] = []
        self._success: bool = False
        self._game_file: str = ""
        self._loaded_game_file: str = ""
        self._loaded_train_eval: str = ""
        self._loaded_game_pool_key: str = ""

    def reset(self, task_data: dict[str, Any]) -> str:
        """Reset with task data.

        task_data keys:
            - game_file (str): Path to the game file
            - task_desc (str): Task description text
            - task_type (str, optional): Task category
            - admissible_commands (list[str], optional): Valid actions
            - use_gym (bool, optional): Override config
        """
        self._trajectory = []
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._success = False

        self._game_file = str(task_data.get("game_file", ""))
        self._task_desc = str(task_data.get("task_desc", ""))
        self._task_type = str(task_data.get("task_type", ""))
        self._admissible_commands = list(task_data.get("admissible_commands", []))

        if not self._task_type and self._game_file:
            for pattern, task_type in _TASK_TYPE_PATTERNS.items():
                if pattern in self._game_file:
                    self._task_type = task_type
                    break

        use_gym = task_data.get("use_gym", self._config.use_gym)
        if use_gym:
            return self._reset_gym(task_data)

        initial_obs = self._task_desc or f"Task: {self._game_file}"
        self._trajectory.append({
            "step": 0,
            "observation": initial_obs,
            "action": None,
            "reward": 0.0,
            "done": False,
        })
        return initial_obs

    def step(self, action: str) -> EnvStep:
        """Execute an action in ALFWorld."""
        if self._done:
            return EnvStep(observation="Episode is done.", reward=0.0, done=True, success=self._success)

        self._step_count += 1
        action = action.strip()

        if self._gym_env is not None:
            return self._step_gym(action)

        return self._step_offline(action)

    def _step_offline(self, action: str) -> EnvStep:
        """Offline step using heuristic simulation for training data generation."""
        reward = self._config.reward_step_penalty
        done = False
        success = False

        if self._step_count >= self._config.max_steps:
            done = True
            observation = "Maximum steps reached."
        elif action.lower().startswith("think"):
            observation = "OK."
        else:

            observation = f"You {action}."

        if done:
            self._done = True
            self._success = success
            reward = self._config.reward_success if success else self._config.reward_failure

        self._total_reward += reward
        self._trajectory.append({
            "step": self._step_count,
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
        })
        return EnvStep(observation=observation, reward=reward, done=done, success=success)

    def _reset_gym(self, task_data: dict[str, Any]) -> str:
        """Reset using alfworld gym environment.

        If task_data contains ``game_file``, that specific game is loaded.
        Otherwise the gym samples a random game from its configured split.
        """
        try:
            config_path = task_data.get("config_path", self._config.alfworld_config_path)
            config = _load_alfworld_config(config_path)
            game_file = str(task_data.get("game_file", ""))
            data_root = os.environ.get("ALFWORLD_DATA", self._config.alfworld_data_path or "")
            resolved_game_file = _resolve_alfworld_game_file(game_file, data_root)
            game_files = task_data.get("game_files") or []
            resolved_game_files = [
                _resolve_alfworld_game_file(str(path), data_root)
                for path in game_files
                if str(path)
            ]
            game_pool_key = "\n".join(resolved_game_files)
            raw_train_eval = str(task_data.get("train_eval", "") or "")
            train_eval = _infer_alfworld_train_eval(raw_train_eval or game_file)

            should_rebuild = self._gym_env is None
            if not should_rebuild and resolved_game_files:
                should_rebuild = game_pool_key != self._loaded_game_pool_key
            elif not should_rebuild and resolved_game_file:
                should_rebuild = resolved_game_file != self._loaded_game_file
            if not should_rebuild:
                should_rebuild = train_eval != self._loaded_train_eval

            if should_rebuild:
                if self._gym_env is not None:
                    try:
                        self._gym_env.close()
                    except Exception:
                        pass
                env_builder = _resolve_alfred_tw_env_class()(config, train_eval=train_eval)
                if resolved_game_files:
                    env_builder.game_files = resolved_game_files
                    env_builder.num_games = len(resolved_game_files)
                elif resolved_game_file:
                    env_builder.game_files = [resolved_game_file]
                    env_builder.num_games = 1
                self._gym_env = env_builder.init_env(batch_size=1)
                self._loaded_game_file = "" if resolved_game_files else resolved_game_file
                self._loaded_game_pool_key = game_pool_key
                self._loaded_train_eval = train_eval

            obs, info = self._gym_env.reset()

            observation = obs[0] if isinstance(obs, list) else str(obs)
            self._admissible_commands = list(info.get("admissible_commands", [[]])[0])
            self._trajectory.append({
                "step": 0,
                "observation": observation,
                "action": None,
                "reward": 0.0,
                "done": False,
            })
            return observation
        except ImportError:
            raise ImportError(
                "alfworld is required for gym mode. Install with: pip install alfworld"
            )

    def _step_gym(self, action: str) -> EnvStep:
        """Step using alfworld gym environment."""
        action = self._normalize_action(action)
        obs, rewards, dones, infos = self._gym_env.step([action])
        observation = str(_unwrap_batch_value(obs))
        reward = float(_unwrap_batch_value(rewards))
        done = bool(_unwrap_batch_value(dones))
        success = reward > 0
        self._admissible_commands = list(_unwrap_batch_value(infos.get("admissible_commands", [[]])))

        if done or self._step_count >= self._config.max_steps:
            self._done = True
            self._success = success

        self._total_reward += reward
        self._trajectory.append({
            "step": self._step_count,
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
        })
        return EnvStep(
            observation=observation,
            reward=reward,
            done=done or self._step_count >= self._config.max_steps,
            success=success,
            info={"admissible_commands": self._admissible_commands},
        )

    def _normalize_action(self, action: str) -> str:
        """Canonicalize a free-form action to the current admissible command set."""
        action = action.strip()
        if not action or not self._admissible_commands:
            return action

        normalized = self._normalize_action_text(action)
        by_normalized = {
            self._normalize_action_text(candidate): candidate
            for candidate in self._admissible_commands
        }

        exact = by_normalized.get(normalized)
        if exact is not None:
            return exact

        if normalized.startswith("put ") and (" on " in normalized or " in " in normalized):
            alt_normalized = normalized.replace(" on ", " in ") if " on " in normalized else normalized.replace(" in ", " on ")
            alt = by_normalized.get(alt_normalized)
            if alt is not None:
                return alt

        return action

    @staticmethod
    def _normalize_action_text(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"[.!?]+$", "", text)
        return re.sub(r"\s+", " ", text)

    def get_task_type(self) -> str:
        return self._task_type or "generic"

    def get_success(self) -> bool:
        return self._success

    def get_admissible_commands(self) -> list[str]:
        return list(self._admissible_commands)

    def close(self) -> None:
        if self._gym_env is not None:
            try:
                self._gym_env.close()
            except Exception:
                pass
            self._gym_env = None
