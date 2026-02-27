"""Record per-episode trajectories and reset metadata for analysis."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class TrajectoryRecorder(gym.Wrapper):
    """Records the agent's position trajectory during episodes."""

    def __init__(self, env: gym.Env):
        """Wrap an environment and initialize trajectory buffers."""
        super().__init__(env)
        self.trajectory: list[np.ndarray] = []
        self.episode_start_pos: np.ndarray | None = None
        self.episode_goal_pos: np.ndarray | None = None
        self.episode_trees: np.ndarray | None = None
        self.episode_reset_info: dict[str, Any] = {}

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and start a fresh trajectory log."""
        obs, info = self.env.reset(**kwargs)
        self.trajectory = []
        self.episode_reset_info = dict(info) if isinstance(info, dict) else {}

        # Capture episode metadata
        if hasattr(self.env.unwrapped, "pos"):
            self.episode_start_pos = np.array(self.env.unwrapped.pos, dtype=np.float64)
            self.trajectory.append(self.episode_start_pos.copy())

        if hasattr(self.env.unwrapped, "goal"):
            self.episode_goal_pos = np.array(self.env.unwrapped.goal, dtype=np.float64)

        if hasattr(self.env.unwrapped, "trees"):
            trees = self.env.unwrapped.trees
            if trees is not None:
                self.episode_trees = np.array(trees, dtype=np.float64)

        return obs, info

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the wrapped environment and append the current position."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        if hasattr(self.env.unwrapped, "pos"):
            self.trajectory.append(np.array(self.env.unwrapped.pos, dtype=np.float64))

        return obs, float(reward), terminated, truncated, info

    def get_trajectory(self) -> np.ndarray:
        """Return the trajectory as an ``(N, 3)`` array of XYZ positions."""
        if not self.trajectory:
            return np.empty((0, 3), dtype=np.float64)
        return np.array(self.trajectory, dtype=np.float64)
