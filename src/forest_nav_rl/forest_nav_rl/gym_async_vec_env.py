from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from gymnasium.vector import AsyncVectorEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv


def _index_value(value: Any, idx: int) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value[idx]

    if isinstance(value, (list, tuple)):
        return value[idx]

    return value


class GymAsyncVecEnv(VecEnv):
    def __init__(
        self,
        env_ctor: type,
        env_kwargs: dict[str, Any],
        n_envs: int,
        seed: int | None = None,
        monitor_dir: str | None = None,
        monitor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")

        self._base_seed = None if seed is None else int(seed)
        self._last_actions: np.ndarray | None = None

        monitor_kwargs = dict(monitor_kwargs or {})
        monitor_info_keys = tuple(monitor_kwargs.get("info_keywords", ()))

        env_fns: list[Callable[[], Any]] = []
        for rank in range(int(n_envs)):
            env_fns.append(
                self._build_env_fn(
                    env_ctor=env_ctor,
                    env_kwargs=env_kwargs,
                    rank=rank,
                    monitor_dir=monitor_dir,
                    monitor_info_keys=monitor_info_keys,
                )
            )

        self._gym_vec_env = AsyncVectorEnv(env_fns)
        super().__init__(
            num_envs=int(n_envs),
            observation_space=self._gym_vec_env.single_observation_space,
            action_space=self._gym_vec_env.single_action_space,
        )

    @staticmethod
    def _build_env_fn(
        env_ctor: type,
        env_kwargs: dict[str, Any],
        rank: int,
        monitor_dir: str | None,
        monitor_info_keys: Sequence[str],
    ) -> Callable[[], Any]:
        def _make() -> Any:
            env = env_ctor(**dict(env_kwargs))
            if monitor_dir is not None:
                monitor_path = Path(monitor_dir)
                monitor_path.mkdir(parents=True, exist_ok=True)
                env = Monitor(
                    env,
                    filename=str(monitor_path / f"{rank}"),
                    info_keywords=tuple(monitor_info_keys),
                )
            return env

        return _make

    def _vector_infos_to_list(self, infos: Any) -> list[dict[str, Any]]:
        if isinstance(infos, list):
            out: list[dict[str, Any]] = []
            for idx in range(self.num_envs):
                item = infos[idx] if idx < len(infos) else {}
                out.append(dict(item) if isinstance(item, dict) else {})
            return out

        if not isinstance(infos, dict):
            return [{} for _ in range(self.num_envs)]

        info_list = [{} for _ in range(self.num_envs)]
        for key, values in infos.items():
            if key.startswith("_"):
                continue
            mask_key = f"_{key}"
            mask = infos.get(mask_key)

            if mask is None:
                for idx in range(self.num_envs):
                    info_list[idx][key] = _index_value(values, idx)
                continue

            for idx in range(self.num_envs):
                if bool(_index_value(mask, idx)):
                    info_list[idx][key] = _index_value(values, idx)

        return info_list

    def reset(self) -> np.ndarray:
        seeds: list[int | None] | None = None
        if self._base_seed is not None:
            seeds = [self._base_seed + idx for idx in range(self.num_envs)]
            self._base_seed = None

        observations, infos = self._gym_vec_env.reset(seed=seeds)
        self.reset_infos = self._vector_infos_to_list(infos)
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        self._last_actions = np.array(actions, copy=False)

    def step_wait(self):
        if self._last_actions is None:
            raise RuntimeError("step_wait called before step_async")

        observations, rewards, terminated, truncated, infos = self._gym_vec_env.step(self._last_actions)
        info_list = self._vector_infos_to_list(infos)

        dones = np.logical_or(terminated, truncated)
        for idx in range(self.num_envs):
            info_list[idx]["TimeLimit.truncated"] = bool(truncated[idx] and not terminated[idx])
            if "final_observation" in info_list[idx]:
                info_list[idx]["terminal_observation"] = info_list[idx]["final_observation"]

        self._last_actions = None
        return observations, rewards, dones, info_list

    def close(self) -> None:
        self._gym_vec_env.close()

    def get_attr(self, attr_name: str, indices=None):
        return self._gym_vec_env.call(attr_name)

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        self._gym_vec_env.set_attr(attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return self._gym_vec_env.call(method_name, *method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.num_envs)]

    def get_images(self):
        return self._gym_vec_env.call("render")

    def seed(self, seed: int | None = None):
        self._base_seed = None if seed is None else int(seed)
        if self._base_seed is None:
            return [None for _ in range(self.num_envs)]
        return [self._base_seed + idx for idx in range(self.num_envs)]
