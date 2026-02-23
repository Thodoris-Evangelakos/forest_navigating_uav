from __future__ import annotations

from typing import Any, Tuple

from fastsim_forest_nav.envs.forest_nav_env import ForestNavEnv, SimParams
from fastsim_forest_nav.envs.gazebo_nav_env import GazeboForestNavEnv, GazeboParams


def get_env_backend(env_cfg: dict[str, Any]) -> str:
	backend = str(env_cfg.get("backend", "fastsim")).strip().lower()
	if backend not in {"fastsim", "gazebo"}:
		raise ValueError(f"Unsupported env backend: {backend}. Expected one of: fastsim, gazebo")
	return backend


def build_env_params(env_cfg: dict[str, Any]) -> SimParams | GazeboParams:
	env_kwargs = dict(env_cfg.get("env_kwargs", {}))
	raw_params = env_kwargs.get("params", env_cfg.get("sim_params", env_cfg.get("params", {})))

	backend = get_env_backend(env_cfg)
	if backend == "gazebo":
		if isinstance(raw_params, GazeboParams):
			return raw_params
		if isinstance(raw_params, dict):
			return GazeboParams(**raw_params)
		return GazeboParams()

	if isinstance(raw_params, SimParams):
		return raw_params
	if isinstance(raw_params, dict):
		return SimParams(**raw_params)
	return SimParams()


def build_env_ctor_and_kwargs(env_cfg: dict[str, Any]) -> Tuple[type, dict[str, Any]]:
	backend = get_env_backend(env_cfg)
	params = build_env_params(env_cfg)

	env_kwargs = dict(env_cfg.get("env_kwargs", {}))
	env_kwargs["params"] = params

	if backend == "gazebo":
		return GazeboForestNavEnv, env_kwargs
	return ForestNavEnv, env_kwargs

