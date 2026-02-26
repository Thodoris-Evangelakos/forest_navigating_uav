from __future__ import annotations

from typing import Any, Tuple

from fastsim_forest_nav.envs import GazeboParams, SimParams
from forest_nav_rl.env.factory import (
	build_env_ctor_and_kwargs as _build_env_ctor_and_kwargs,
	build_env_params as _build_env_params,
	get_env_backend as _get_env_backend,
)


def get_env_backend(env_cfg: dict[str, Any]) -> str:
	return _get_env_backend(env_cfg)


def build_env_params(env_cfg: dict[str, Any]) -> SimParams | GazeboParams:
	return _build_env_params(env_cfg)


def build_env_ctor_and_kwargs(env_cfg: dict[str, Any]) -> Tuple[type, dict[str, Any]]:
	return _build_env_ctor_and_kwargs(env_cfg)

