"""Expose convenience wrappers for environment construction utilities."""

from __future__ import annotations

from typing import Any, Tuple

from fastsim_forest_nav.envs import GazeboParams, SimParams
from forest_nav_rl.env.factory import (
    build_env_ctor_and_kwargs as _build_env_ctor_and_kwargs,
    build_env_params as _build_env_params,
    get_env_backend as _get_env_backend,
)


def get_env_backend(env_cfg: dict[str, Any]) -> str:
    """Return the configured environment backend name."""
    return _get_env_backend(env_cfg)


def build_env_params(env_cfg: dict[str, Any]) -> SimParams | GazeboParams:
    """Build typed environment parameters from configuration mapping."""
    return _build_env_params(env_cfg)


def build_env_ctor_and_kwargs(env_cfg: dict[str, Any]) -> Tuple[type, dict[str, Any]]:
    """Return an environment constructor and keyword arguments tuple."""
    return _build_env_ctor_and_kwargs(env_cfg)
