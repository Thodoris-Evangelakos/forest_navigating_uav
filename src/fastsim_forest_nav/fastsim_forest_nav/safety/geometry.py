from __future__ import annotations

from fastsim_forest_nav.envs.params import SimParams


def effective_drone_radius(params: SimParams) -> float:
    configured = float(getattr(params, "drone_radius", 0.0))
    if configured > 0.0:
        return configured
    return float(params.collision_threshold)


def soft_clearance_margin(params: SimParams) -> float:
    return float(max(0.0, float(params.r_safe) - effective_drone_radius(params)))


def protected_radius(params: SimParams) -> float:
    return effective_drone_radius(params) + soft_clearance_margin(params)
