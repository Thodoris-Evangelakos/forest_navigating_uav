"""Provide geometry helpers for safety radii and collision clearance."""

from __future__ import annotations

from fastsim_forest_nav.envs.params import SimParams


def effective_drone_radius(params: SimParams) -> float:
    """Return the effective drone radius used for collision checks."""
    configured = float(getattr(params, "drone_radius", 0.0))
    if configured > 0.0:
        return configured
    return float(params.collision_threshold)


def soft_clearance_margin(params: SimParams) -> float:
    """Return the soft safety margin outside the physical drone radius."""
    return float(max(0.0, float(params.r_safe) - effective_drone_radius(params)))


def protected_radius(params: SimParams) -> float:
    """Return the full protected radius combining body and soft margin."""
    return effective_drone_radius(params) + soft_clearance_margin(params)
