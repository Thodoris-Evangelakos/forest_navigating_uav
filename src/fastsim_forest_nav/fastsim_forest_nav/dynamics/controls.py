"""Provide low-level control helpers for acceleration-limited commands."""

from __future__ import annotations

import numpy as np


def accel_limit_velocity(
    desired: float,
    current: float,
    accel_max: float,
    dt: float,
    decel_max: float | None = None,
) -> tuple[float, bool]:
    """Apply acceleration and deceleration limits to a target velocity."""
    if accel_max <= 0.0:
        return float(desired), False

    effective_decel = (
        float(decel_max) if decel_max is not None and float(decel_max) > 0.0 else float(accel_max)
    )
    accel_step = float(accel_max) * float(dt)
    decel_step = float(effective_decel) * float(dt)

    delta = float(desired) - float(current)
    if delta >= 0.0:
        clipped = delta > accel_step
        applied = float(current) + float(np.clip(delta, 0.0, accel_step))
    else:
        clipped = abs(delta) > decel_step
        applied = float(current) + float(np.clip(delta, -decel_step, 0.0))

    return applied, clipped


def map_normalized_accel(action_component: float, accel_max: float, decel_max: float) -> float:
    """Map a normalized action component in [-1, 1] to signed acceleration."""
    u = float(np.clip(action_component, -1.0, 1.0))
    pos_limit = float(max(accel_max, 0.0))
    neg_limit = float(decel_max) if float(decel_max) > 0.0 else pos_limit
    if u >= 0.0:
        return u * pos_limit
    return u * neg_limit


def approach_speed_cap(gap: float, decel_max: float, dt: float) -> float:
    """Compute the maximum speed that can still stop within the given gap."""
    g = float(max(gap, 0.0))
    a = float(max(decel_max, 1e-6))
    d = float(max(dt, 1e-6))
    return float(max(0.0, -a * d + np.sqrt((a * d) ** 2 + 2.0 * a * g)))
