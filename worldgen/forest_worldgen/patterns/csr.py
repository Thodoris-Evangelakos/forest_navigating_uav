"""
Complete Spatial Randomness (CSR) — homogeneous Poisson point process.

Each point is sampled i.i.d. uniformly over the domain.  There is
**no interaction** between points by default; an optional small
``min_distance`` can be applied purely to avoid exact overlaps
(e.g. two trunks at the same pixel), but the world-level
``min_distance`` is intentionally *not* used here so that the
resulting pattern remains a valid CSR null model.

Validation targets (logged, not enforced):
    Clark–Evans  R  ≈ 1
    g(r)             ≈ 1  across tested r
    L(r) − r         ≈ 0
"""

import random
import math


# ---------------------------------------------------------------------------
# Geometry helpers  (re-used by other pattern modules)
# ---------------------------------------------------------------------------

def _point_in_rect(region):
    """Sample a uniform random point inside a rectangular region."""
    x = random.uniform(region['x_min'], region['x_max'])
    y = random.uniform(region['y_min'], region['y_max'])
    return x, y


def _point_in_area(K):
    """Sample a uniform random point inside a K×K area centred at origin."""
    return random.uniform(-K / 2, K / 2), random.uniform(-K / 2, K / 2)


def _check_min_distance(x, y, positions, min_distance):
    """Return True if (x, y) is at least *min_distance* from every position."""
    for px, py in positions:
        if math.hypot(x - px, y - py) < min_distance:
            return False
    return True


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def sample_csr(count, region, K, min_distance, existing_positions, params):
    """
    Homogeneous Poisson point process (CSR) with an optional overlap guard.

    Parameters
    ----------
    count : int
        Number of points to generate.
    region : dict | None
        Rectangular sub-region ``{x_min, x_max, y_min, y_max}`` or *None*
        for the full K×K world.
    K : float
        World side length (domain is ``[-K/2, K/2]²``).
    min_distance : float
        World-level minimum distance — **ignored by default** for CSR so
        the pattern stays interaction-free.  Use ``params['overlap_guard']``
        to set a small anti-overlap distance instead.
    existing_positions : list[tuple[float, float]]
        Already-placed points (respected for the overlap guard only).
    params : dict
        Recognised keys:

        ``overlap_guard`` – small distance to prevent exact overlaps
            (default **0.0**, i.e. pure CSR).

        ``use_world_min_distance`` – if *True*, fall back to the
            world-level *min_distance* instead of the overlap guard
            (default *False*).  Useful when CSR is used as a sub-sampler
            inside another pattern (e.g. background fill in clustered).

    Returns
    -------
    list[tuple[float, float]]
    """
    params = params or {}

    # --- resolve effective guard distance ---
    if params.get('use_world_min_distance', False):
        guard = min_distance
    else:
        guard = float(params.get('overlap_guard', 0.0))

    max_attempts = int(params.get('max_attempts', 200))

    positions = []
    all_existing = list(existing_positions)   # mutable working copy
    relaxed_count = 0

    for _ in range(count):
        placed = False
        for _ in range(max_attempts):
            if region is not None:
                x, y = _point_in_rect(region)
            else:
                x, y = _point_in_area(K)

            if guard <= 0 or _check_min_distance(x, y, all_existing + positions, guard):
                positions.append((x, y))
                placed = True
                break

        if not placed:
            # place anyway to honour the requested count
            if region is not None:
                x, y = _point_in_rect(region)
            else:
                x, y = _point_in_area(K)
            positions.append((x, y))
            relaxed_count += 1

    if relaxed_count:
        print(f"Warning [csr]: relaxed overlap guard for {relaxed_count}/{count} points")

    return positions
