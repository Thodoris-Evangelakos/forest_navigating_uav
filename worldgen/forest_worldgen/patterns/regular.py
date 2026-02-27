"""
Model a regular inhibitory point process.

Generates spatially repulsive tree placements that approximate
plantation-style or competition-driven spacing.

Algorithm
---------
Bridson-style Poisson-disc with a background grid for O(1) neighbour
lookup, falling back to rejection sampling when the active list is
exhausted before ``count`` points are placed.

Validation targets (logged, not enforced):
    Clark-Evans R   > 1
    g(r) < 1        at small r
    L(r) - r < 0    at small r
"""

import math
import random


# ---------------------------------------------------------------------------
# Bridson Poisson-disc core
# ---------------------------------------------------------------------------


def _bridson_poisson_disc(
    count, x_min, x_max, y_min, y_max, d_min, existing_positions, k_candidates=30
):
    """
    Run Bridson's algorithm for Poisson-disc sampling.

    Apply it in a rectangular
    domain [x_min, x_max] x [y_min, y_max].

    Parameters
    ----------
    count : int
        Desired number of points.
    x_min, x_max, y_min, y_max : float
        Domain bounds.
    d_min : float
        Hard minimum separation distance.
    existing_positions : list[(float, float)]
        Points already placed (respected for distance checks).
    k_candidates : int
        Candidates tested around each active point before rejection.

    Returns
    -------
    list[(float, float)]
        Generated positions (up to *count*).
    """
    width = x_max - x_min
    height = y_max - y_min

    if d_min <= 0:
        # degenerate: just uniform
        pts = []
        for _ in range(count):
            pts.append((random.uniform(x_min, x_max), random.uniform(y_min, y_max)))
        return pts

    cell = d_min / math.sqrt(2)
    cols = max(1, int(math.ceil(width / cell)))
    rows = max(1, int(math.ceil(height / cell)))

    # spatial grid  (-1 = empty, otherwise index into *all_points*)
    grid = [[-1] * cols for _ in range(rows)]

    all_points = list(existing_positions)  # indices 0 .. len(existing)-1

    def _grid_col(x):
        return max(0, min(cols - 1, int((x - x_min) / cell)))

    def _grid_row(y):
        return max(0, min(rows - 1, int((y - y_min) / cell)))

    # seed existing points into the grid
    for idx, (px, py) in enumerate(existing_positions):
        if x_min <= px <= x_max and y_min <= py <= y_max:
            grid[_grid_row(py)][_grid_col(px)] = idx

    def _neighbours_ok(x, y):
        """Check d_min against grid neighbourhood (5x5 cells)."""
        gc = _grid_col(x)
        gr = _grid_row(y)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr = gr + dr
                cc = gc + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    pi = grid[rr][cc]
                    if pi != -1:
                        px, py2 = all_points[pi]
                        if math.hypot(x - px, y - py2) < d_min:
                            return False
        return True

    # --- phase 1: Bridson active-list ---
    new_points = []
    active = []

    def _insert(x, y):
        idx = len(all_points)
        all_points.append((x, y))
        new_points.append((x, y))
        grid[_grid_row(y)][_grid_col(x)] = idx
        active.append(idx)

    # first seed
    for _attempt in range(200):
        sx = random.uniform(x_min, x_max)
        sy = random.uniform(y_min, y_max)
        if _neighbours_ok(sx, sy):
            _insert(sx, sy)
            break
    else:
        # couldn't even place the seed; just uniform-fill
        sx = random.uniform(x_min, x_max)
        sy = random.uniform(y_min, y_max)
        _insert(sx, sy)

    while active and len(new_points) < count:
        # pick random active point
        ai = random.randrange(len(active))
        pi = active[ai]
        px, py = all_points[pi]
        found = False
        for _ in range(k_candidates):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(d_min, 2 * d_min)
            cx = px + r * math.cos(angle)
            cy = py + r * math.sin(angle)
            if cx < x_min or cx > x_max or cy < y_min or cy > y_max:
                continue
            if _neighbours_ok(cx, cy):
                _insert(cx, cy)
                found = True
                if len(new_points) >= count:
                    break
        if not found:
            active.pop(ai)

    # --- phase 2: rejection-sampling fallback ---
    max_fallback = count * 50
    attempts = 0
    while len(new_points) < count and attempts < max_fallback:
        cx = random.uniform(x_min, x_max)
        cy = random.uniform(y_min, y_max)
        attempts += 1
        if _neighbours_ok(cx, cy):
            _insert(cx, cy)

    # --- phase 3: if still short, relax and warn ---
    if len(new_points) < count:
        deficit = count - len(new_points)
        print(
            f"Warning [regular]: could only place {len(new_points)}/{count} "
            f"with d_min={d_min:.2f}; relaxing for {deficit} remaining points"
        )
        for _ in range(deficit):
            cx = random.uniform(x_min, x_max)
            cy = random.uniform(y_min, y_max)
            new_points.append((cx, cy))

    return new_points


# ---------------------------------------------------------------------------
# Public API  (matches the sampler signature used by the rest of the system)
# ---------------------------------------------------------------------------


def sample_regular(count, region, K, min_distance, existing_positions, params):
    """
    Poisson-disc / inhibitory point process placement.

    Parameters
    ----------
    count : int
        Number of points to generate.
    region : dict or None
        Rectangular sub-region ``{x_min, x_max, y_min, y_max}`` or None
        for the full K*K world.
    K : float
        World side length (domain is [-K/2, K/2]^2).
    min_distance : float
        Global minimum distance from world config.
    existing_positions : list[(float, float)]
        Already-placed points.
    params : dict
        Distribution parameters.  Recognised keys:
            ``min_distance``  - override d_min (default: world min_distance)
            ``k_candidates``  - Bridson candidates per active point (default 30)

    Returns
    -------
    list[(float, float)]
    """
    d_min = params.get("min_distance", min_distance)
    k_cand = int(params.get("k_candidates", 30))

    if region is not None:
        x_min = region["x_min"]
        x_max = region["x_max"]
        y_min = region["y_min"]
        y_max = region["y_max"]
    else:
        x_min = -K / 2
        x_max = K / 2
        y_min = -K / 2
        y_max = K / 2

    return _bridson_poisson_disc(
        count,
        x_min,
        x_max,
        y_min,
        y_max,
        d_min,
        existing_positions,
        k_candidates=k_cand,
    )
