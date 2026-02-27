"""
Complete Spatial Randomness (CSR) — homogeneous Poisson point process.

Each point is sampled i.i.d. uniformly over the domain.  There is
**no interaction** between points by default; an optional small
``min_distance`` can be applied purely to avoid exact overlaps
(e.g. two trunks at the same pixel), but the world-level
``min_distance`` is intentionally *not* used here so that the
resulting pattern remains a valid CSR null model.

Validation targets (logged, not enforced):
    Clark-Evans  R  ≈ 1
    g(r)             ≈ 1  across tested r
    L(r) - r         ≈ 0
"""

import random
import math


# ---------------------------------------------------------------------------
# Geometry helpers  (re-used by other pattern modules)
# ---------------------------------------------------------------------------


def _point_in_rect(region):
    """Sample a uniform random point inside a rectangular region."""
    x = random.uniform(region["x_min"], region["x_max"])
    y = random.uniform(region["y_min"], region["y_max"])
    return x, y


def _point_in_area(K):
    """Sample a uniform random point inside a K*K area centred at origin."""
    return random.uniform(-K / 2, K / 2), random.uniform(-K / 2, K / 2)


def _check_min_distance(x, y, positions, min_distance):
    """Return True if (x, y) is at least *min_distance* from every position.

    Brute-force O(n) fallback — prefer using ProximityGrid for hot loops.
    """
    for px, py in positions:
        if math.hypot(x - px, y - py) < min_distance:
            return False
    return True


# ---------------------------------------------------------------------------
# Grid-based spatial index for O(1) amortised proximity checks
# ---------------------------------------------------------------------------


class ProximityGrid:
    """Uniform grid that accelerates minimum-distance queries.

    Cell size equals *min_distance* so only a 5×5 neighbourhood
    needs checking — constant-time per query regardless of total
    point count.
    """

    __slots__ = ("cell", "grid", "points")

    def __init__(self, min_distance, initial_points=None):
        """Initialize the grid with a minimum distance and optional seed points."""
        self.cell = max(min_distance, 1e-9)
        self.grid = {}  # (col, row) -> list of (x, y)
        self.points = []
        if initial_points:
            for pt in initial_points:
                self.insert(*pt)

    def _key(self, x, y):
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def insert(self, x, y):
        """Add a point to the grid."""
        k = self._key(x, y)
        bucket = self.grid.get(k)
        if bucket is None:
            self.grid[k] = [(x, y)]
        else:
            bucket.append((x, y))
        self.points.append((x, y))

    def check(self, x, y, min_distance):
        """Return True if (x, y) is >= *min_distance* from every stored point."""
        min_d_sq = min_distance * min_distance
        gc, gr = self._key(x, y)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                bucket = self.grid.get((gc + dc, gr + dr))
                if bucket is not None:
                    for px, py in bucket:
                        dx = x - px
                        dy = y - py
                        if dx * dx + dy * dy < min_d_sq:
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
        for the full K*K world.
    K : float
        World side length (domain is ``[-K/2, K/2]**2``).
    min_distance : float
        World-level minimum distance — **ignored by default** for CSR so
        the pattern stays interaction-free.  Use ``params['overlap_guard']``
        to set a small anti-overlap distance instead.
    existing_positions : list[tuple[float, float]]
        Already-placed points (respected for the overlap guard only).
    params : dict
        Recognised keys:

        ``overlap_guard`` - small distance to prevent exact overlaps
            (default **0.0**, i.e. pure CSR).

        ``use_world_min_distance`` - if *True*, fall back to the
            world-level *min_distance* instead of the overlap guard
            (default *False*).  Useful when CSR is used as a sub-sampler
            inside another pattern (e.g. background fill in clustered).

    Returns
    -------
    list[tuple[float, float]]
    """
    params = params or {}

    # --- resolve effective guard distance ---
    if params.get("use_world_min_distance", False):
        guard = min_distance
    else:
        guard = float(params.get("overlap_guard", 0.0))

    max_attempts = int(params.get("max_attempts", 200))

    # Use grid-accelerated proximity checks when guard > 0
    use_grid = guard > 0
    if use_grid:
        grid = ProximityGrid(guard, existing_positions)

    positions = []
    relaxed_count = 0

    for _ in range(count):
        placed = False
        for _ in range(max_attempts):
            if region is not None:
                x, y = _point_in_rect(region)
            else:
                x, y = _point_in_area(K)

            if not use_grid:
                positions.append((x, y))
                placed = True
                break

            if grid.check(x, y, guard):
                positions.append((x, y))
                grid.insert(x, y)
                placed = True
                break

        if not placed:
            # place anyway to honour the requested count
            if region is not None:
                x, y = _point_in_rect(region)
            else:
                x, y = _point_in_area(K)
            positions.append((x, y))
            if use_grid:
                grid.insert(x, y)
            relaxed_count += 1

    if relaxed_count:
        print(f"Warning [csr]: relaxed overlap guard for {relaxed_count}/{count} points")

    return positions
