"""
Neyman-Scott cluster process.

Trees occur in clumps around latent parent (cluster) centres.

Algorithm
---------
1.  Sample *C* parent centres uniformly in the domain, optionally
    enforcing a minimum separation ``min_parent_distance`` so that
    clusters are spread out (set to 0 to allow full overlap).
2.  For each parent, draw a child count from Poisson(mean_per_cluster)
    (clamped so the total budget *count* is respected).
3.  Scatter children around each parent using either:
        - ``gaussian``      - isotropic Gaussian (σ = cluster_radius / 2)
        - ``uniform_disk``  - uniform within a disk of radius cluster_radius
4.  Optionally fill a ``background_fraction`` of the budget with pure
    CSR points (avoids empty gaps between clusters).

Key parameters (all in ``params`` dict)
---------------------------------------
    cluster_count           C  - number of parent centres (default 5)
    cluster_radius          r_c - scatter radius (default 3.0)
    scatter_shape           'gaussian' | 'uniform_disk' (default 'gaussian')
    mean_per_cluster        expected children per parent (default: count/C)
    min_parent_distance     hard spacing between parents
                            (default: cluster_radius * 0.5)
    background_fraction     fraction of count placed as CSR (default 0.15)
    allow_cluster_overlap   if True, min_parent_distance is 0 (default False)

Validation targets (logged, not enforced):
    Clark-Evans R   < 1
    g(r) > 1        at small r
    L(r) − r > 0    at small / mid r
"""

import random
import math
from .csr import sample_csr, _point_in_rect, _point_in_area, _check_min_distance


# ---------------------------------------------------------------------------
# Child-scatter helpers
# ---------------------------------------------------------------------------

def _scatter_gaussian(cx, cy, radius):
    """Isotropic Gaussian scatter (σ = radius / 2) around (cx, cy)."""
    angle = random.uniform(0, 2 * math.pi)
    r = random.gauss(0, radius / 2)
    return cx + r * math.cos(angle), cy + r * math.sin(angle)


def _scatter_uniform_disk(cx, cy, radius):
    """Uniform scatter inside a disk of given radius around (cx, cy)."""
    angle = random.uniform(0, 2 * math.pi)
    r = radius * math.sqrt(random.random())      # sqrt for uniform area
    return cx + r * math.cos(angle), cy + r * math.sin(angle)


_SCATTER_FNS = {
    'gaussian': _scatter_gaussian,
    'uniform_disk': _scatter_uniform_disk,
}


# ---------------------------------------------------------------------------
# Parent placement
# ---------------------------------------------------------------------------

def _place_parents(count, region, K, min_parent_dist, max_attempts=200):
    """Place *count* parent centres with optional minimum separation."""
    parents = []
    for _ in range(count):
        placed = False
        for _ in range(max_attempts):
            if region is not None:
                cx, cy = _point_in_rect(region)
            else:
                cx, cy = _point_in_area(K)
            if min_parent_dist <= 0 or _check_min_distance(cx, cy, parents, min_parent_dist):
                parents.append((cx, cy))
                placed = True
                break
        if not placed:
            # fall back - place anyway so we don't lose a cluster
            if region is not None:
                cx, cy = _point_in_rect(region)
            else:
                cx, cy = _point_in_area(K)
            parents.append((cx, cy))
    return parents


# ---------------------------------------------------------------------------
# Per-cluster child counts  (Poisson, clamped to budget)
# ---------------------------------------------------------------------------

def _poisson_child_counts(n_clusters, total_children, mean_per_cluster):
    """
    Draw Poisson-distributed counts for each cluster, then rescale so
    they sum to *total_children*.
    """
    if n_clusters <= 0:
        return []

    raw = [max(0, int(random.gauss(mean_per_cluster, math.sqrt(mean_per_cluster))))
           for _ in range(n_clusters)]

    raw_sum = sum(raw)
    if raw_sum == 0:
        # degenerate - fall back to even split
        base = total_children // n_clusters
        counts = [base] * n_clusters
        remainder = total_children - base * n_clusters
        indices = list(range(n_clusters))
        random.shuffle(indices)
        for i in indices[:remainder]:
            counts[i] += 1
        return counts

    # rescale to hit total_children exactly
    counts = [max(0, round(c / raw_sum * total_children)) for c in raw]
    diff = total_children - sum(counts)
    # distribute the rounding residual randomly
    indices = list(range(n_clusters))
    random.shuffle(indices)
    for i in indices:
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        counts[i] = max(0, counts[i] + step)
        diff -= step

    return counts


# ---------------------------------------------------------------------------
# Clamp helpers
# ---------------------------------------------------------------------------

def _clamp_to_bounds(x, y, region, K):
    """Clamp (x, y) inside the placement bounds."""
    if region is not None:
        x = max(region['x_min'], min(region['x_max'], x))
        y = max(region['y_min'], min(region['y_max'], y))
    else:
        x = max(-K / 2, min(K / 2, x))
        y = max(-K / 2, min(K / 2, y))
    return x, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_clustered(count, region, K, min_distance, existing_positions, params):
    """
    Neyman-Scott cluster process.

    Parameters
    ----------
    count : int
        Total number of child points to generate (including background).
    region : dict | None
        Rectangular sub-region or *None* for the full K×K world.
    K : float
        World side length.
    min_distance : float
        World-level hard minimum distance between any two trees.
    existing_positions : list[tuple[float, float]]
        Already-placed points.
    params : dict
        See module docstring for recognised keys.

    Returns
    -------
    list[tuple[float, float]]
    """
    params = params or {}

    cluster_count = max(1, int(params.get('cluster_count', 5)))
    cluster_radius = float(params.get('cluster_radius', 3.0))
    scatter_shape = params.get('scatter_shape', 'gaussian')
    bg_fraction = float(params.get('background_fraction', 0.15))
    allow_overlap = bool(params.get('allow_cluster_overlap', False))

    # parent separation
    if allow_overlap:
        min_parent_dist = 0.0
    else:
        min_parent_dist = float(params.get('min_parent_distance',
                                            cluster_radius * 0.5))

    # child budget
    n_background = max(1, int(count * bg_fraction))
    n_clustered = count - n_background

    mean_per_cluster = float(params.get('mean_per_cluster',
                                         n_clustered / max(cluster_count, 1)))

    # resolve scatter function
    scatter_fn = _SCATTER_FNS.get(scatter_shape)
    if scatter_fn is None:
        raise ValueError(
            f"Unknown scatter_shape '{scatter_shape}'; "
            f"choose from {list(_SCATTER_FNS.keys())}"
        )

    # --- 1. place parent centres ---
    parents = _place_parents(cluster_count, region, K, min_parent_dist)

    # --- 2. draw per-cluster child counts ---
    child_counts = _poisson_child_counts(cluster_count, n_clustered,
                                          mean_per_cluster)

    # --- 3. scatter children ---
    positions = []
    relaxed = 0
    max_attempts = 200

    for (cx, cy), n_kids in zip(parents, child_counts):
        for _ in range(n_kids):
            placed = False
            for _ in range(max_attempts):
                x, y = scatter_fn(cx, cy, cluster_radius)
                x, y = _clamp_to_bounds(x, y, region, K)
                if _check_min_distance(x, y,
                                       existing_positions + positions,
                                       min_distance):
                    positions.append((x, y))
                    placed = True
                    break
            if not placed:
                x, y = scatter_fn(cx, cy, cluster_radius)
                x, y = _clamp_to_bounds(x, y, region, K)
                positions.append((x, y))
                relaxed += 1

    if relaxed:
        print(f"Warning [clustered]: relaxed min_distance for "
              f"{relaxed}/{n_clustered} clustered points")

    # --- 4. background fill ---
    bg = sample_csr(n_background, region, K, min_distance,
                    existing_positions + positions,
                    {'use_world_min_distance': True})
    positions.extend(bg)

    return positions
