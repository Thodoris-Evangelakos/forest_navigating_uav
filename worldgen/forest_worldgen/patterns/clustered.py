"""
Neyman–Scott cluster process (Thomas / Matérn variant).

Trees occur in clumps around latent cluster centres.  The algorithm:

  1) Sample C parent (cluster centre) positions uniformly.
  2) For each parent, draw k_i children:
       k_i ~ Poisson(mean_per_cluster)          (if variable_counts)
       k_i  = total_clustered // C              (otherwise, even split)
     Each child is displaced from its parent by:
       Gaussian scatter   r ~ N(0, σ=r_c/2)     (scatter='gaussian')
       Uniform-disk       r ~ U(0, r_c)         (scatter='disk')
  3) Optionally add background CSR noise points.

Key parameters (all in ``params``):
    cluster_count          C    – number of cluster centres
    cluster_radius         r_c  – characteristic cluster size
    mean_per_cluster            – mean children per cluster (Poisson draw)
    background_fraction         – fraction of total count placed as CSR
    scatter                     – 'gaussian' (default) | 'disk'
    variable_counts             – True → Poisson k_i; False → even split
    allow_overlap               – True → parents may be close together
    parent_separation_factor    – fraction of r_c used as min parent distance
                                  (default 0.5, ignored when allow_overlap)

Validation targets (logged, not enforced):
    Clark–Evans  R   < 1
    g(r)             > 1   at small r
    L(r) − r         > 0   at small / mid r
"""

import random
import math
from .csr import sample_csr, _point_in_rect, _point_in_area, _check_min_distance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _poisson(lam):
    """Sample from Poisson(lam) using Knuth's algorithm (fine for lam < ~50)."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p < L:
            return k - 1


def _scatter_gaussian(cx, cy, radius):
    """Return (x, y) displaced from (cx, cy) by isotropic Gaussian with σ = radius/2."""
    angle = random.uniform(0, 2 * math.pi)
    r = random.gauss(0, radius / 2)
    return cx + r * math.cos(angle), cy + r * math.sin(angle)


def _scatter_disk(cx, cy, radius):
    """Return (x, y) displaced uniformly inside a disk of given radius."""
    angle = random.uniform(0, 2 * math.pi)
    r = radius * math.sqrt(random.random())      # sqrt for uniform area
    return cx + r * math.cos(angle), cy + r * math.sin(angle)


_SCATTER = {
    'gaussian': _scatter_gaussian,
    'disk':     _scatter_disk,
}


def _clamp_to_bounds(x, y, region, K):
    """Clamp (x, y) inside the rectangular region or K×K world."""
    if region is not None:
        x = max(region['x_min'], min(region['x_max'], x))
        y = max(region['y_min'], min(region['y_max'], y))
    else:
        x = max(-K / 2, min(K / 2, x))
        y = max(-K / 2, min(K / 2, y))
    return x, y


# ---------------------------------------------------------------------------
# Parent placement
# ---------------------------------------------------------------------------

def _place_parents(cluster_count, region, K, cluster_radius,
                   allow_overlap, sep_factor, max_attempts=200):
    """Place *cluster_count* parent centres, optionally enforcing separation."""
    parents = []
    min_parent_dist = 0.0 if allow_overlap else cluster_radius * sep_factor

    for _ in range(cluster_count):
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
            # fallback: place anyway
            if region is not None:
                cx, cy = _point_in_rect(region)
            else:
                cx, cy = _point_in_area(K)
            parents.append((cx, cy))

    return parents


# ---------------------------------------------------------------------------
# Per-cluster child counts
# ---------------------------------------------------------------------------

def _child_counts(n_clustered, cluster_count, mean_per_cluster, variable):
    """
    Return a list of *cluster_count* integers summing to *n_clustered*.

    If *variable* is True each cluster draws from Poisson(mean_per_cluster),
    then counts are rescaled to sum to n_clustered.  Otherwise an even split
    is used (matching old behaviour).
    """
    if not variable or cluster_count <= 0:
        per = max(1, n_clustered // cluster_count) if cluster_count > 0 else 0
        remainder = n_clustered - per * cluster_count
        return [per + (1 if i < remainder else 0) for i in range(cluster_count)]

    raw = [max(0, _poisson(mean_per_cluster)) for _ in range(cluster_count)]
    total_raw = sum(raw)
    if total_raw == 0:
        # degenerate draw – fall back to even split
        per = max(1, n_clustered // cluster_count)
        remainder = n_clustered - per * cluster_count
        return [per + (1 if i < remainder else 0) for i in range(cluster_count)]

    # rescale so the total is exactly n_clustered
    scaled = [max(0, int(round(r / total_raw * n_clustered))) for r in raw]
    diff = n_clustered - sum(scaled)
    # distribute rounding residual
    for i in range(abs(diff)):
        idx = i % cluster_count
        scaled[idx] += 1 if diff > 0 else -1
    # ensure no negatives
    scaled = [max(0, s) for s in scaled]
    return scaled


# ---------------------------------------------------------------------------
# Public sampler
# ---------------------------------------------------------------------------

def sample_clustered(count, region, K, min_distance, existing_positions, params):
    """
    Neyman–Scott cluster process with configurable scatter and optional
    background noise.

    Parameters
    ----------
    count : int
        Total number of points to generate (clustered + background).
    region : dict | None
        Rectangular sub-region or None for the full K×K world.
    K : float
        World side length.
    min_distance : float
        Hard minimum inter-tree distance (world-level).
    existing_positions : list[tuple[float, float]]
        Already-placed points.
    params : dict
        See module docstring for recognised keys.

    Returns
    -------
    list[tuple[float, float]]
    """
    params = params or {}

    cluster_count    = int(params.get('cluster_count', 5))
    cluster_radius   = float(params.get('cluster_radius', 3.0))
    bg_fraction      = float(params.get('background_fraction', 0.15))
    mean_per_cluster = float(params.get('mean_per_cluster',
                              params.get('points_per_cluster_mean', 0)))
    scatter_name     = str(params.get('scatter', 'gaussian'))
    variable_counts  = bool(params.get('variable_counts', False))
    allow_overlap    = bool(params.get('allow_overlap', False))
    sep_factor       = float(params.get('parent_separation_factor', 0.5))
    max_attempts     = int(params.get('max_attempts', 200))

    scatter_fn = _SCATTER.get(scatter_name, _scatter_gaussian)

    # --- partition total count into clustered + background ---
    n_background = max(1, int(count * bg_fraction))
    n_clustered  = count - n_background

    # if mean_per_cluster was given but no explicit cluster_count, derive C
    if mean_per_cluster > 0 and cluster_count <= 0:
        cluster_count = max(1, int(round(n_clustered / mean_per_cluster)))
    # if mean_per_cluster wasn't given, derive it from the even split
    if mean_per_cluster <= 0 and cluster_count > 0:
        mean_per_cluster = n_clustered / cluster_count

    # --- 1) place parent centres ---
    parents = _place_parents(
        cluster_count, region, K, cluster_radius,
        allow_overlap, sep_factor, max_attempts,
    )

    # --- 2) generate per-cluster child counts ---
    counts = _child_counts(n_clustered, len(parents), mean_per_cluster, variable_counts)

    # --- 3) scatter children ---
    positions = []
    relaxed = 0

    for (cx, cy), n_children in zip(parents, counts):
        for _ in range(n_children):
            placed = False
            for _ in range(max_attempts):
                x, y = scatter_fn(cx, cy, cluster_radius)
                x, y = _clamp_to_bounds(x, y, region, K)
                if _check_min_distance(x, y, existing_positions + positions, min_distance):
                    positions.append((x, y))
                    placed = True
                    break
            if not placed:
                x, y = scatter_fn(cx, cy, cluster_radius)
                x, y = _clamp_to_bounds(x, y, region, K)
                positions.append((x, y))
                relaxed += 1

    if relaxed:
        print(f"Warning [clustered]: relaxed min_distance for {relaxed}/{n_clustered} "
              f"clustered points")

    # --- 4) background CSR fill ---
    bg = sample_csr(n_background, region, K, min_distance,
                    existing_positions + positions,
                    {'use_world_min_distance': True})
    positions.extend(bg)

    return positions
