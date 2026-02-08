"""
Scale-dependent (multi-scale) point process.

Combines **clustering at small scale** with **inhibition at a larger
scale**, mimicking natural forests where saplings form clumps but
adult trees compete and thin each other out.

Algorithm
---------
1.  Generate an *over-sampled* clustered pattern (Neyman–Scott) using
    ``sample_clustered`` with a small hard ``min_distance``.
2.  Apply a **secondary thinning** pass at a larger inhibition scale
    ``d_mid``:
        * Deterministic mode (default) – greedily remove the later of
          any two points closer than ``d_mid``.
        * Probabilistic mode – for each pair closer than ``d_mid``,
          remove one with probability ``thin_probability``.
3.  If thinning removed too many points, top up with additional
    clustered children (using the same parents) to approach the
    requested ``count``.

Key parameters (all in ``params`` dict)
---------------------------------------
    d_mid               mid-range inhibition distance (required).
    thin_mode           'deterministic' | 'probabilistic'
                        (default 'deterministic')
    thin_probability    removal probability per violating pair when
                        thin_mode='probabilistic' (default 0.8)
    oversample_factor   how much to oversample before thinning
                        (default 1.5)
    topup_attempts      max rejection-sampling rounds for the top-up
                        phase (default 300)

    Plus all ``sample_clustered`` parameters (cluster_count,
    cluster_radius, scatter_shape, mean_per_cluster, etc.).

Validation targets (logged, not enforced):
    g(r) > 1     at small r   (clustering signature)
    g(r) < 1     at mid r     (inhibition signature)
    L(r) - r     crosses from positive to negative as r increases
"""

import math
import random

from .csr import _check_min_distance, _point_in_area, _point_in_rect
from .clustered import (
    sample_clustered,
    _place_parents,
    _SCATTER_FNS,
    _scatter_gaussian,
    _clamp_to_bounds,
)


# ---------------------------------------------------------------------------
# Thinning passes
# ---------------------------------------------------------------------------

def _thin_deterministic(positions, d_mid):
    """
    Greedy deterministic thinning.

    Walk through *positions* in order; for each point, check against all
    already-accepted points.  Reject if any accepted neighbour is within
    ``d_mid``.  The ordering is randomised first so that the retained set
    isn't biased toward early indices.
    """
    indices = list(range(len(positions)))
    random.shuffle(indices)

    accepted_idx = []
    accepted_pts = []
    for i in indices:
        x, y = positions[i]
        if _check_min_distance(x, y, accepted_pts, d_mid):
            accepted_idx.append(i)
            accepted_pts.append((x, y))

    return accepted_pts


def _thin_probabilistic(positions, d_mid, p_remove=0.8):
    """
    Probabilistic thinning.

    For every point, if it violates ``d_mid`` against any already-accepted
    point, remove it with probability ``p_remove`` (otherwise keep it).
    """
    indices = list(range(len(positions)))
    random.shuffle(indices)

    accepted_pts = []
    for i in indices:
        x, y = positions[i]
        if _check_min_distance(x, y, accepted_pts, d_mid):
            accepted_pts.append((x, y))
        elif random.random() > p_remove:
            # survived the thinning coin-flip
            accepted_pts.append((x, y))

    return accepted_pts


# ---------------------------------------------------------------------------
# Top-up: fill deficit by scattering more children around existing parents
# ---------------------------------------------------------------------------

def _topup(positions, deficit, parents, cluster_radius, scatter_fn,
           region, K, min_distance, d_mid, max_attempts=500):
    """
    Try to add *deficit* more points that respect both ``min_distance``
    (hard overlap guard) **and** ``d_mid`` (mid-scale inhibition) against
    all existing *positions*.

    Strategy: alternate between cluster-scatter candidates (to preserve
    the clustering signal) and uniform candidates (to fill gaps the
    clusters can't reach).
    """
    added = []
    all_pts = list(positions)
    attempts = 0
    total_budget = deficit * max_attempts

    while len(added) < deficit and attempts < total_budget:
        attempts += 1

        # 60 % cluster scatter, 40 % uniform fill
        if parents and random.random() < 0.6:
            cx, cy = random.choice(parents)
            x, y = scatter_fn(cx, cy, cluster_radius)
        else:
            if region is not None:
                x, y = _point_in_rect(region)
            else:
                x, y = _point_in_area(K)

        x, y = _clamp_to_bounds(x, y, region, K)

        if not _check_min_distance(x, y, all_pts, min_distance):
            continue
        if not _check_min_distance(x, y, all_pts, d_mid):
            continue

        added.append((x, y))
        all_pts.append((x, y))

    return added


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_scale_dependent(count, region, K, min_distance, existing_positions, params):
    """
    Scale-dependent multi-scale process.

    Parameters
    ----------
    count : int
        Desired number of output points.
    region : dict | None
        Rectangular sub-region or *None* for the full K×K world.
    K : float
        World side length.
    min_distance : float
        World-level hard minimum distance (small-scale overlap guard).
    existing_positions : list[tuple[float, float]]
        Already-placed points.
    params : dict
        See module docstring for recognised keys.

    Returns
    -------
    list[tuple[float, float]]
    """
    params = params or {}

    d_mid = float(params.get('d_mid', min_distance * 2.5))
    thin_mode = params.get('thin_mode', 'deterministic')
    thin_prob = float(params.get('thin_probability', 0.8))
    oversample = float(params.get('oversample_factor', 1.5))
    topup_attempts = int(params.get('topup_attempts', 300))

    # cluster params forwarded as-is (sample_clustered reads its own keys)
    cluster_radius = float(params.get('cluster_radius', 3.0))
    scatter_shape = params.get('scatter_shape', 'gaussian')
    scatter_fn = _SCATTER_FNS.get(scatter_shape, _scatter_gaussian)

    # --- 1. over-sample a clustered pattern ---
    n_oversample = max(count, int(math.ceil(count * oversample)))
    raw = sample_clustered(
        n_oversample, region, K, min_distance, existing_positions, params,
    )

    # --- 2. thin at mid-range scale ---
    if thin_mode == 'probabilistic':
        thinned = _thin_probabilistic(raw, d_mid, p_remove=thin_prob)
    else:
        thinned = _thin_deterministic(raw, d_mid)

    # also enforce d_mid against existing_positions
    if existing_positions:
        final = []
        for x, y in thinned:
            if _check_min_distance(x, y, existing_positions, d_mid):
                final.append((x, y))
        thinned = final

    # --- 3. trim or top-up to hit count ---
    if len(thinned) > count:
        random.shuffle(thinned)
        thinned = thinned[:count]

    elif len(thinned) < count:
        deficit = count - len(thinned)

        # recover parent centres for top-up scatter
        cluster_count = max(1, int(params.get('cluster_count', 5)))
        allow_overlap = bool(params.get('allow_cluster_overlap', False))
        if allow_overlap:
            min_parent_dist = 0.0
        else:
            min_parent_dist = float(params.get('min_parent_distance',
                                                cluster_radius * 0.5))
        parents = _place_parents(cluster_count, region, K, min_parent_dist)

        extra = _topup(
            existing_positions + thinned, deficit, parents,
            cluster_radius, scatter_fn, region, K,
            min_distance, d_mid, max_attempts=topup_attempts,
        )
        thinned.extend(extra)

        if len(thinned) < count:
            shortfall = count - len(thinned)
            print(f"Warning [scale_dependent]: could only place "
                  f"{len(thinned)}/{count} after thinning + top-up "
                  f"(d_mid={d_mid:.2f}, shortfall={shortfall})")

    return thinned
