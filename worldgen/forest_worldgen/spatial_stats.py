"""
Spatial statistics for validating point patterns.

Computes post-generation diagnostics:
  - Clark-Evans nearest-neighbor index  R
  - Pair correlation function            g(r)
  - Ripley's L-function deviation        L(r) - r

All functions operate on a plain list of (x, y) tuples inside a square
domain [-L/2, L/2]^2 and return JSON-serialisable dicts / lists.
"""

import math

try:
    import numpy as np
    from scipy.spatial import KDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _nn_distances(positions):
    """Return list of nearest-neighbour distances.
    
    Uses scipy KDTree for O(n log n) when available, falls back to
    brute-force O(n²) otherwise.
    """
    n = len(positions)
    if n < 2:
        return []

    if _HAS_SCIPY:
        pts = np.asarray(positions, dtype=np.float64)
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=2)  # k=2: closest is self (dist=0)
        return dists[:, 1].tolist()

    # brute-force fallback
    nn = []
    for i in range(n):
        xi, yi = positions[i]
        best = math.inf
        for j in range(n):
            if i == j:
                continue
            d = math.hypot(xi - positions[j][0], yi - positions[j][1])
            if d < best:
                best = d
        nn.append(best)
    return nn


def _pairwise_distances(positions, r_max=None):
    """Return flat list of pairwise distances (no duplicates).
    
    When *r_max* is given and scipy is available, uses KDTree.sparse_distance_matrix
    to avoid materialising all n*(n-1)/2 pairs — only pairs within r_max are stored.
    """
    n = len(positions)

    if _HAS_SCIPY and r_max is not None and r_max > 0:
        pts = np.asarray(positions, dtype=np.float64)
        tree = KDTree(pts)
        sparse = tree.sparse_distance_matrix(tree, r_max, output_type='ndarray')
        # sparse is structured array with (i, j, v); i < j not guaranteed
        dists = [float(row[2]) for row in sparse if row[0] < row[1]]
        return dists

    # full brute-force fallback (or when r_max is None)
    dists = []
    for i in range(n):
        xi, yi = positions[i]
        for j in range(i + 1, n):
            dists.append(math.hypot(xi - positions[j][0], yi - positions[j][1]))
    return dists


# ---------------------------------------------------------------------------
# Clark-Evans R
# ---------------------------------------------------------------------------

def clark_evans_R(positions, area):
    """
    Clark-Evans nearest-neighbour index.

    R = mean_obs_nn / mean_csr_nn
    where mean_csr_nn = 1 / (2 * sqrt(density))

    Returns:
        float  R  (>1 regular, ~1 random, <1 clustered)
    """
    nn = _nn_distances(positions)
    if not nn or area <= 0:
        return float('nan')
    mean_obs = sum(nn) / len(nn)
    density = len(positions) / area
    mean_csr = 0.5 / math.sqrt(density)
    if mean_csr <= 0:
        return float('nan')
    return mean_obs / mean_csr


# ---------------------------------------------------------------------------
# Pair correlation  g(r)
# ---------------------------------------------------------------------------

def pair_correlation(positions, area, r_max=None, n_bins=25):
    """
    Estimate the pair correlation function g(r) in annular bins.

    g(r) = K'(r) / (2 * pi * r)   where K'(r) is the derivative of K,
    but we estimate directly from pair counts in annular rings.

    Returns:
        list of dicts  [{r, g}, ...]   one per bin centre.
    """
    n = len(positions)
    if n < 2 or area <= 0:
        return []

    density = n / area

    if r_max is None:
        # need all distances to determine r_max; fall back to full computation
        dists = _pairwise_distances(positions)
        if not dists:
            return []
        r_max = max(dists) * 0.5  # don't trust edge region
    else:
        dists = _pairwise_distances(positions, r_max=r_max)
        if not dists:
            return []

    dr = r_max / n_bins
    if dr <= 0:
        return []

    counts = [0] * n_bins
    for d in dists:
        idx = int(d / dr)
        if 0 <= idx < n_bins:
            counts[idx] += 1

    results = []
    for b in range(n_bins):
        r_lo = b * dr
        r_hi = (b + 1) * dr
        r_mid = (r_lo + r_hi) / 2.0
        ring_area = math.pi * (r_hi ** 2 - r_lo ** 2)
        if ring_area <= 0:
            continue
        # each pair counted once; expected under CSR = n*(n-1)/2 * ring_area/area
        expected = 0.5 * n * (n - 1) * ring_area / area
        g = counts[b] / expected if expected > 0 else 0.0
        results.append({'r': round(r_mid, 4), 'g': round(g, 4)})

    return results


# ---------------------------------------------------------------------------
# Ripley's L(r) - r
# ---------------------------------------------------------------------------

def ripley_L_minus_r(positions, area, r_max=None, n_bins=25):
    """
    Compute L(r) - r  for a set of evaluation radii.

    K(r) = area / n^2  *  sum_{i!=j} 1(d_ij <= r)
    L(r) = sqrt(K(r) / pi)
    deviation = L(r) - r

    Returns:
        list of dicts  [{r, L_minus_r}, ...]
    """
    n = len(positions)
    if n < 2 or area <= 0:
        return []

    if r_max is None:
        half_side = math.sqrt(area) / 2.0
        r_max = half_side * 0.5

    dists = _pairwise_distances(positions, r_max=r_max)
    dists.sort()

    dr = r_max / n_bins
    if dr <= 0:
        return []

    results = []
    pair_idx = 0
    cum_count = 0  # cumulative count of pairs with dist <= r

    for b in range(1, n_bins + 1):
        r = b * dr
        while pair_idx < len(dists) and dists[pair_idx] <= r:
            cum_count += 1
            pair_idx += 1
        # each pair counted once in dists; K uses ordered pairs so multiply by 2
        K = area / (n * n) * (2 * cum_count)
        L = math.sqrt(K / math.pi) if K >= 0 else 0.0
        results.append({'r': round(r, 4), 'L_minus_r': round(L - r, 4)})

    return results


# ---------------------------------------------------------------------------
# Public convenience: compute all stats at once
# ---------------------------------------------------------------------------

def compute_validation_stats(positions, area_size):
    """
    Compute a full suite of spatial statistics for the generated positions.

    Parameters:
        positions : list of (x, y)
        area_size : side length K of the square world

    Returns:
        dict ready for JSON serialisation.
    """
    area = area_size ** 2
    R = clark_evans_R(positions, area)

    # choose r_max relative to world; ~ 1/4 of the side
    r_max = area_size * 0.25

    g = pair_correlation(positions, area, r_max=r_max, n_bins=20)
    L = ripley_L_minus_r(positions, area, r_max=r_max, n_bins=20)

    # summarise g(r) at small r  (first 5 bins)
    g_small = g[:5] if g else []
    g_small_mean = (
        sum(item['g'] for item in g_small) / len(g_small) if g_small else float('nan')
    )

    # summarise L(r)-r at small r
    L_small = L[:5] if L else []
    L_small_mean = (
        sum(item['L_minus_r'] for item in L_small) / len(L_small)
        if L_small
        else float('nan')
    )

    return {
        'clark_evans_R': round(R, 4) if not math.isnan(R) else None,
        'g_small_r_mean': round(g_small_mean, 4) if not math.isnan(g_small_mean) else None,
        'L_small_r_mean': round(L_small_mean, 4) if not math.isnan(L_small_mean) else None,
        'pair_correlation': g,
        'ripley_L_minus_r': L,
    }
