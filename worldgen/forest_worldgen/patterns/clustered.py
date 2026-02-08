"""
Thomas-like cluster process
Creates spatial clusters with Gaussian scatter around parent points
"""

import random
import math
from .csr import sample_csr, _point_in_rect, _point_in_area, _check_min_distance


def sample_clustered(count, region, K, min_distance, existing_positions, params):
    """
    Place cluster_count parent points, then scatter children around each parent
    A background_fraction of points is placed uniformly (CSR) to fill gaps
    
    params:
      - cluster_count: number of cluster centers (default: 5)
      - cluster_radius: radius of each cluster (default: 3.0)
      - background_fraction: fraction of points placed uniformly (default: 0.15)
    
    Returns list of (x, y) positions
    """
    cluster_count = params.get('cluster_count', 5)
    cluster_radius = params.get('cluster_radius', 3.0)
    bg_fraction = params.get('background_fraction', 0.15)

    n_background = max(1, int(count * bg_fraction))
    n_clustered = count - n_background

    # distribute clustered points evenly-ish across parents
    per_cluster = max(1, n_clustered // cluster_count)
    remainder = n_clustered - per_cluster * cluster_count

    positions = []

    # generate parent centres
    parents = []
    for _ in range(cluster_count):
        for _ in range(200):
            if region is not None:
                cx, cy = _point_in_rect(region)
            else:
                cx, cy = _point_in_area(K)
            if _check_min_distance(cx, cy, [p for p in parents], cluster_radius * 0.5):
                parents.append((cx, cy))
                break
        else:
            if region is not None:
                cx, cy = _point_in_rect(region)
            else:
                cx, cy = _point_in_area(K)
            parents.append((cx, cy))

    # scatter children around each parent
    for idx, (cx, cy) in enumerate(parents):
        n = per_cluster + (1 if idx < remainder else 0)
        for _ in range(n):
            for _ in range(200):
                angle = random.uniform(0, 2 * math.pi)
                r = random.gauss(0, cluster_radius / 2)
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                # clamp inside region / area
                if region is not None:
                    x = max(region['x_min'], min(region['x_max'], x))
                    y = max(region['y_min'], min(region['y_max'], y))
                else:
                    x = max(-K / 2, min(K / 2, x))
                    y = max(-K / 2, min(K / 2, y))
                if _check_min_distance(x, y, existing_positions + positions, min_distance):
                    positions.append((x, y))
                    break
            else:
                positions.append((x, y))

    # background fill  (use world min_distance as overlap guard)
    bg = sample_csr(n_background, region, K, min_distance,
                    existing_positions + positions,
                    {'use_world_min_distance': True})
    positions.extend(bg)

    return positions
