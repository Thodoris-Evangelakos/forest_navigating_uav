"""
Complete Spatial Randomness (CSR) pattern
Uniform random placement with optional min-distance constraint
"""

import random
import math


def _point_in_rect(region):
    """Sample a uniform random point inside a rectangular region"""
    x = random.uniform(region['x_min'], region['x_max'])
    y = random.uniform(region['y_min'], region['y_max'])
    return x, y


def _point_in_area(K):
    """Sample a uniform random point inside a K*K area centered at origin"""
    return random.uniform(-K / 2, K / 2), random.uniform(-K / 2, K / 2)


def _check_min_distance(x, y, positions, min_distance):
    """Return True if (x, y) is at least min_distance from every position"""
    for px, py in positions:
        if math.hypot(x - px, y - py) < min_distance:
            return False
    return True


def sample_csr(count, region, K, min_distance, existing_positions, _params):
    """
    Uniform random placement with optional min-distance rejection sampling
    
    count: how many points to generate
    region: optional dict with x_min, x_max, y_min, y_max (None = use K)
    K: world size (K*K area centered at origin)
    min_distance: minimum distance between points
    existing_positions: list of (x, y) already placed
    _params: distribution parameters (unused for CSR)
    
    Returns list of (x, y) positions
    """
    positions = []
    max_attempts = 200

    for _ in range(count):
        placed = False
        for _ in range(max_attempts):
            if region is not None:
                x, y = _point_in_rect(region)
            else:
                x, y = _point_in_area(K)

            if _check_min_distance(x, y, existing_positions + positions, min_distance):
                positions.append((x, y))
                placed = True
                break

        if not placed:
            # fall back - place anyway to keep count
            if region is not None:
                x, y = _point_in_rect(region)
            else:
                x, y = _point_in_area(K)
            positions.append((x, y))
            print(f"Warning: relaxed min_distance for point #{len(positions)}")

    return positions
