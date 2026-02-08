"""
Regular / inhibitory process (Poisson-disc style)
Enforces stronger minimum distance constraints
"""

from .csr import sample_csr


def sample_regular(count, region, K, min_distance, existing_positions, params):
    """
    Poisson-disc style placement with stronger spacing
    Uses min_distance from params if provided, otherwise falls back to world-level min_distance
    
    Returns list of (x, y) positions
    """
    dist_min = params.get('min_distance', min_distance)
    return sample_csr(count, region, K, dist_min, existing_positions, params)
