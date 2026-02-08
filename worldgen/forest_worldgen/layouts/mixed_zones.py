"""
Mixed zones layout - multiple rectangular regions with different distributions
"""

from ..config import load_distribution
from ..patterns import PATTERN_SAMPLERS


def generate_mixed_zones(layout_config, world_config, project_root):
    """
    Each zone has its own region, count, and distribution
    
    Returns list of (x, y) positions
    """
    K = world_config['generation']['area_size']
    min_distance = world_config['generation']['min_distance']
    all_positions = []

    for zone in layout_config['layout']['zones']:
        region = zone['region']
        count = zone['count']
        dist_ref = zone['distribution_ref']
        dist_config = load_distribution(dist_ref, project_root)
        dist_type = dist_config['distribution']['type']
        dist_params = dist_config['distribution'].get('params', {}) or {}

        sampler = PATTERN_SAMPLERS.get(dist_type)
        if sampler is None:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        points = sampler(count, region, K, min_distance, all_positions, dist_params)
        all_positions.extend(points)
        print(f"  zone '{zone['name']}': placed {len(points)} objects ({dist_type})")

    return all_positions
