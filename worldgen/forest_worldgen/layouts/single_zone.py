"""
Single zone layout - simple uniform distribution across entire world
Fallback when no layout is specified
"""

from ..patterns import PATTERN_SAMPLERS


def generate_single_zone(layout_config, world_config, project_root):
    """
    Fallback when no layout is specified - uses generation.object_count with plain CSR
    
    Returns list of (x, y) positions
    """
    K = world_config['generation']['area_size']
    N = world_config['generation']['object_count']
    min_distance = world_config['generation']['min_distance']
    
    sampler = PATTERN_SAMPLERS['csr']
    positions = sampler(N, None, K, min_distance, [], {'use_world_min_distance': True})
    
    print(f"  single zone: placed {len(positions)} objects (csr)")
    return positions
