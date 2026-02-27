"""Generate a single-zone fallback layout across the full world."""

from ..patterns import PATTERN_SAMPLERS


def generate_single_zone(layout_config, world_config, project_root):
    """Generate CSR samples using world-level object count and spacing settings."""
    K = world_config["generation"]["area_size"]
    N = world_config["generation"]["object_count"]
    min_distance = world_config["generation"]["min_distance"]

    sampler = PATTERN_SAMPLERS["csr"]
    positions = sampler(N, None, K, min_distance, [], {"use_world_min_distance": True})

    print(f"  single zone: placed {len(positions)} objects (csr)")
    return positions
