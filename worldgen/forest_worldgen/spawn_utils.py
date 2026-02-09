"""
Spawn point utilities for placing UAVs at world edges.

Provides functions to generate spawn locations along the perimeter of a world
based on configuration, with optional margins from boundaries.
"""

import json
import math
import random
from datetime import datetime
from .config import load_config, resolve_path


def calculate_edge_spawns(area_size, margin=1.0, count=4, distribution='uniform', z_height=2.0):
    """
    Calculate spawn points along the edge of a square world.
    
    Parameters
    ----------
    area_size : float
        Side length of the square world (e.g., 50 meters).
        World spans [-area_size/2, area_size/2] in x and y.
    margin : float
        Distance inset from the boundary edge (default 1.0 m).
    count : int
        Number of spawn points to generate (default 4, one per edge).
    distribution : str
        'uniform' - evenly spaced along perimeter
        'random' - randomly distributed along perimeter
        (default 'uniform')
    z_height : float
        Height above ground for spawn pose (default 2.0 m).
    
    Returns
    -------
    list[dict]
        Each dict has keys: x, y, z
    """
    half_side = area_size / 2.0
    edge_x = half_side - margin
    edge_y = half_side - margin
    
    spawns = []
    
    if distribution == 'random':
        for _ in range(count):
            # randomly choose which edge
            edge = random.choice(['north', 'south', 'east', 'west'])
            
            if edge == 'north':
                x = random.uniform(-edge_x, edge_x)
                y = edge_y
            elif edge == 'south':
                x = random.uniform(-edge_x, edge_x)
                y = -edge_y
            elif edge == 'east':
                x = edge_x
                y = random.uniform(-edge_y, edge_y)
            else:  # west
                x = -edge_x
                y = random.uniform(-edge_y, edge_y)
            
            spawns.append({'x': x, 'y': y, 'z': z_height})
    else:
        # uniform distribution: count/4 per edge, evenly spaced
        per_edge = max(1, count // 4)
        extra = count % 4
        
        edges = [
            ('north', edge_y, lambda i, n: (-edge_x + i * 2 * edge_x / n, edge_y)),
            ('south', -edge_y, lambda i, n: (-edge_x + i * 2 * edge_x / n, -edge_y)),
            ('east', edge_x, lambda i, n: (edge_x, -edge_y + i * 2 * edge_y / n)),
            ('west', -edge_x, lambda i, n: (-edge_x, -edge_y + i * 2 * edge_y / n)),
        ]
        
        edge_idx = 0
        for i in range(count):
            if edge_idx < len(edges):
                name, _, pos_fn = edges[edge_idx]
                n_on_edge = per_edge + (1 if extra > 0 else 0)
                local_idx = i % n_on_edge
                x, y = pos_fn(local_idx, n_on_edge)
                
                spawns.append({'x': x, 'y': y, 'z': z_height})
                
                if (i + 1) % n_on_edge == 0:
                    edge_idx += 1
                    if extra > 0:
                        extra -= 1
    
    return spawns


def generate_spawn_points(config_file, world_file=None, margin=1.0, z_height=2.0,
                         count=4, distribution='uniform', seed=None):
    """
    Generate spawn points for a given world configuration.
    
    This function reads the world config, calculates edge-based spawn locations,
    and optionally saves results to JSON.
    
    Parameters
    ----------
    config_file : str
        Path to worldgen_run.yaml or world.default.yaml
    world_file : str, optional
        Path to generated world.sdf (informational only, not used for calculations)
    margin : float
        Distance inset from boundary (default 1.0 m)
    z_height : float
        Height above ground for spawns (default 2.0 m)
    count : int
        Number of spawn points (default 4)
    distribution : str
        'uniform' or 'random' distribution strategy (default 'uniform')
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Spawn metadata with keys: spawn_points, area_size, margin, z_height,
        count, distribution, seed, timestamp
    """
    if seed is not None:
        random.seed(seed)
    
    # Load config to get area_size
    import os
    config_path = os.path.abspath(config_file)
    config = load_config(config_path)
    
    # Determine project root (same way generate_world.py does it)
    # We are in worldgen/forest_worldgen/, so ../../ is project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # Handle both worldgen_run.yaml and legacy world.default.yaml
    if 'include' in config:
        # worldgen_run.yaml style
        world_path = resolve_path(config['include']['world'], project_root)
        world_config = load_config(world_path)
    else:
        # legacy single-file mode
        world_config = config
    
    area_size = world_config['generation']['area_size']
    
    # Calculate spawn points
    spawn_points = calculate_edge_spawns(
        area_size,
        margin=margin,
        count=count,
        distribution=distribution,
        z_height=z_height
    )
    
    return {
        'spawn_points': spawn_points,
        'area_size': area_size,
        'margin': margin,
        'z_height': z_height,
        'count': count,
        'distribution': distribution,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
    }


def save_spawn_metadata(spawn_meta, output_path):
    """
    Save spawn metadata to a JSON file.
    
    Parameters
    ----------
    spawn_meta : dict
        Spawn metadata dictionary (from generate_spawn_points)
    output_path : str
        Path to write JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(spawn_meta, f, indent=2)


def main():
    """
    Command-line interface for spawn point generation.
    
    Usage:
        python3 -m worldgen.forest_worldgen.spawn_utils <config_file> [--seed SEED] [--margin MARGIN] [--height HEIGHT] [--count COUNT] [--dist DIST]
    """
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python3 -m worldgen.forest_worldgen.spawn_utils <config_file> [--seed SEED] [--margin MARGIN] [--height HEIGHT] [--count COUNT] [--dist DIST]")
        sys.exit(1)
    
    config_file = os.path.abspath(sys.argv[1])
    
    # Parse optional arguments
    seed = None
    margin = 1.0
    z_height = 2.0
    count = 4
    distribution = 'uniform'
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--seed' and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--margin' and i + 1 < len(sys.argv):
            margin = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--height' and i + 1 < len(sys.argv):
            z_height = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--count' and i + 1 < len(sys.argv):
            count = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--dist' and i + 1 < len(sys.argv):
            distribution = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    try:
        spawn_meta = generate_spawn_points(
            config_file,
            margin=margin,
            z_height=z_height,
            count=count,
            distribution=distribution,
            seed=seed
        )
        
        print(f"Spawn points calculated for area_size={spawn_meta['area_size']}")
        print(f"Generated {len(spawn_meta['spawn_points'])} spawn points:")
        for i, pt in enumerate(spawn_meta['spawn_points']):
            print(f"  [{i}] x={pt['x']:.2f}, y={pt['y']:.2f}, z={pt['z']:.2f}")
        
        # Optionally save to JSON
        output_path = os.path.join(
            os.path.dirname(config_file),
            '..', '..', 'worldgen', 'outputs', 'latest', 'spawn_points.json'
        )
        output_path = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_spawn_metadata(spawn_meta, output_path)
        print(f"Spawn metadata saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
