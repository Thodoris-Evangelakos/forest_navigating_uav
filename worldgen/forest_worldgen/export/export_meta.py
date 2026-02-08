"""
Metadata export functionality
"""

import json
from datetime import datetime


def export_meta(positions, world_config, layout_config, output_path, seed=None):
    """
    Export metadata about the generated world to JSON
    """
    gen_config = world_config['generation']
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'world': {
            'name': world_config['world_name'],
            'area_size': gen_config['area_size'],
            'min_distance': gen_config['min_distance'],
            'tree_height_range': [
                gen_config['min_tree_height'],
                gen_config['max_tree_height']
            ],
        },
        'layout': {
            'type': layout_config['layout']['type'] if layout_config else 'single',
        },
        'statistics': {
            'total_objects': len(positions),
            'density': len(positions) / (gen_config['area_size'] ** 2),
        },
        'positions': [
            {'x': float(x), 'y': float(y)} 
            for x, y in positions
        ],
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
