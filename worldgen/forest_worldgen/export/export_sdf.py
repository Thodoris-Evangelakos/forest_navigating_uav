"""
SDF export functionality
"""

import random
import math


def positions_to_includes(positions, model_uris, min_height, max_height, include_template):
    """
    Turn a list of (x, y) positions into SDF <include> blocks
    
    Returns string of concatenated include blocks
    """
    includes = []
    for i, (x, y) in enumerate(positions):
        uri = random.choice(model_uris)
        z = 0
        height_scale = random.uniform(min_height, max_height)
        yaw = random.uniform(-math.pi, math.pi)

        include_block = include_template.format(
            name=f"object_{i}",
            x=f"{x:.2f}",
            y=f"{y:.2f}",
            z=f"{z:.2f}",
            yaw=f"{yaw:.2f}",
            uri=uri,
        )
        includes.append(include_block)

    return "\n".join(includes)


def build_world_sdf(include_blocks, config, world_template):
    """
    Build the complete world SDF from template and configuration
    
    Returns complete SDF string
    """
    lighting = config['lighting']
    physics = config['physics']

    return world_template.format(
        world_name=config['world_name'],
        max_step_size=physics['max_step_size'],
        real_time_factor=physics['real_time_factor'],
        cast_shadows='true' if lighting['cast_shadows'] else 'false',
        light_pose=lighting['pose'],
        light_diffuse=lighting['diffuse'],
        light_specular=lighting['specular'],
        light_range=lighting['attenuation']['range'],
        light_constant=lighting['attenuation']['constant'],
        light_linear=lighting['attenuation']['linear'],
        light_quadratic=lighting['attenuation']['quadratic'],
        light_direction=lighting['direction'],
        include_blocks=include_blocks,
    )


def export_sdf(positions, world_config, world_template, include_template, output_path):
    """
    Export positions as SDF world file
    
    :param positions: List of (x, y) tuples
    :param world_config: World configuration dict
    :param world_template: World SDF template string
    :param include_template: Include SDF template string
    :param output_path: Path to write SDF file
    """
    gen_config = world_config['generation']
    min_height = gen_config['min_tree_height']
    max_height = gen_config['max_tree_height']
    model_uris = [m['uri'] for m in world_config['models']]

    include_blocks = positions_to_includes(
        positions, model_uris, min_height, max_height, include_template
    )
    world_sdf = build_world_sdf(include_blocks, world_config, world_template)

    with open(output_path, "w") as f:
        f.write(world_sdf)
