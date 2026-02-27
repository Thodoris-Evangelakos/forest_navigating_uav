"""Export generated object placements to SDF world files."""

import random
import math


def positions_to_includes(positions, model_uris, min_height, max_height, include_template):
    """Convert XY positions into concatenated SDF ``<include>`` blocks."""
    n = len(positions)
    # Pre-generate random choices in bulk for less per-iteration overhead
    uris = [model_uris[int(random.random() * len(model_uris))] for _ in range(n)]
    yaws = [random.uniform(-math.pi, math.pi) for _ in range(n)]

    parts = []
    for i, (x, y) in enumerate(positions):
        parts.append(
            include_template.format(
                name=f"object_{i}",
                x=f"{x:.2f}",
                y=f"{y:.2f}",
                z="0.00",
                yaw=f"{yaws[i]:.2f}",
                uri=uris[i],
            )
        )

    return "\n".join(parts)


def build_world_sdf(include_blocks, config, world_template):
    """Build and return a complete world SDF from template data."""
    lighting = config["lighting"]
    physics = config["physics"]
    generation = config["generation"]

    # Scale ground plane to be larger than world area to prevent falling through edges
    # Use 1.5x the area_size to provide margin beyond world boundaries
    ground_size = generation["area_size"] * 1.5

    return world_template.format(
        world_name=config["world_name"],
        max_step_size=physics["max_step_size"],
        real_time_factor=physics["real_time_factor"],
        cast_shadows="true" if lighting["cast_shadows"] else "false",
        light_pose=lighting["pose"],
        light_diffuse=lighting["diffuse"],
        light_specular=lighting["specular"],
        light_range=lighting["attenuation"]["range"],
        light_constant=lighting["attenuation"]["constant"],
        light_linear=lighting["attenuation"]["linear"],
        light_quadratic=lighting["attenuation"]["quadratic"],
        light_direction=lighting["direction"],
        ground_size=ground_size,
        include_blocks=include_blocks,
    )


def export_sdf(positions, world_config, world_template, include_template, output_path):
    """Export positions as an SDF world file at ``output_path``."""
    gen_config = world_config["generation"]
    min_height = gen_config["min_tree_height"]
    max_height = gen_config["max_tree_height"]
    model_uris = [m["uri"] for m in world_config["models"]]

    include_blocks = positions_to_includes(
        positions, model_uris, min_height, max_height, include_template
    )
    world_sdf = build_world_sdf(include_blocks, world_config, world_template)

    with open(output_path, "w") as f:
        f.write(world_sdf)
