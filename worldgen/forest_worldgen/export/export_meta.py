"""Export metadata and validation statistics for generated worlds."""

import json
from datetime import datetime

from ..spatial_stats import compute_validation_stats


def export_meta(
    positions, world_config, layout_config, output_path, seed=None, start_goal_anchors=None
):
    """
    Export metadata about the generated world to JSON.

    Includes per-pattern spatial statistics (Clark-Evans R, g(r), L(r)-r)
    computed *after* generation so they serve as validation, not constraints.
    """
    gen_config = world_config["generation"]
    area_size = gen_config["area_size"]

    # --- validation statistics ---
    validation = compute_validation_stats(positions, area_size)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "world": {
            "name": world_config["world_name"],
            "area_size": area_size,
            "min_distance": gen_config["min_distance"],
            "tree_height_range": [gen_config["min_tree_height"], gen_config["max_tree_height"]],
        },
        "layout": {
            "type": layout_config["layout"]["type"] if layout_config else "single",
        },
        "statistics": {
            "total_objects": len(positions),
            "density": len(positions) / (area_size**2),
        },
        "validation": {
            "clark_evans_R": validation["clark_evans_R"],
            "g_small_r_mean": validation["g_small_r_mean"],
            "L_small_r_mean": validation["L_small_r_mean"],
            "pair_correlation": validation["pair_correlation"],
            "ripley_L_minus_r": validation["ripley_L_minus_r"],
        },
        "positions": [{"x": float(x), "y": float(y)} for x, y in positions],
    }

    if start_goal_anchors is not None:
        metadata["start_goal"] = {
            "start_xy": [
                float(start_goal_anchors["start_xy"][0]),
                float(start_goal_anchors["start_xy"][1]),
            ],
            "goal_xy": [
                float(start_goal_anchors["goal_xy"][0]),
                float(start_goal_anchors["goal_xy"][1]),
            ],
            "tree_exclusion_radius": float(start_goal_anchors["exclusion_radius"]),
            "removed_objects": int(start_goal_anchors["removed_objects"]),
        }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
