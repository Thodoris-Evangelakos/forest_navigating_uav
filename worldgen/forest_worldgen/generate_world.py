"""
Main world generation script
Orchestrates configuration loading, position generation, and export
"""

import os
import sys
import random
import shutil
import io
from contextlib import contextmanager
from contextlib import redirect_stdout
from datetime import datetime

from .config import load_config, load_template, resolve_path
from .layouts import LAYOUT_HANDLERS, generate_single_zone
from .export import export_sdf, export_meta, generate_preview
from .spatial_stats import compute_validation_stats


@contextmanager
def _local_random_seed(seed):
    if seed is None:
        yield
        return

    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def _load_world_and_layout_configs(config_file):
    """Resolve world/layout configs from a worldgen run-config."""
    config_file = os.path.abspath(config_file)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # hella junky, I should use a proper structure
    # HACK: assume project root is two levels up from this script
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    run_config = load_config(config_file)
    include_cfg = run_config.get('include')
    if not isinstance(include_cfg, dict):
        raise ValueError(
            "Invalid worldgen run config: missing 'include' mapping. "
            "Use configs/worldgen/worldgen_run.yaml-style config."
        )

    world_ref = include_cfg.get('world')
    if not world_ref:
        raise ValueError("Invalid worldgen run config: include.world is required")

    world_path = resolve_path(world_ref, project_root)
    world_config = load_config(world_path)

    layout_ref = include_cfg.get('layout')
    layout_config = None
    if layout_ref:
        layout_path = resolve_path(layout_ref, project_root)
        layout_config = load_config(layout_path)

    return world_config, layout_config, project_root


def _sample_start_goal_anchors(
    area_size,
    min_start_goal_distance=8.0,
    band_ratio=0.55,
    max_attempts=500,
):
    """Sample FastSim-like opposite-band start/goal anchor points."""
    half = float(area_size) / 2.0
    band_inner = float(max(0.0, min(1.0, band_ratio))) * half

    for _ in range(max(1, int(max_attempts))):
        axis = int(random.randint(0, 1))
        start_side = -1.0 if bool(random.randint(0, 1)) else 1.0
        goal_side = -start_side

        if axis == 0:
            start_x = random.uniform(start_side * band_inner, start_side * half)
            start_y = random.uniform(-half, half)
            goal_x = random.uniform(goal_side * band_inner, goal_side * half)
            goal_y = random.uniform(-half, half)
        else:
            start_x = random.uniform(-half, half)
            start_y = random.uniform(start_side * band_inner, start_side * half)
            goal_x = random.uniform(-half, half)
            goal_y = random.uniform(goal_side * band_inner, goal_side * half)

        dx = float(goal_x - start_x)
        dy = float(goal_y - start_y)
        if (dx * dx + dy * dy) ** 0.5 >= float(min_start_goal_distance):
            return (float(start_x), float(start_y)), (float(goal_x), float(goal_y))

    return (float(band_inner), 0.0), (-float(band_inner), 0.0)


def _exclude_positions_near_anchors(positions, anchors, exclusion_radius):
    """Drop obstacle points near start/goal anchors."""
    if not positions:
        return positions, 0
    if exclusion_radius <= 0.0:
        return positions, 0

    keep = []
    removed = 0
    radius_sq = float(exclusion_radius) ** 2

    for x, y in positions:
        is_clear = True
        for ax, ay in anchors:
            dx = float(x) - float(ax)
            dy = float(y) - float(ay)
            if (dx * dx + dy * dy) < radius_sq:
                is_clear = False
                break
        if is_clear:
            keep.append((x, y))
        else:
            removed += 1

    return keep, removed


def generate_positions_from_config(
    config_file,
    seed=None,
    verbose=True,
    apply_start_goal_exclusion=False,
):
    """Generate world positions in-memory without file export (pure memory API)

    Args:
        config_file (string): Path to worldgen run config
        seed (int, optional): Seed to be used. Defaults to None.

    Raises:
        ValueError: Raised if an unknown layout type is specified in the layout config

    Returns:
        tuple[list[tuple[float, float]], dict, dict | None]: Generated positions, world config, and optional layout config.
    """

    """Generate world positions in-memory without exporting files.

    Parameters
    ----------
    config_file : str
        Path to worldgen run config.
    seed : int | None
        Optional deterministic seed for generation.

    Returns
    -------
    tuple[list[tuple[float, float]], dict, dict | None]
        Generated positions, world config, and optional layout config.
    """
    world_config, layout_config, project_root = _load_world_and_layout_configs(config_file)

    with _local_random_seed(seed):
        if not bool(verbose):
            with redirect_stdout(io.StringIO()):
                if layout_config is not None:
                    layout_type = layout_config['layout']['type']
                    handler = LAYOUT_HANDLERS.get(layout_type)
                    if handler is None:
                        raise ValueError(f"Unknown layout type: {layout_type}")
                    positions = handler(layout_config, world_config, project_root)
                else:
                    positions = generate_single_zone(None, world_config, project_root)
        else:
            if layout_config is not None:
                layout_type = layout_config['layout']['type']
                handler = LAYOUT_HANDLERS.get(layout_type)
                if handler is None:
                    raise ValueError(f"Unknown layout type: {layout_type}")
                positions = handler(layout_config, world_config, project_root)
            else:
                positions = generate_single_zone(None, world_config, project_root)

    if bool(apply_start_goal_exclusion):
        gen = world_config.get('generation', {})
        exclusion_radius = float(gen.get('start_goal_tree_exclusion_radius', 0.0))
        if exclusion_radius > 0.0:
            anchors = _sample_start_goal_anchors(
                area_size=float(gen.get('area_size', 50.0)),
                min_start_goal_distance=float(gen.get('start_goal_min_distance', 8.0)),
                band_ratio=float(gen.get('start_goal_band_ratio', 0.55)),
                max_attempts=int(gen.get('spawn_max_attempts', 500)),
            )
            positions, removed = _exclude_positions_near_anchors(positions, anchors, exclusion_radius)
            world_config['_start_goal_anchors'] = {
                'start_xy': [float(anchors[0][0]), float(anchors[0][1])],
                'goal_xy': [float(anchors[1][0]), float(anchors[1][1])],
                'exclusion_radius': exclusion_radius,
                'removed_objects': int(removed),
            }

    return positions, world_config, layout_config


def main():
    """Driver code for world generation
        Accepts a worldgen_run.yaml (with 'include' key referencing world + layout)
    """
    if len(sys.argv) < 2:
        print("Usage: python3 -m forest_worldgen.generate_world <run_config> [--seed SEED]")
        sys.exit(1)

    config_file = os.path.abspath(sys.argv[1])
    
    # Parse optional seed
    seed = None
    if len(sys.argv) > 2 and sys.argv[2] == '--seed' and len(sys.argv) > 3:
        seed = int(sys.argv[3])
        random.seed(seed)
        print(f"Using random seed: {seed}")

    # Determine project root and worldgen root
    # HACK: assume project root is two levels up from this script, and worldgen root is one level up
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    worldgen_root = os.path.abspath(os.path.join(script_dir, '..'))

    try:
        run_config = load_config(config_file)
        include_cfg = run_config.get('include')
        if not isinstance(include_cfg, dict):
            raise ValueError(
                "Invalid worldgen run config: missing 'include' mapping. "
                "Use configs/worldgen/worldgen_run.yaml-style config."
            )

        world_ref = include_cfg.get('world')
        if not world_ref:
            raise ValueError("Invalid worldgen run config: include.world is required")

        world_path = resolve_path(world_ref, project_root)
        print(f"world config  : {world_path}")
        layout_ref = include_cfg.get('layout')
        if layout_ref:
            layout_path = resolve_path(layout_ref, project_root)
            print(f"layout config : {layout_path}")

        # load templates
        world_template = load_template('world_base.sdf', worldgen_root)
        include_template = load_template('include.sdf', worldgen_root)

        positions, world_config, layout_config = generate_positions_from_config(
            config_file,
            seed=seed,
            apply_start_goal_exclusion=True,
        )
        if layout_config is not None:
            print(f"layout type   : {layout_config['layout']['type']}")

        print(f"total objects : {len(positions)}")
        start_goal_meta = world_config.get('_start_goal_anchors')
        if start_goal_meta is not None:
            sx, sy = start_goal_meta['start_xy']
            gx, gy = start_goal_meta['goal_xy']
            rr = start_goal_meta['exclusion_radius']
            removed = start_goal_meta['removed_objects']
            print(
                "start/goal    : "
                f"start=({sx:.2f},{sy:.2f}) goal=({gx:.2f},{gy:.2f}) "
                f"clearance_r={rr:.2f} removed={removed}"
            )

        # validation statistics (logged, not enforced)
        area_size = world_config['generation']['area_size']
        stats = compute_validation_stats(positions, area_size)
        R = stats['clark_evans_R']
        g_s = stats['g_small_r_mean']
        L_s = stats['L_small_r_mean']
        print(f"validation    : R={R}  g_small={g_s}  L_small={L_s}")

        # prepare output directories
        # always save to timestamped run directory and mirror to outputs/latest
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        if seed is not None:
            run_name = f'{timestamp}_seed{seed:04d}'
        else:
            run_name = f'{timestamp}_random'

        run_dir = os.path.join(worldgen_root, 'outputs', 'runs', run_name)
        latest_dir = os.path.join(worldgen_root, 'outputs', 'latest')
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)

        # --- Export SDF ---
        sdf_path = os.path.join(run_dir, 'world.sdf')
        export_sdf(positions, world_config, world_template, include_template, sdf_path)
        print(f"generated sdf : {sdf_path}")

        latest_sdf_path = os.path.join(latest_dir, 'world.sdf')
        shutil.copy2(sdf_path, latest_sdf_path)
        print(f"latest sdf    : {latest_sdf_path}")

        # --- Export metadata ---
        meta_path = os.path.join(run_dir, 'meta.json')
        export_meta(
            positions,
            world_config,
            layout_config,
            meta_path,
            seed,
            start_goal_anchors=start_goal_meta,
        )
        print(f"metadata      : {meta_path}")

        latest_meta_path = os.path.join(latest_dir, 'meta.json')
        shutil.copy2(meta_path, latest_meta_path)
        print(f"latest meta   : {latest_meta_path}")

        # --- Generate preview ---
        preview_path = os.path.join(run_dir, 'preview.png')
        generate_preview(positions, world_config, preview_path)
        print(f"preview       : {preview_path}")

        latest_preview_path = os.path.join(latest_dir, 'preview.png')
        shutil.copy2(preview_path, latest_preview_path)
        print(f"latest prev   : {latest_preview_path}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
