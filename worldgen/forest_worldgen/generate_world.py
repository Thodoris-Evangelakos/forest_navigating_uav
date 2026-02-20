"""
Main world generation script
Orchestrates configuration loading, position generation, and export
"""

import os
import sys
import random
import shutil
from contextlib import contextmanager
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
    """Resolve world/layout configs from either run-config or legacy world config."""
    config_file = os.path.abspath(config_file)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    run_config = load_config(config_file)
    if 'include' in run_config:
        world_path = resolve_path(run_config['include']['world'], project_root)
        world_config = load_config(world_path)

        layout_ref = run_config['include'].get('layout')
        layout_config = None
        if layout_ref:
            layout_path = resolve_path(layout_ref, project_root)
            layout_config = load_config(layout_path)
    else:
        world_config = run_config
        layout_config = None

    return world_config, layout_config, project_root


def generate_positions_from_config(config_file, seed=None):
    """Generate world positions in-memory without exporting files.

    Parameters
    ----------
    config_file : str
        Path to worldgen run config or legacy world config.
    seed : int | None
        Optional deterministic seed for generation.

    Returns
    -------
    tuple[list[tuple[float, float]], dict, dict | None]
        Generated positions, world config, and optional layout config.
    """
    world_config, layout_config, project_root = _load_world_and_layout_configs(config_file)

    with _local_random_seed(seed):
        if layout_config is not None:
            layout_type = layout_config['layout']['type']
            handler = LAYOUT_HANDLERS.get(layout_type)
            if handler is None:
                raise ValueError(f"Unknown layout type: {layout_type}")
            positions = handler(layout_config, world_config, project_root)
        else:
            positions = generate_single_zone(None, world_config, project_root)

    return positions, world_config, layout_config


def main():
    """
    Driver code
    
    Accepts either:
      - a worldgen_run.yaml (has 'include' key referencing world + layout)
      - a legacy world.default.yaml (single-file mode, no layout)
    """
    if len(sys.argv) < 2:
        print("Usage: python3 -m forest_worldgen.generate_world <run_config | world_config> [--seed SEED]")
        sys.exit(1)

    config_file = os.path.abspath(sys.argv[1])
    
    # Parse optional seed
    seed = None
    if len(sys.argv) > 2 and sys.argv[2] == '--seed' and len(sys.argv) > 3:
        seed = int(sys.argv[3])
        random.seed(seed)
        print(f"Using random seed: {seed}")

    # Determine project root and worldgen root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    worldgen_root = os.path.abspath(os.path.join(script_dir, '..'))

    try:
        run_config = load_config(config_file)
        if 'include' in run_config:
            world_path = resolve_path(run_config['include']['world'], project_root)
            print(f"world config  : {world_path}")
            layout_ref = run_config['include'].get('layout')
            if layout_ref:
                layout_path = resolve_path(layout_ref, project_root)
                print(f"layout config : {layout_path}")
        else:
            print("(legacy single-file config)")

        # Load templates
        world_template = load_template('world_base.sdf', worldgen_root)
        include_template = load_template('include.sdf', worldgen_root)

        positions, world_config, layout_config = generate_positions_from_config(
            config_file,
            seed=seed,
        )
        if layout_config is not None:
            print(f"layout type   : {layout_config['layout']['type']}")

        print(f"total objects : {len(positions)}")

        # --- Validation statistics (logged, not enforced) ---
        area_size = world_config['generation']['area_size']
        stats = compute_validation_stats(positions, area_size)
        R = stats['clark_evans_R']
        g_s = stats['g_small_r_mean']
        L_s = stats['L_small_r_mean']
        print(f"validation    : R={R}  g_small={g_s}  L_small={L_s}")

        # --- Prepare output directories ---
        # Always save to timestamped run directory and mirror to outputs/latest
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
        export_meta(positions, world_config, layout_config, meta_path, seed)
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
