"""
Main world generation script
Orchestrates configuration loading, position generation, and export
"""

import os
import sys
import random
from datetime import datetime

from .config import load_config, load_template, resolve_path
from .layouts import LAYOUT_HANDLERS, generate_single_zone
from .export import export_sdf, export_meta, generate_preview


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

        # --- Decide whether this is a run config or a legacy world config ---
        if 'include' in run_config:
            # worldgen_run.yaml style
            world_path = resolve_path(run_config['include']['world'], project_root)
            world_config = load_config(world_path)
            print(f"world config  : {world_path}")

            layout_ref = run_config['include'].get('layout')
            layout_config = None
            if layout_ref:
                layout_path = resolve_path(layout_ref, project_root)
                layout_config = load_config(layout_path)
                print(f"layout config : {layout_path}")
        else:
            # legacy single-file mode
            world_config = run_config
            layout_config = None
            print("(legacy single-file config)")

        # Load templates
        world_template = load_template('world_base.sdf', worldgen_root)
        include_template = load_template('include.sdf', worldgen_root)

        # --- Generate positions ---
        if layout_config is not None:
            layout_type = layout_config['layout']['type']
            handler = LAYOUT_HANDLERS.get(layout_type)
            if handler is None:
                raise ValueError(f"Unknown layout type: {layout_type}")
            print(f"layout type   : {layout_type}")
            positions = handler(layout_config, world_config, project_root)
        else:
            positions = generate_single_zone(None, world_config, project_root)

        print(f"total objects : {len(positions)}")

        # --- Prepare output directory ---
        output_dir = os.path.join(worldgen_root, 'outputs', 'latest')
        os.makedirs(output_dir, exist_ok=True)

        # Also save to timestamped run directory if seed provided
        if seed is not None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
            run_dir = os.path.join(worldgen_root, 'outputs', 'runs', f'{timestamp}_seed{seed:04d}')
            os.makedirs(run_dir, exist_ok=True)
        else:
            run_dir = None

        # --- Export SDF ---
        sdf_path = os.path.join(output_dir, 'world.sdf')
        export_sdf(positions, world_config, world_template, include_template, sdf_path)
        print(f"generated sdf : {sdf_path}")

        if run_dir:
            run_sdf_path = os.path.join(run_dir, 'world.sdf')
            export_sdf(positions, world_config, world_template, include_template, run_sdf_path)
            print(f"              : {run_sdf_path}")

        # --- Export metadata ---
        meta_path = os.path.join(output_dir, 'meta.json')
        export_meta(positions, world_config, layout_config, meta_path, seed)
        print(f"metadata      : {meta_path}")

        if run_dir:
            run_meta_path = os.path.join(run_dir, 'meta.json')
            export_meta(positions, world_config, layout_config, run_meta_path, seed)
            print(f"              : {run_meta_path}")

        # --- Generate preview ---
        preview_path = os.path.join(output_dir, 'preview.png')
        generate_preview(positions, world_config, preview_path)
        print(f"preview       : {preview_path}")

        if run_dir:
            run_preview_path = os.path.join(run_dir, 'preview.png')
            generate_preview(positions, world_config, run_preview_path)
            print(f"              : {run_preview_path}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
