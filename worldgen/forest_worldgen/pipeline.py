from __future__ import annotations

import copy
import io
import os
import random
import shutil
from contextlib import contextmanager
from contextlib import redirect_stdout
from datetime import datetime

from .config import load_config, load_template, resolve_path
from .export import export_meta, export_sdf, generate_preview
from .layouts import LAYOUT_HANDLERS, generate_single_zone
from .spatial_stats import compute_validation_stats


@contextmanager
def _local_random_seed(seed: int | None):
    if seed is None:
        yield
        return

    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def _resolve_roots() -> tuple[str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    worldgen_root = os.path.abspath(os.path.join(script_dir, ".."))
    return project_root, worldgen_root


def _rewrite_distribution_refs(layout_config: dict, distribution_choices: list[str], mode: str):
    if not distribution_choices:
        return layout_config, []

    layout_copy = copy.deepcopy(layout_config)
    selected_refs: list[str] = []

    normalized_mode = str(mode or "global").strip().lower()
    per_entry_modes = {"per_entry", "per_component", "per_zone"}

    if normalized_mode in per_entry_modes:
        def choose_ref() -> str:
            ref = str(random.choice(distribution_choices))
            selected_refs.append(ref)
            return ref
    else:
        chosen = str(random.choice(distribution_choices))

        def choose_ref() -> str:
            selected_refs.append(chosen)
            return chosen

    def apply(obj):
        if isinstance(obj, dict):
            updated = {}
            for key, value in obj.items():
                if key == "distribution_ref" and isinstance(value, str):
                    updated[key] = choose_ref()
                else:
                    updated[key] = apply(value)
            return updated
        if isinstance(obj, list):
            return [apply(item) for item in obj]
        return obj

    rewritten = apply(layout_copy)
    return rewritten, selected_refs


def _load_world_and_layout_configs(config_file: str):
    config_file = os.path.abspath(config_file)
    project_root, _ = _resolve_roots()

    run_config = load_config(config_file)
    include_cfg = run_config.get("include")
    if not isinstance(include_cfg, dict):
        raise ValueError(
            "Invalid worldgen run config: missing 'include' mapping. "
            "Use configs/worldgen/worldgen_run.yaml-style config."
        )

    world_ref = include_cfg.get("world")
    if not world_ref:
        raise ValueError("Invalid worldgen run config: include.world is required")

    world_path = resolve_path(world_ref, project_root)
    world_config = load_config(world_path)

    stochastic_cfg = run_config.get("stochastic") or {}
    stochastic_enabled = bool(stochastic_cfg.get("enabled", False)) if isinstance(stochastic_cfg, dict) else False

    layout_ref = include_cfg.get("layout")
    layout_choices = []
    if isinstance(stochastic_cfg, dict):
        layout_choices = stochastic_cfg.get("layout_choices") or []
    if stochastic_enabled and isinstance(layout_choices, list) and layout_choices:
        layout_ref = random.choice(layout_choices)

    layout_config = None
    selected_distribution_refs: list[str] = []
    distribution_mode = "global"
    if layout_ref:
        layout_path = resolve_path(layout_ref, project_root)
        layout_config = load_config(layout_path)

        distribution_choices = []
        if isinstance(stochastic_cfg, dict):
            distribution_choices = stochastic_cfg.get("distribution_choices") or []
            distribution_mode = str(stochastic_cfg.get("distribution_mode", "global"))

        if stochastic_enabled and isinstance(distribution_choices, list) and distribution_choices:
            layout_config, selected_distribution_refs = _rewrite_distribution_refs(
                layout_config,
                [str(ref) for ref in distribution_choices],
                distribution_mode,
            )

    selection_meta = {
        "world_path": world_path,
        "layout_ref": str(layout_ref) if layout_ref else None,
        "stochastic_enabled": stochastic_enabled,
        "distribution_mode": distribution_mode,
        "selected_distribution_refs": selected_distribution_refs,
    }

    return world_config, layout_config, project_root, selection_meta


def _sample_start_goal_anchors(
    area_size,
    min_start_goal_distance=8.0,
    band_ratio=0.55,
    max_attempts=500,
):
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
    return_selection_meta=False,
):
    selection_meta = None

    with _local_random_seed(seed):
        world_config, layout_config, project_root, selection_meta = _load_world_and_layout_configs(config_file)
        if not bool(verbose):
            with redirect_stdout(io.StringIO()):
                if layout_config is not None:
                    layout_type = layout_config["layout"]["type"]
                    handler = LAYOUT_HANDLERS.get(layout_type)
                    if handler is None:
                        raise ValueError(f"Unknown layout type: {layout_type}")
                    positions = handler(layout_config, world_config, project_root)
                else:
                    positions = generate_single_zone(None, world_config, project_root)
        else:
            if layout_config is not None:
                layout_type = layout_config["layout"]["type"]
                handler = LAYOUT_HANDLERS.get(layout_type)
                if handler is None:
                    raise ValueError(f"Unknown layout type: {layout_type}")
                positions = handler(layout_config, world_config, project_root)
            else:
                positions = generate_single_zone(None, world_config, project_root)

    if bool(apply_start_goal_exclusion):
        gen = world_config.get("generation", {})
        exclusion_radius = float(gen.get("start_goal_tree_exclusion_radius", 0.0))
        if exclusion_radius > 0.0:
            anchors = _sample_start_goal_anchors(
                area_size=float(gen.get("area_size", 50.0)),
                min_start_goal_distance=float(gen.get("start_goal_min_distance", 8.0)),
                band_ratio=float(gen.get("start_goal_band_ratio", 0.55)),
                max_attempts=int(gen.get("spawn_max_attempts", 500)),
            )
            positions, removed = _exclude_positions_near_anchors(positions, anchors, exclusion_radius)
            world_config["_start_goal_anchors"] = {
                "start_xy": [float(anchors[0][0]), float(anchors[0][1])],
                "goal_xy": [float(anchors[1][0]), float(anchors[1][1])],
                "exclusion_radius": exclusion_radius,
                "removed_objects": int(removed),
            }

    if bool(return_selection_meta):
        return positions, world_config, layout_config, selection_meta

    return positions, world_config, layout_config


def run_generation(
    config_file: str,
    seed: int | None = None,
    apply_start_goal_exclusion: bool = True,
) -> dict[str, str]:
    _, worldgen_root = _resolve_roots()

    if seed is not None:
        print(f"Using random seed: {seed}")

    world_template = load_template("world_base.sdf", worldgen_root)
    include_template = load_template("include.sdf", worldgen_root)

    positions, world_config, layout_config, selection_meta = generate_positions_from_config(
        config_file,
        seed=seed,
        apply_start_goal_exclusion=apply_start_goal_exclusion,
        return_selection_meta=True,
    )

    print(f"world config  : {selection_meta['world_path']}")
    if selection_meta["layout_ref"]:
        print(f"layout config : {selection_meta['layout_ref']}")

    selected_distribution_refs = selection_meta["selected_distribution_refs"]
    if selection_meta["stochastic_enabled"] and selected_distribution_refs:
        if selection_meta["distribution_mode"].lower() in {"per_entry", "per_component", "per_zone"}:
            unique_refs = sorted(set(selected_distribution_refs))
            print(f"dist choices  : {', '.join(unique_refs)}")
        else:
            print(f"distribution  : {selected_distribution_refs[0]}")

    if layout_config is not None:
        print(f"layout type   : {layout_config['layout']['type']}")

    print(f"total objects : {len(positions)}")
    start_goal_meta = world_config.get("_start_goal_anchors")
    if start_goal_meta is not None:
        sx, sy = start_goal_meta["start_xy"]
        gx, gy = start_goal_meta["goal_xy"]
        rr = start_goal_meta["exclusion_radius"]
        removed = start_goal_meta["removed_objects"]
        print(
            "start/goal    : "
            f"start=({sx:.2f},{sy:.2f}) goal=({gx:.2f},{gy:.2f}) "
            f"clearance_r={rr:.2f} removed={removed}"
        )

    area_size = world_config["generation"]["area_size"]
    stats = compute_validation_stats(positions, area_size)
    print(
        "validation    : "
        f"R={stats['clark_evans_R']}  g_small={stats['g_small_r_mean']}  "
        f"L_small={stats['L_small_r_mean']}"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{timestamp}_seed{seed:04d}" if seed is not None else f"{timestamp}_random"

    run_dir = os.path.join(worldgen_root, "outputs", "runs", run_name)
    latest_dir = os.path.join(worldgen_root, "outputs", "latest")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)

    sdf_path = os.path.join(run_dir, "world.sdf")
    export_sdf(positions, world_config, world_template, include_template, sdf_path)
    print(f"generated sdf : {sdf_path}")

    latest_sdf_path = os.path.join(latest_dir, "world.sdf")
    shutil.copy2(sdf_path, latest_sdf_path)
    print(f"latest sdf    : {latest_sdf_path}")

    meta_path = os.path.join(run_dir, "meta.json")
    export_meta(
        positions,
        world_config,
        layout_config,
        meta_path,
        seed,
        start_goal_anchors=start_goal_meta,
    )
    print(f"metadata      : {meta_path}")

    latest_meta_path = os.path.join(latest_dir, "meta.json")
    shutil.copy2(meta_path, latest_meta_path)
    print(f"latest meta   : {latest_meta_path}")

    preview_path = os.path.join(run_dir, "preview.png")
    generate_preview(positions, world_config, preview_path)
    print(f"preview       : {preview_path}")

    latest_preview_path = os.path.join(latest_dir, "preview.png")
    shutil.copy2(preview_path, latest_preview_path)
    print(f"latest prev   : {latest_preview_path}")

    return {
        "run_dir": run_dir,
        "latest_dir": latest_dir,
        "world_sdf": sdf_path,
        "meta": meta_path,
        "preview": preview_path,
    }
