"""Visualize rollout trajectories on generated forest maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

from fastsim_forest_nav.wrappers import TrajectoryRecorder
from forest_nav_rl.utils import build_env_ctor_and_kwargs
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for trajectory visualization."""
    parser = argparse.ArgumentParser(description="Visualize agent trajectories on forest map")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config file for environment (if not specified, uses default params)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for trajectory plots (default: same dir as model)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for episodes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (default)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_false",
        dest="deterministic",
        help="Use stochastic actions",
    )
    parser.set_defaults(deterministic=True)
    return parser.parse_args()


def load_env_ctor_and_kwargs(config_path: Path | None):
    """Load environment constructor and kwargs from config path."""
    if config_path is None or not config_path.exists():
        cfg: dict[str, dict] = {"env": {"backend": "fastsim", "env_kwargs": {"params": {}}}}
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    env_cfg = cfg.get("env", {"backend": "fastsim", "env_kwargs": {"params": {}}})
    return build_env_ctor_and_kwargs(env_cfg)


def resolve_config_path(model_path: Path, config_path: Path | None) -> Path | None:
    """Resolve config file path from explicit input or run defaults."""
    if config_path is not None:
        return config_path

    model_parent = model_path.parent
    candidate_run_config = model_parent.parent / "config_used.yaml"
    if candidate_run_config.exists():
        return candidate_run_config

    return None


def resolve_model_path(model_path: Path) -> Path:
    """Resolve the best existing model path from common naming variants."""
    candidates: list[Path] = [model_path]

    path_str = str(model_path)
    if path_str.endswith(".zip.zip"):
        candidates.append(Path(path_str[:-4]))

    if model_path.suffix != ".zip":
        candidates.append(model_path.with_suffix(".zip"))

    if model_path.name == "best_model.zip" and model_path.parent.name == "best":
        run_dir = model_path.parent.parent
        candidates.append(run_dir / "final" / "sac_final_model.zip")

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    attempted = "\n".join(f"  - {candidate}" for candidate in unique_candidates)
    raise FileNotFoundError(
        "Could not find model file. Attempted paths:\n"
        f"{attempted}\n"
        "Tip: if /best/best_model.zip is missing, try /final/sac_final_model.zip"
    )


def resolve_vecnormalize_path(model_path: Path) -> Path | None:
    """Resolve candidate VecNormalize statistics file for a given model."""
    model_parent = model_path.parent

    candidates = [
        model_parent / "vecnormalize.pkl",
        model_parent.parent / "vecnormalize.pkl",
        model_parent.parent / "final" / "vecnormalize.pkl",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_obs_normalizer(
    vecnormalize_path: Path | None,
    env_ctor,
    env_kwargs: dict,
) -> VecNormalize | None:
    """Load VecNormalize statistics for observation normalization."""
    if vecnormalize_path is None:
        return None

    dummy_env = DummyVecEnv([lambda: env_ctor(**env_kwargs)])
    vecnorm = VecNormalize.load(str(vecnormalize_path), dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


def plot_trajectory_map(
    trajectory: np.ndarray,
    trees: np.ndarray,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    output_path: Path,
    episode_idx: int,
    success: bool,
    collision: bool,
    boundary_half_extent: float | None = None,
) -> None:
    """Plot forest map with trees, start/goal, and a compact height-over-time panel."""
    fig, (ax, ax_z) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [6, 1], "hspace": 0.2},
    )
    plasma_cmap = plt.get_cmap("plasma")

    # Plot trees as circles
    if trees is not None and len(trees) > 0:
        tree_patches = [
            Circle(
                (tree[0], tree[1]),
                tree[2],
                facecolor="darkgreen",
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            for tree in trees
        ]
        tree_collection = PatchCollection(tree_patches, match_original=True)
        ax.add_collection(tree_collection)

    # Plot start position
    ax.scatter(
        start_pos[0],
        start_pos[1],
        s=200,
        c="blue",
        marker="o",
        edgecolors="black",
        linewidths=2,
        label="Start",
        zorder=10,
    )

    # Plot goal position
    ax.scatter(
        goal_pos[0],
        goal_pos[1],
        s=200,
        c="gold",
        marker="*",
        edgecolors="black",
        linewidths=2,
        label="Goal",
        zorder=10,
    )

    # Plot virtual boundary used for collision checks (if available)
    if boundary_half_extent is not None and boundary_half_extent > 0.0:
        side = 2.0 * boundary_half_extent
        boundary = Rectangle(
            (-boundary_half_extent, -boundary_half_extent),
            side,
            side,
            fill=False,
            edgecolor="crimson",
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
            zorder=6,
            label="Virtual boundary",
        )
        ax.add_patch(boundary)

    # Plot trajectory with time-based color gradient
    if len(trajectory) > 1:
        points = trajectory[:, :2]  # xy positions
        time_steps = np.arange(len(points))

        # Create line segments for color mapping
        for i in range(len(points) - 1):
            ax.plot(
                points[i : i + 2, 0],
                points[i : i + 2, 1],
                color=plasma_cmap(i / max(len(points) - 1, 1)),
                linewidth=2.0,
                alpha=0.8,
            )

        # Add colorbar to show time progression
        sm = cm.ScalarMappable(cmap=plasma_cmap, norm=Normalize(vmin=0, vmax=len(points) - 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Time step", shrink=0.8)

    # Set equal aspect ratio and centered limits around origin
    x_values = [float(start_pos[0]), float(goal_pos[0])]
    y_values = [float(start_pos[1]), float(goal_pos[1])]

    if trajectory is not None and len(trajectory) > 0:
        x_values.extend(np.asarray(trajectory[:, 0], dtype=np.float64).tolist())
        y_values.extend(np.asarray(trajectory[:, 1], dtype=np.float64).tolist())

    if trees is not None and len(trees) > 0:
        x_values.extend(np.asarray(trees[:, 0], dtype=np.float64).tolist())
        y_values.extend(np.asarray(trees[:, 1], dtype=np.float64).tolist())

    extent_from_data = max(
        max(abs(v) for v in x_values) if x_values else 0.0,
        max(abs(v) for v in y_values) if y_values else 0.0,
    )
    extent_from_boundary = float(boundary_half_extent) if boundary_half_extent is not None else 0.0
    half_extent = max(extent_from_data, extent_from_boundary) + 2.0
    half_extent = max(half_extent, 1.0)

    ax.set_xlim(-half_extent, half_extent)
    ax.set_ylim(-half_extent, half_extent)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Title with outcome
    outcome = "SUCCESS" if success else ("COLLISION" if collision else "TRUNCATED")
    color = "green" if success else ("red" if collision else "orange")
    ax.set_title(f"Episode {episode_idx} - {outcome}", fontsize=14, fontweight="bold", color=color)

    ax.legend(loc="upper right")

    # Bottom panel: compact height-over-time graph
    if trajectory is not None and len(trajectory) > 0:
        z_values = trajectory[:, 2]
        time_steps = np.arange(len(z_values))
        ax_z.plot(time_steps, z_values, color="tab:purple", linewidth=1.8)
        ax_z.scatter(time_steps[-1], z_values[-1], color="tab:purple", s=20, zorder=3)
    else:
        ax_z.plot([], [])

    ax_z.grid(True, alpha=0.3)
    ax_z.set_ylabel("Z (m)")
    ax_z.set_xlabel("Time step")
    ax_z.set_title("Height over time", fontsize=10)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compute_collision_diagnostics(
    env: TrajectoryRecorder,
    trajectory: np.ndarray,
    trees: np.ndarray,
) -> dict[str, Any]:
    """Estimate final-step collision source and clearances for a trajectory."""
    if trajectory.size == 0:
        return {
            "final_x": None,
            "final_y": None,
            "final_z": None,
            "final_tree_clearance": None,
            "final_boundary_clearance": None,
            "collision_source_guess": "unknown",
            "effective_world_half_extent": None,
        }

    final_pos = trajectory[-1]
    final_x = float(final_pos[0])
    final_y = float(final_pos[1])
    final_z = float(final_pos[2])

    unwrapped = env.unwrapped

    tree_clearance = None
    if trees is not None and len(trees) > 0:
        dxy = trees[:, :2] - np.array([final_x, final_y], dtype=np.float64)
        d = np.linalg.norm(dxy, axis=1)
        clearance = d - trees[:, 2]
        tree_clearance = float(np.min(clearance))

    boundary_half = None
    boundary_clearance = None
    boundary_half_getter = getattr(unwrapped, "_effective_world_half_extent", None)
    if callable(boundary_half_getter):
        boundary_val = boundary_half_getter()
        if isinstance(boundary_val, (int, float)):
            boundary_half = float(boundary_val)
            dx = boundary_half - abs(final_x)
            dy = boundary_half - abs(final_y)
            boundary_clearance = float(min(dx, dy))

    params = getattr(unwrapped, "p", object())
    configured_radius = float(getattr(params, "drone_radius", 0.0))
    fallback_threshold = float(getattr(params, "collision_threshold", 0.18))
    collision_radius = configured_radius if configured_radius > 0.0 else fallback_threshold

    candidates: list[tuple[str, float]] = []
    if tree_clearance is not None:
        candidates.append(("tree", tree_clearance))
    if boundary_clearance is not None:
        candidates.append(("boundary", boundary_clearance))

    source = "unknown"
    if candidates:
        source = min(candidates, key=lambda item: item[1])[0]
        # if neither is actually below threshold, keep informative label
        if min(candidates, key=lambda item: item[1])[1] >= collision_radius:
            source = "none_below_threshold"

    return {
        "final_x": final_x,
        "final_y": final_y,
        "final_z": final_z,
        "final_tree_clearance": tree_clearance,
        "final_boundary_clearance": boundary_clearance,
        "collision_radius": collision_radius,
        "collision_source_guess": source,
        "effective_world_half_extent": boundary_half,
    }


def run_episode_with_trajectory(
    env: TrajectoryRecorder,
    model: SAC,
    deterministic: bool,
    obs_normalizer: VecNormalize | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
    """Run one episode and return trajectory data."""
    obs, info = env.reset()
    terminated = False
    truncated = False
    success = False
    collision = False

    while not (terminated or truncated):
        policy_obs = obs
        if obs_normalizer is not None:
            normalized_obs = obs_normalizer.normalize_obs(np.asarray([obs], dtype=np.float32))
            policy_obs = np.asarray(normalized_obs, dtype=np.float32)[0]
        action, _states = model.predict(policy_obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

    success = bool(info.get("success", False))
    collision = bool(info.get("collision", False))

    trajectory = env.get_trajectory()
    start_pos = env.episode_start_pos if env.episode_start_pos is not None else np.zeros(3)
    goal_pos = env.episode_goal_pos if env.episode_goal_pos is not None else np.zeros(3)
    trees = env.episode_trees if env.episode_trees is not None else np.empty((0, 3))

    diagnostics = compute_collision_diagnostics(env, trajectory, trees)

    return trajectory, trees, start_pos, goal_pos, success, collision, diagnostics


def main() -> None:
    """Generate trajectory plots and diagnostics for one or more episodes."""
    matplotlib.use("Agg")
    args = parse_args()

    resolved_model_path = resolve_model_path(args.model)
    if resolved_model_path != args.model:
        print(f"Resolved model path: {resolved_model_path}")

    # Load model
    model = SAC.load(str(resolved_model_path), device=args.device)

    resolved_config = resolve_config_path(resolved_model_path, args.config)
    if resolved_config is not None:
        print(f"Using config: {resolved_config}")

    env_ctor, env_kwargs = load_env_ctor_and_kwargs(resolved_config)
    base_env = env_ctor(**env_kwargs)
    env = TrajectoryRecorder(base_env)

    vecnormalize_path = resolve_vecnormalize_path(resolved_model_path)
    obs_normalizer = load_obs_normalizer(vecnormalize_path, env_ctor, env_kwargs)
    if vecnormalize_path is not None:
        print(f"Using VecNormalize stats: {vecnormalize_path}")
    else:
        print("No VecNormalize stats found; running with raw observations.")

    # Set seed if provided
    if args.seed is not None:
        env.reset(seed=args.seed)

    # Determine output directory
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else resolved_model_path.parent / "trajectories"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run episodes and generate plots
    episode_stats = []
    for episode_idx in range(args.num_episodes):
        trajectory, trees, start_pos, goal_pos, success, collision, diagnostics = (
            run_episode_with_trajectory(
                env,
                model,
                deterministic=args.deterministic,
                obs_normalizer=obs_normalizer,
            )
        )

        episode_stats.append(
            {
                "episode": episode_idx,
                "success": bool(success),
                "collision": bool(collision),
                "trajectory_length": len(trajectory),
                "start_x": float(start_pos[0]),
                "start_y": float(start_pos[1]),
                "goal_x": float(goal_pos[0]),
                "goal_y": float(goal_pos[1]),
                "num_trees": len(trees),
                "worldgen_seed": env.episode_reset_info.get("worldgen_seed"),
                "worldgen_layout": env.episode_reset_info.get("worldgen_layout"),
                "worldgen_distribution_refs": list(
                    env.episode_reset_info.get("worldgen_distribution_refs", [])
                ),
                **diagnostics,
            }
        )

        output_path = output_dir / f"trajectory_episode_{episode_idx:03d}.png"
        boundary_half_extent = diagnostics.get("effective_world_half_extent")
        if not isinstance(boundary_half_extent, (int, float)):
            boundary_half_extent = None

        plot_trajectory_map(
            trajectory,
            trees,
            start_pos,
            goal_pos,
            output_path,
            episode_idx,
            success,
            collision,
            boundary_half_extent=float(boundary_half_extent)
            if boundary_half_extent is not None
            else None,
        )
        print(f"Saved trajectory plot: {output_path}")

    # Save episode statistics
    stats_path = output_dir / "episode_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(episode_stats, f, indent=2)

    # Print summary
    success_count = sum(1 for stat in episode_stats if stat["success"])
    collision_count = sum(1 for stat in episode_stats if stat["collision"])
    print(f"\nSummary ({args.num_episodes} episodes):")
    print(f"  Success: {success_count} ({100 * success_count / args.num_episodes:.1f}%)")
    print(f"  Collision: {collision_count} ({100 * collision_count / args.num_episodes:.1f}%)")
    print(f"  Truncated: {args.num_episodes - success_count - collision_count}")
    print(f"\nAll trajectory plots saved to: {output_dir}")

    env.close()
    if obs_normalizer is not None:
        obs_normalizer.close()


if __name__ == "__main__":
    main()
