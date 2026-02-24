from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from fastsim_forest_nav.wrappers import TrajectoryRecorder
from forest_nav_rl.utils import build_env_ctor_and_kwargs
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def parse_args() -> argparse.Namespace:
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
    if config_path is None or not config_path.exists():
        cfg: dict[str, dict] = {"env": {"backend": "fastsim", "env_kwargs": {"params": {}}}}
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    env_cfg = cfg.get("env", {"backend": "fastsim", "env_kwargs": {"params": {}}})
    return build_env_ctor_and_kwargs(env_cfg)


def resolve_config_path(model_path: Path, config_path: Path | None) -> Path | None:
    if config_path is not None:
        return config_path

    model_parent = model_path.parent
    candidate_run_config = model_parent.parent / "config_used.yaml"
    if candidate_run_config.exists():
        return candidate_run_config

    return None


def resolve_vecnormalize_path(model_path: Path) -> Path | None:
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
) -> None:
    """Plot forest map with trees, start/goal, and time-colored trajectory."""
    fig, ax = plt.subplots(figsize=(10, 10))
    plasma_cmap = plt.get_cmap("plasma")

    # Plot trees as circles
    if trees is not None and len(trees) > 0:
        tree_patches = [
            Circle((tree[0], tree[1]), tree[2], facecolor="darkgreen", alpha=0.7, edgecolor="black", linewidth=0.5)
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
        cbar = plt.colorbar(sm, ax=ax, label="Time step", shrink=0.8)

    # Set equal aspect ratio and limits
    if trees is not None and len(trees) > 0:
        all_x = np.concatenate([trees[:, 0], [start_pos[0], goal_pos[0]], trajectory[:, 0]])
        all_y = np.concatenate([trees[:, 1], [start_pos[1], goal_pos[1]], trajectory[:, 1]])
        margin = 2.0
        x_min, x_max = all_x.min() - margin, all_x.max() + margin
        y_min, y_max = all_y.min() - margin, all_y.max() + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Title with outcome
    outcome = "SUCCESS" if success else ("COLLISION" if collision else "TRUNCATED")
    color = "green" if success else ("red" if collision else "orange")
    ax.set_title(f"Episode {episode_idx} - {outcome}", fontsize=14, fontweight="bold", color=color)

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_episode_with_trajectory(
    env: TrajectoryRecorder,
    model: SAC,
    deterministic: bool,
    obs_normalizer: VecNormalize | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Run one episode and return trajectory data."""
    obs, info = env.reset()
    terminated = False
    truncated = False
    success = False
    collision = False

    while not (terminated or truncated):
        policy_obs = obs
        if obs_normalizer is not None:
            normalized_obs = obs_normalizer.normalize_obs(
                np.asarray([obs], dtype=np.float32)
            )
            policy_obs = np.asarray(normalized_obs, dtype=np.float32)[0]
        action, _states = model.predict(policy_obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

    success = bool(info.get("success", False))
    collision = bool(info.get("collision", False))

    trajectory = env.get_trajectory()
    start_pos = env.episode_start_pos if env.episode_start_pos is not None else np.zeros(3)
    goal_pos = env.episode_goal_pos if env.episode_goal_pos is not None else np.zeros(3)
    trees = env.episode_trees if env.episode_trees is not None else np.empty((0, 3))

    return trajectory, trees, start_pos, goal_pos, success, collision


def main() -> None:
    matplotlib.use("Agg")
    args = parse_args()

    # Load model
    model = SAC.load(args.model)

    resolved_config = resolve_config_path(args.model, args.config)
    if resolved_config is not None:
        print(f"Using config: {resolved_config}")

    env_ctor, env_kwargs = load_env_ctor_and_kwargs(resolved_config)
    base_env = env_ctor(**env_kwargs)
    env = TrajectoryRecorder(base_env)

    vecnormalize_path = resolve_vecnormalize_path(args.model)
    obs_normalizer = load_obs_normalizer(vecnormalize_path, env_ctor, env_kwargs)
    if vecnormalize_path is not None:
        print(f"Using VecNormalize stats: {vecnormalize_path}")
    else:
        print("No VecNormalize stats found; running with raw observations.")

    # Set seed if provided
    if args.seed is not None:
        env.reset(seed=args.seed)

    # Determine output directory
    output_dir = args.output_dir if args.output_dir is not None else args.model.parent / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run episodes and generate plots
    episode_stats = []
    for episode_idx in range(args.num_episodes):
        trajectory, trees, start_pos, goal_pos, success, collision = run_episode_with_trajectory(
            env,
            model,
            deterministic=args.deterministic,
            obs_normalizer=obs_normalizer,
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
            }
        )

        output_path = output_dir / f"trajectory_episode_{episode_idx:03d}.png"
        plot_trajectory_map(trajectory, trees, start_pos, goal_pos, output_path, episode_idx, success, collision)
        print(f"Saved trajectory plot: {output_path}")

    # Save episode statistics
    stats_path = output_dir / "episode_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(episode_stats, f, indent=2)

    # Print summary
    success_count = sum(1 for stat in episode_stats if stat["success"])
    collision_count = sum(1 for stat in episode_stats if stat["collision"])
    print(f"\nSummary ({args.num_episodes} episodes):")
    print(f"  Success: {success_count} ({100*success_count/args.num_episodes:.1f}%)")
    print(f"  Collision: {collision_count} ({100*collision_count/args.num_episodes:.1f}%)")
    print(f"  Truncated: {args.num_episodes - success_count - collision_count}")
    print(f"\nAll trajectory plots saved to: {output_dir}")

    env.close()
    if obs_normalizer is not None:
        obs_normalizer.close()


if __name__ == "__main__":
    main()
