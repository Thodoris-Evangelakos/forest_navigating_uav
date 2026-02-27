"""Generate training/evaluation plots and CSV summaries for SAC runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RunMetrics:
    """Store aggregate metrics extracted from a single training run."""

    run_name: str
    episodes: int
    reward_last100_mean: float | None
    reward_best: float | None
    success_last100_rate: float | None
    collision_last100_rate: float | None
    shield_last100_rate: float | None
    eval_latest_mean: float | None
    eval_best_mean: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for run report generation."""
    parser = argparse.ArgumentParser(description="Visualize SAC training artifacts")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory to visualize (default: latest run under --runs-root)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/runs"),
        help="Root directory containing run folders",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="If set, generate a comparison report across all runs in --runs-root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for report artifacts (default: <run>/report or <runs-root>/reports)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="Window size for rolling means in episode-based curves",
    )
    return parser.parse_args()


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _load_monitor_file(file_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline().strip()
        if not header_line.startswith("#"):
            raise ValueError(f"Invalid monitor header in {file_path}")
        metadata = json.loads(header_line[1:])
        t_start = float(metadata.get("t_start", 0.0))

        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float] = {"abs_t": t_start + float(row.get("t", 0.0))}
            for key, value in row.items():
                if key is None or key == "t":
                    continue
                if value is None or value == "":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def load_monitor_dir(monitor_dir: Path) -> dict[str, np.ndarray]:
    """Load and merge monitor CSV files from a monitor directory."""
    all_rows: list[dict[str, float]] = []
    for csv_path in sorted(monitor_dir.glob("*.monitor.csv")):
        all_rows.extend(_load_monitor_file(csv_path))

    if not all_rows:
        return {}

    all_rows.sort(key=lambda row: row.get("abs_t", 0.0))
    keys = sorted({key for row in all_rows for key in row.keys()})
    data: dict[str, np.ndarray] = {}
    for key in keys:
        data[key] = np.array([row.get(key, np.nan) for row in all_rows], dtype=np.float64)

    if "l" in data:
        lengths = np.nan_to_num(data["l"], nan=0.0)
        data["cum_steps"] = np.cumsum(lengths)

    data["episode"] = np.arange(1, len(all_rows) + 1, dtype=np.int64)
    return data


def load_eval_npz(eval_file: Path) -> dict[str, np.ndarray] | None:
    """Load evaluation arrays from an SB3 ``evaluations.npz`` file."""
    if not eval_file.exists():
        return None

    npz = np.load(eval_file)
    result: dict[str, np.ndarray] = {}
    for key in npz.files:
        result[key] = np.array(npz[key])
    return result


def _nanmean_tail(values: np.ndarray | None, tail: int = 100) -> float | None:
    if values is None or len(values) == 0:
        return None
    tail_values = values[-tail:]
    if np.all(np.isnan(tail_values)):
        return None
    return float(np.nanmean(tail_values))


def _safe_nanmax(values: np.ndarray | None) -> float | None:
    if values is None or len(values) == 0:
        return None
    if np.all(np.isnan(values)):
        return None
    return float(np.nanmax(values))


def summarize_run(
    run_dir: Path, train_data: dict[str, np.ndarray], eval_data: dict[str, np.ndarray] | None
) -> RunMetrics:
    """Compute summary metrics for one run from train/eval artifacts."""
    rewards = train_data.get("r")
    success = train_data.get("success")
    collision = train_data.get("collision")
    shield = train_data.get("shield_active")

    eval_latest_mean = None
    eval_best_mean = None
    if eval_data is not None and "results" in eval_data:
        results = eval_data["results"]
        eval_mean = (
            np.nanmean(results, axis=1) if results.ndim == 2 else np.array([], dtype=np.float64)
        )
        if len(eval_mean) > 0:
            eval_latest_mean = float(eval_mean[-1])
            eval_best_mean = float(np.nanmax(eval_mean))

    return RunMetrics(
        run_name=run_dir.name,
        episodes=int(len(train_data.get("episode", np.array([], dtype=np.float64)))),
        reward_last100_mean=_nanmean_tail(rewards),
        reward_best=_safe_nanmax(rewards),
        success_last100_rate=_nanmean_tail(success),
        collision_last100_rate=_nanmean_tail(collision),
        shield_last100_rate=_nanmean_tail(shield),
        eval_latest_mean=eval_latest_mean,
        eval_best_mean=eval_best_mean,
    )


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_summary_csv(output_file: Path, metrics: list[RunMetrics]) -> None:
    """Write run summary metrics to CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_name",
                "episodes",
                "reward_last100_mean",
                "reward_best",
                "success_last100_rate",
                "collision_last100_rate",
                "shield_last100_rate",
                "eval_latest_mean",
                "eval_best_mean",
            ]
        )
        for item in metrics:
            writer.writerow(
                [
                    item.run_name,
                    item.episodes,
                    _format_optional(item.reward_last100_mean),
                    _format_optional(item.reward_best),
                    _format_optional(item.success_last100_rate),
                    _format_optional(item.collision_last100_rate),
                    _format_optional(item.shield_last100_rate),
                    _format_optional(item.eval_latest_mean),
                    _format_optional(item.eval_best_mean),
                ]
            )


def _plot_train_rewards(
    output_dir: Path, train_data: dict[str, np.ndarray], rolling_window: int
) -> None:
    if "r" not in train_data:
        return

    episodes = train_data["episode"]
    rewards = train_data["r"]
    smooth = _rolling_mean(rewards, rolling_window)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(episodes, rewards, alpha=0.35, label="episode reward")
    ax.plot(episodes, smooth, linewidth=2.0, label=f"rolling mean ({rolling_window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training reward")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "train_reward.png", dpi=160)
    plt.close(fig)


def _plot_safety(output_dir: Path, train_data: dict[str, np.ndarray], rolling_window: int) -> None:
    episodes = train_data.get("episode")
    if episodes is None or len(episodes) == 0:
        return

    safety_keys = [
        ("success", "Success rate"),
        ("collision", "Collision rate"),
        ("shield_active", "Shield active rate"),
    ]
    available = [(key, label) for key, label in safety_keys if key in train_data]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for key, label in available:
        curve = _rolling_mean(train_data[key], rolling_window)
        ax.plot(episodes, curve, linewidth=2.0, label=f"{label} ({rolling_window})")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_title("Safety metrics")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "safety_rates.png", dpi=160)
    plt.close(fig)


def _plot_eval_rewards(output_dir: Path, eval_data: dict[str, np.ndarray] | None) -> None:
    if eval_data is None:
        return
    if "timesteps" not in eval_data or "results" not in eval_data:
        return

    timesteps = np.array(eval_data["timesteps"], dtype=np.float64)
    results = np.array(eval_data["results"], dtype=np.float64)
    if results.ndim != 2 or len(timesteps) == 0:
        return

    mean_rewards = np.nanmean(results, axis=1)
    std_rewards = np.nanstd(results, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(timesteps, mean_rewards, linewidth=2.0, label="eval mean reward")
    ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.set_title("Evaluation reward")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "eval_reward.png", dpi=160)
    plt.close(fig)


def _plot_run_comparison(output_dir: Path, metrics: list[RunMetrics]) -> None:
    if not metrics:
        return

    run_names = [item.run_name for item in metrics]
    reward = np.array(
        [
            np.nan if item.reward_last100_mean is None else item.reward_last100_mean
            for item in metrics
        ],
        dtype=np.float64,
    )
    success = np.array(
        [
            np.nan if item.success_last100_rate is None else item.success_last100_rate
            for item in metrics
        ],
        dtype=np.float64,
    )
    collision = np.array(
        [
            np.nan if item.collision_last100_rate is None else item.collision_last100_rate
            for item in metrics
        ],
        dtype=np.float64,
    )

    x = np.arange(len(run_names))
    width = 0.26

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[1.2, 1.0])

    axes[0].bar(x, reward, width=0.5)
    axes[0].set_title("Reward mean (last 100 episodes)")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, axis="y", alpha=0.2)

    axes[1].bar(x - width / 2, success, width=width, label="success")
    axes[1].bar(x + width / 2, collision, width=width, label="collision")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Safety rates (last 100 episodes)")
    axes[1].set_ylabel("Rate")
    axes[1].grid(True, axis="y", alpha=0.2)
    axes[1].legend()

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(run_names, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_overview.png", dpi=160)
    plt.close(fig)


def _find_latest_run(runs_root: Path) -> Path | None:
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda path: path.stat().st_mtime)
    return run_dirs[-1]


def _collect_run_dirs(runs_root: Path) -> list[Path]:
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    run_dirs.sort(key=lambda path: path.name)
    return run_dirs


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def generate_single_run_report(run_dir: Path, output_dir: Path, rolling_window: int) -> RunMetrics:
    """Generate plots and summary files for a single training run."""
    train_monitor_dir = run_dir / "monitors" / "train"
    eval_file = run_dir / "eval" / "evaluations.npz"

    train_data = load_monitor_dir(train_monitor_dir)
    eval_data = load_eval_npz(eval_file)
    metrics = summarize_run(run_dir, train_data, eval_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_train_rewards(output_dir, train_data, rolling_window=rolling_window)
    _plot_safety(output_dir, train_data, rolling_window=rolling_window)
    _plot_eval_rewards(output_dir, eval_data)
    write_summary_csv(output_dir / "summary.csv", [metrics])

    config = _load_config(run_dir / "config_used.yaml")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_dir": str(run_dir),
                "has_eval": eval_data is not None,
                "episodes": metrics.episodes,
                "experiment_name": config.get("experiment", {}).get("name"),
                "total_timesteps": config.get("experiment", {}).get("total_timesteps"),
            },
            handle,
            indent=2,
        )

    return metrics


def generate_comparison_report(runs_root: Path, output_dir: Path, rolling_window: int) -> None:
    """Generate multi-run comparison plots and per-run subreports."""
    metrics: list[RunMetrics] = []
    for run_dir in _collect_run_dirs(runs_root):
        train_dir = run_dir / "monitors" / "train"
        if not train_dir.exists():
            continue
        train_data = load_monitor_dir(train_dir)
        if not train_data:
            continue
        eval_data = load_eval_npz(run_dir / "eval" / "evaluations.npz")
        metrics.append(summarize_run(run_dir, train_data, eval_data))

    if not metrics:
        raise FileNotFoundError(f"No valid runs found under {runs_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(output_dir / "comparison_summary.csv", metrics)
    _plot_run_comparison(output_dir, metrics)

    for run_dir in _collect_run_dirs(runs_root):
        if not (run_dir / "monitors" / "train").exists():
            continue
        run_output = output_dir / run_dir.name
        generate_single_run_report(run_dir, run_output, rolling_window=rolling_window)


def main() -> None:
    """Run the visualization CLI for single-run or comparison reports."""
    args = parse_args()

    runs_root = args.runs_root
    if args.compare:
        output_dir = args.output_dir if args.output_dir is not None else (runs_root / "reports")
        generate_comparison_report(runs_root, output_dir, rolling_window=args.rolling_window)
        print(f"Comparison report generated at: {output_dir}")
        return

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _find_latest_run(runs_root)
        if run_dir is None:
            raise FileNotFoundError(f"No run directories found in: {runs_root}")

    output_dir = args.output_dir if args.output_dir is not None else (run_dir / "report")
    generate_single_run_report(run_dir, output_dir, rolling_window=args.rolling_window)
    print(f"Run report generated at: {output_dir}")


if __name__ == "__main__":
    main()
