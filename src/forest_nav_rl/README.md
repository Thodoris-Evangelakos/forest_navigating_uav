# Forest RL Utilities

This package contains SAC training and result visualization tools for the forest UAV environment.

## Install

From repository root:

```bash
pip install -e src/fastsim_forest_nav -e src/forest_nav_rl
```

## Train SAC

```bash
forest-nav-train-sac --config configs/training/sac.yaml
```

Gazebo backend:

```bash
forest-nav-train-sac --config configs/training/sac_gazebo.yaml
```

`env.backend` in config chooses `fastsim` or `gazebo`.

Outputs are saved under `outputs/runs/<experiment_name>_<id>/`.

## Live Monitoring (TensorBoard)

```bash
tensorboard --logdir outputs/runs --port 6006
```

Open http://localhost:6006 and select a run.

## Offline Visualization (PNG + CSV)

Single run (default: latest run):

```bash
forest-nav-visualize
```

Explicit run:

```bash
forest-nav-visualize --run-dir outputs/runs/sac_fastsim_005
```

Compare all runs:

```bash
forest-nav-visualize --compare
```

## Trajectory Visualization

Visualize agent trajectories on forest maps with time-colored paths:

```bash
forest-nav-trajectories --model outputs/runs/sac_fastsim_005/final/sac_final_model.zip --num-episodes 5
```

With explicit config:

```bash
forest-nav-trajectories --model outputs/runs/sac_fastsim_005/final/sac_final_model.zip --config configs/training/sac.yaml --num-episodes 10 --deterministic
```

Gazebo backend trajectory rollouts:

```bash
forest-nav-trajectories --model outputs/runs/sac_gazebo_001/final/sac_final_model.zip --config configs/training/sac_gazebo.yaml --num-episodes 5 --deterministic
```

Gazebo backend requires ROS2 Python interfaces (`rclpy`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`) and active topics for `/odom` and `/scan`.

### Generated files

- Single run: `outputs/runs/<run>/report/`
	- `train_reward.png`
	- `safety_rates.png` (when safety columns exist)
	- `eval_reward.png` (when eval artifacts exist)
	- `summary.csv`
- Compare mode: `outputs/runs/reports/`
	- `comparison_summary.csv`
	- `comparison_overview.png`
	- per-run subdirectories with the same single-run report files- Trajectory visualization: `outputs/runs/<run>/trajectories/` (or custom output dir)
  - `trajectory_episode_000.png`, `trajectory_episode_001.png`, ...
  - `episode_stats.json`