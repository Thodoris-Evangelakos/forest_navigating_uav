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

`env.backend` in config chooses `fastsim` or `gazebo`. Gazebo is for demonstration/rollouts only.

Outputs are saved under `outputs/runs/<experiment_name>_<id>/`.

## Evaluate Policy

```bash
forest-nav-eval-policy --model outputs/runs/sac_fastsim_005/best/best_model.zip --num-episodes 20 --deterministic
```

The evaluator auto-loads `config_used.yaml` and `vecnormalize.pkl` from the run when available.

## Live Monitoring (TensorBoard)

```bash
make rl-tensorboard TB_PORT=6006
```

Open http://localhost:6006 and select a run.

For remote/WSL setups:

```bash
make rl-tensorboard TB_LOGDIR=outputs/runs TB_HOST=0.0.0.0 TB_PORT=6006
```

When resuming a run, TensorBoard timesteps continue from prior training instead of restarting at zero.

## Offline Visualization (PNG + CSV)

Single-run report is generated automatically at training end into `outputs/runs/<run>/report/`.

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

## Output folder semantics

- `best/`: best checkpoint from periodic eval (`best_model.zip`)
- `final/`: end-of-run state (`sac_final_model.zip` + summary + optional replay/vecnormalize)
- `monitors/`: episode monitor CSVs used for offline metrics/plots
- `eval/`: evaluation traces (`evaluations.npz`) from eval callback
- `tb/`: TensorBoard event files for live monitoring