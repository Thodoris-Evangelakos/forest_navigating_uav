# Forest Navigating UAV

Safe UAV navigation in randomized forest environments using reinforcement learning.

This repository combines:
- A fast in-memory Gymnasium simulator for scalable SAC training
- A Gazebo + ROS2 backend for rollout/demo transfer
- A configurable forest world generator with reproducible seeds

## Highlights

- SAC training pipeline with Stable-Baselines3
- Built-in safety shield (command clamping near obstacles/boundaries)
- Offline metrics + plotting (single-run and multi-run reports)
- Trajectory visualization with per-episode diagnostics
- Reproducible world generation (`world.sdf`, `meta.json`, `preview.png`)

## Repository Structure

```text
.
├── configs/                    # training/sim/eval/worldgen YAML configs
├── scripts/
│   ├── rl/                     # train, Gazebo bridge, plotting wrappers
│   └── worldgen/               # world generation + UAV spawn scripts
├── src/
│   ├── fastsim_forest_nav/     # simulator envs + safety logic
│   └── forest_nav_rl/          # train/eval/visualization modules
├── worldgen/                   # forest_worldgen package
├── models/drones/uav_simple/   # Gazebo UAV model
└── outputs/runs/               # checkpoints, logs, eval, reports
```

## Requirements

- Linux
- Python 3.10+
- Optional GPU for faster training (`DEVICE=cuda`)
- Gazebo (`gz sim`) + ROS2 (for Gazebo rollouts)

Python dependencies are in `requirements.txt`.

## Setup

### Recommended

```bash
make setup
```

### Manual

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -e worldgen -e src/fastsim_forest_nav -e src/forest_nav_rl
```

### Verify

```bash
make verify
```

## Quickstart

Generate a world:

```bash
make worldgen
```

Generate + run world in Gazebo:

```bash
make gazebo-world WORLD_CONFIG=configs/worldgen/worldgen_run.yaml WORLD_SEED=42
```

Spawn UAV in the running world:

```bash
make gazebo-spawn-uav
```

## Training (FastSim)

Train with default config:

```bash
make rl-train
```

Train with explicit config/device:

```bash
make rl-train RL_CONFIG=configs/training/sac.yaml DEVICE=cuda
```

Resume a run:

```bash
make rl-resume RUN=outputs/runs/<run_name> RESUME_CONFIG=configs/training/sac_hybrid_stage1.yaml DEVICE=cuda
```

Important:
- Training is supported in `fastsim`
- Gazebo backend is for demo/rollout evaluation

## Evaluation

Evaluate a trained model:

```bash
make rl-eval MODEL=outputs/runs/<run_name>/best/best_model.zip NUM_EPISODES=20 DEVICE=cpu
```

Metrics are written to:
- `<model_dir>/eval/eval_summary.json`

## Visualization

Launch TensorBoard:

```bash
make rl-tensorboard TB_PORT=6006
```

Generate multi-run comparison report:

```bash
make rl-compare
```

Generate trajectory plots:

```bash
make rl-trajectories MODEL=outputs/runs/<run_name>/best/best_model.zip NUM_EPISODES=5 DEVICE=cpu
```

## Output Artifacts

Each run under `outputs/runs/<run_name>/` typically contains:

- `best/` → best checkpoint (`best_model.zip`)
- `final/` → final model + summary (`sac_final_model.zip`, `training_summary.yaml`)
- `eval/` → periodic eval data (`evaluations.npz`) and eval JSON summaries
- `monitors/` → per-episode monitor CSVs
- `tb/` → TensorBoard event files
- `report/` → generated plots and CSV summaries

## Useful Make Targets

```bash
make check        # lint + tests (if discovered) + verify
make package      # build wheels in dist/
make clean        # remove generated caches/artifacts
make deep-clean   # clean + remove .venv and dist/
```

## Troubleshooting

- Moved repo path / stale venv:
  - Run `make venv-rebuild`
- `MODEL is required` errors:
  - Pass `MODEL=outputs/runs/.../best/best_model.zip`
- Gazebo topics not flowing (`/odom`, `/scan`):
  - Ensure Gazebo world is running and UAV is spawned before `make gazebo-agent`
- CPU-only machine:
  - Set `DEVICE=cpu` for `rl-train`, `rl-eval`, and `rl-trajectories`