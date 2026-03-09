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
make clean        # remove generated caches/artifacts
```

## Reward Signal

### Per-step reward components

At every simulation step the environment (`ForestNavEnv.step()`) computes a scalar
reward by summing the following terms:

| Component | Sign | Formula / condition | Default weight |
|---|---|---|---|
| Progress | + | `reward_progress_scale × Δdist_to_goal` | 2.0 |
| Speed | + | `reward_speed_scale × (v / v_max)` | 0.02 |
| Step penalty | − | `reward_step_penalty` (applied every step) | 0.02 |
| Yaw-rate penalty | − | `reward_yaw_rate_scale × speed_norm × yaw_rate_norm` | 0.03 |
| Stall penalty | − | `reward_stall_penalty × clip((progress_stall_threshold − Δprogress) / progress_stall_threshold, 0, 1)` when `Δprogress < progress_stall_threshold`; an extra `0.5 × reward_stall_penalty` term is added when speed is also below `yaw_penalty_speed_gate` | 0.05 |
| Proximity penalty | − | `reward_proximity_scale × (soft_margin − clearance) / soft_margin` when clearance < soft margin | 0.2 |
| Accel-clip penalty | − | `reward_accel_clip_penalty` when the commanded acceleration is clipped | 0.0 |
| Shield penalty | − | `reward_shield_penalty` when the safety shield overrides the action | 0.02 |
| Collision penalty | − | `reward_collision_penalty` on collision (terminates episode) | 20.0 |
| Success bonus | + | `reward_success_bonus` on reaching the goal (terminates episode) | 5.0 |
| Truncation penalty | − | `reward_truncation_penalty` when the episode times out without success | 1.0 |

All weights are fields of `SimParams` in
`src/fastsim_forest_nav/fastsim_forest_nav/envs/params.py` and can be
overridden via the training YAML config under `env.env_kwargs.params`.

### Per-episode total reward

The **episode reward** (also called *episode return*) is the **sum of all
per-step rewards** collected within one episode:

```
episode_reward = Σ reward_t   for t = 0 … T
```

This is what the SB3 monitor and `eval_policy.py` record as `r` / `total_reward`.

### Mean reward

**Mean reward** is the **arithmetic mean of episode rewards** across a set of
episodes.  The exact set depends on the context:

| Context | What is averaged | Where it appears |
|---|---|---|
| **Evaluation (`make rl-eval`)** | All `--num-episodes` evaluation episodes | `eval/eval_summary.json` → `reward_mean`; printed as *Reward mean* |
| **Periodic eval callback (SB3)** | The N parallel eval episodes run every `eval_freq` steps | `eval/evaluations.npz` → `results`; TensorBoard tag `eval/mean_reward` |
| **Training summary / report** | Last 100 training episodes | `report/summary.csv` → `reward_last100_mean`; training-reward plot rolling mean |

#### Evaluation mean reward

```
reward_mean = mean(episode_reward_1, …, episode_reward_N)
```

Computed in `eval_policy.py`:

```python
"reward_mean": float(np.mean(rewards))   # rewards = list of per-episode totals
```

#### Training rolling mean

During training the report shows a **rolling mean** over the last 100 episodes
(and optionally a configurable window in the reward plot):

```
reward_last100_mean = arithmetic mean of episode_rewards[-100:], ignoring any NaN values
```

Computed in `visualize_training.py → _nanmean_tail()` using `np.nanmean()`.

A higher mean reward indicates the agent reaches the goal more reliably,
navigates faster, and avoids collisions and obstacles.

## Troubleshooting

- Moved repo path / stale venv:
  - Run `make venv-rebuild`
- `MODEL is required` errors:
  - Pass `MODEL=outputs/runs/.../best/best_model.zip`
- Gazebo topics not flowing (`/model/uav1/odometry`, `/scan`):
  - Ensure Gazebo world is running and UAV is spawned before `make gazebo-agent`
- Gazebo eval debugging data:
  - Each `make gazebo-agent ...` run stores telemetry under `outputs/debug/gazebo_evals/<timestamp>/`
  - UAV state stream: `uav_odom.yaml` (position, orientation, linear/angular velocity)
  - Command stream: `uav_cmd_vel.yaml`
  - Run metadata and paths: `run_meta.txt`
- CPU-only machine:
  - Set `DEVICE=cpu` for `rl-train`, `rl-eval`, and `rl-trajectories`