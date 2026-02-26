# Forest Navigating UAV

This workspace contains three main components:

- `src/fastsim_forest_nav`: Fast in-memory Gymnasium environment (`ForestNavEnv`)
- `src/forest_nav_rl`: RL training/evaluation package (SAC tooling)
- `worldgen/forest_worldgen`: Forest world generation utilities and exporters

## Quickstart (Dev Setup)

If you copy/move this repository to a new directory, `make` targets now auto-detect stale `.venv` paths and rebuild the virtual environment as needed.

### Option 1: One-Command Setup (Recommended)

From the repository root:

```bash
make setup
```

This creates `.venv` and installs all packages plus dependencies in one shot.

### Option 2: Manual Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[all]"
```

This installs the entire workspace (root meta-package + all subpackages) with all extras.

## Verify Install

```bash
./.venv/bin/python -c "from fastsim_forest_nav.envs.forest_nav_env import ForestNavEnv; print(ForestNavEnv)"
```

Expected output includes:

```text
<class 'fastsim_forest_nav.envs.forest_nav_env.ForestNavEnv'>
```

## Why `python3 -c ...` May Fail

If you run:

```bash
python3 -c "from fastsim_forest_nav.envs.forest_nav_env import ForestNavEnv"
```

it may fail with `ModuleNotFoundError` when your system Python is not the project `.venv` and the editable package is not installed there.

Use `./.venv/bin/python` (or activate `.venv`) to ensure the correct environment.

## Common Commands

- Generate world only:

	```bash
	./scripts/worldgen/generate_world.sh --seed 42
	```

- Generate world and launch Gazebo:

	```bash
	./scripts/worldgen/generate_world_and_run.sh --seed 42
	```

- Launch world, then spawn UAV (separate steps):

	```bash
	./scripts/worldgen/generate_world_and_run.sh --seed 42
	```

	```bash
	make gazebo-spawn-uav
	```

## RL via Make

From repository root (uses `.venv` tools via Makefile):

- Train SAC (default config):

	```bash
	make rl-train
	```

- Train SAC with overrides:

	```bash
	make rl-train RL_CONFIG=configs/training/sac.yaml DEVICE=cuda
	```

	`make rl-train`, `make rl-eval`, and `make rl-trajectories` default to `DEVICE=cuda` (override with `DEVICE=cpu` if needed).

Gazebo is for demonstration/rollouts only; training is supported in fastsim.

- Launch TensorBoard for runs:

	```bash
	make rl-tensorboard TB_PORT=6006
	```

	Overrides:

	```bash
	make rl-tensorboard TB_LOGDIR=outputs/runs TB_HOST=0.0.0.0 TB_PORT=6006
	```

	`rl-resume` now keeps TensorBoard curves continuous within the same run directory.

- Single-run visualization is generated automatically at training end (including graceful Ctrl+C) into `outputs/runs/<run>/report/`.

- Generate multi-run comparison report:

	```bash
	make rl-compare
	```

- Visualize agent trajectories on forest maps:

	```bash
	make rl-trajectories MODEL=outputs/runs/sac_fastsim_005/final/sac_final_model.zip
	```

	Gazebo backend (for demo rollouts only):

	```bash
	make rl-trajectories MODEL=outputs/runs/sac_gazebo_001/final/sac_final_model.zip RL_CONFIG=configs/training/sac_gazebo.yaml
	```

	With overrides:

	```bash
	make rl-trajectories MODEL=outputs/runs/sac_fastsim_005/final/sac_final_model.zip NUM_EPISODES=10 DEVICE=cuda
	```

For script details, see `scripts/README.md` and `worldgen/README.md`.

## RL Output Folders

- `best/` stores the best eval checkpoint (`best_model.zip`) from `EvalCallback`.
- `final/` stores end-of-run artifacts (`sac_final_model.zip`, summary, and optional replay/vecnorm state).
- `monitors/` stores per-episode monitor CSV logs (train/eval) used for offline reporting.
- `eval/` stores evaluation history (`evaluations.npz`) from periodic eval callbacks.
- `tb/` stores TensorBoard event files for live scalar inspection during/after training.

## Delivery and Maintenance Targets

- Run quality gates:

	```bash
	make check
	```

- Build wheel artifacts for delivery:

	```bash
	make package
	```

- Clean generated artifacts (preserves `outputs/runs`):

	```bash
	make clean
	```

- Deep clean including `.venv` and `dist/`:

	```bash
	make deep-clean
	```