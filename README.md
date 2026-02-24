# Forest Navigating UAV

This workspace contains three main components:

- `src/fastsim_forest_nav`: Fast in-memory Gymnasium environment (`ForestNavEnv`)
- `src/forest_nav_rl`: RL training/evaluation package (SAC tooling)
- `worldgen/forest_worldgen`: Forest world generation utilities and exporters

## Quickstart (Dev Setup)

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

- Generate, launch, and spawn UAV:

	```bash
	./scripts/worldgen/generate_world_with_uav.sh --seed 42 --index 0
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

Gazebo is for demonstration/rollouts only; training is supported in fastsim.

- Launch TensorBoard for runs:

	```bash
	make rl-tensorboard TB_PORT=6006
	```

- Generate single-run visualization (latest run by default):

	```bash
	make rl-visualize
	```

- Generate single-run visualization for a specific run:

	```bash
	make rl-visualize RUN=outputs/runs/sac_fastsim_005
	```

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
	make rl-trajectories MODEL=outputs/runs/sac_fastsim_005/final/sac_final_model.zip NUM_EPISODES=10
	```

For script details, see `scripts/README.md` and `worldgen/README.md`.