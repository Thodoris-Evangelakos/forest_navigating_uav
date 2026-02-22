# Forest Navigating UAV

This workspace contains three main components:

- `src/faststim_forest_nav`: Fast in-memory Gymnasium environment (`ForestNavEnv`)
- `src/forest_nav_rl`: RL training/evaluation package (SAC tooling)
- `worldgen/forest_worldgen`: Forest world generation utilities and exporters

## Quickstart (Dev Setup)

From the repository root:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e src/faststim_forest_nav -e src/forest_nav_rl
```

Optional worldgen dependencies:

```bash
./.venv/bin/python -m pip install pyyaml matplotlib
```

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

For script details, see `scripts/README.md` and `worldgen/README.md`.