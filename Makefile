.PHONY: setup venv-rebuild ensure-venv verify check lint test package worldgen worldgen-run gazebo-world gazebo-spawn-uav gazebo-agent gazebo-stop rl-train rl-resume rl-eval rl-tensorboard rl-compare rl-trajectories rl-gazebo-demo clean-generated clean deep-clean help

PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip
VENV_STAMP := ./.venv/.project-root
RL_CONFIG ?= configs/training/sac.yaml
RESUME_CONFIG ?=
TRAJ_CONFIG ?=
RUN ?=
TB_PORT ?= 6006
TB_HOST ?= 127.0.0.1
TB_LOGDIR ?= outputs/runs
MODEL ?=
NUM_EPISODES ?= 5
DEVICE ?= cuda
WORLD_CONFIG ?= configs/worldgen/worldgen_run.yaml
WORLD_SEED ?= 42
SPAWN_INDEX ?= 0
SPAWN_MARGIN ?= 1.0
SPAWN_HEIGHT ?= 0.3

help:
	@echo "Forest Navigating UAV - Development Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  setup       - Create/recreate venv and install all packages (editable + deps)"
	@echo "  venv-rebuild - Recreate only .venv and reinstall packages"
	@echo "  verify      - Test imports and environment setup"
	@echo "  check       - Run lint + tests + import verification"
	@echo "  lint        - Run Ruff checks on source packages"
	@echo "  test        - Run pytest when tests exist"
	@echo "  package     - Build wheel artifacts for subpackages into dist/"
	@echo "  worldgen    - Generate a forest world with seed 42"
	@echo "  worldgen-run - Generate world and launch Gazebo"
	@echo "  gazebo-world - Generate world and launch Gazebo (WORLD_CONFIG=... WORLD_SEED=...)"
	@echo "  gazebo-spawn-uav - Spawn UAV into running Gazebo using latest world (SPAWN_INDEX=... SPAWN_MARGIN=... SPAWN_HEIGHT=...)"
	@echo "  gazebo-agent - Start ROS-Gazebo bridge and run policy control (MODEL=... NUM_EPISODES=...)"
	@echo "  gazebo-stop - Stop all Gazebo/bridge background processes"
	@echo "  rl-train    - Train SAC policy (override RL_CONFIG=... DEVICE=...)"
	@echo "  rl-resume   - Resume from run (override RUN=... RESUME_CONFIG=... DEVICE=...)"
	@echo "  rl-eval     - Evaluate trained policy (requires MODEL=..., override DEVICE=...)"
	@echo "  rl-tensorboard - Launch TensorBoard (override TB_LOGDIR=... TB_HOST=... TB_PORT=...)"
	@echo "  rl-train auto-generates report/ on run end (including graceful Ctrl+C)"
	@echo "  rl-compare  - Build multi-run comparison report"
	@echo "  rl-trajectories - Visualize trajectories (MODEL=..., TRAJ_CONFIG=... optional, DEVICE=...)"
	@echo "  rl-gazebo-demo - Launch Gazebo (paused by default), run demo (override MODEL=... NUM_EPISODES=... GAZEBO_PAUSED_START=0)"
	@echo "  clean-generated - Remove caches, egg-info, and generated worldgen outputs"
	@echo "  clean       - Alias to clean-generated"
	@echo "  deep-clean  - Clean generated artifacts plus .venv and dist/"
	@echo "  help        - Show this help message"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e worldgen
	$(PIP) install -e src/fastsim_forest_nav
	$(PIP) install -e src/forest_nav_rl
	@pwd -P > $(VENV_STAMP)
	@echo "Setup complete. Use '$(PYTHON)' or activate .venv"

venv-rebuild:
	rm -rf .venv
	$(MAKE) setup

ensure-venv:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "Error: .venv not found."; \
		echo "Run 'make setup' first."; \
		exit 1; \
	fi
	@CURRENT_ROOT="$$(pwd -P)"; \
	if [ ! -f "$(VENV_STAMP)" ]; then \
		echo "Error: missing venv relocation stamp ($(VENV_STAMP))."; \
		echo "Run 'make setup' to (re)initialize the environment."; \
		exit 1; \
	elif [ "$$CURRENT_ROOT" != "$$(cat "$(VENV_STAMP)")" ]; then \
		echo "Error: project path changed since .venv was created."; \
		echo "Run 'make setup' to recreate the environment for this location."; \
		exit 1; \
	fi

verify: ensure-venv
	@echo "Verifying environment..."
	$(PYTHON) -c "from fastsim_forest_nav.envs.forest_nav_env import ForestNavEnv; print('✓ ForestNavEnv imported:', ForestNavEnv)"
	$(PYTHON) -c "import gymnasium; print('✓ gymnasium available')"
	$(PYTHON) -c "import stable_baselines3; print('✓ stable_baselines3 available')"
	$(PYTHON) -c "import tensorboard; print('✓ tensorboard available')"
	$(PYTHON) -c "import matplotlib; print('✓ matplotlib available')"
	$(PYTHON) -c "import yaml; print('✓ PyYAML available')"
	@echo "✓ All imports verified"

lint: ensure-venv
	$(PYTHON) -m ruff check src worldgen

test: ensure-venv
	@if find src worldgen -type f -name 'test_*.py' | grep -q .; then \
		$(PYTHON) -m pytest -q; \
	else \
		echo "No tests discovered; skipping pytest"; \
	fi

check: lint test verify

package: ensure-venv
	rm -rf dist
	$(PYTHON) -m pip wheel --no-deps --wheel-dir dist ./worldgen ./src/fastsim_forest_nav ./src/forest_nav_rl

worldgen:
	./scripts/worldgen/generate_world.sh configs/worldgen/worldgen_run.yaml --seed 42

worldgen-run:
	./scripts/worldgen/generate_world_and_run.sh configs/worldgen/worldgen_run.yaml --seed 42

gazebo-world:
	./scripts/worldgen/generate_world_and_run.sh $(WORLD_CONFIG) --seed $(WORLD_SEED)

gazebo-spawn-uav:
	./scripts/worldgen/spawn_uav.sh worldgen/outputs/latest/world.sdf --index $(SPAWN_INDEX) --margin $(SPAWN_MARGIN) --height $(SPAWN_HEIGHT)

gazebo-agent:
	bash ./scripts/rl/gazebo_agent_bridge.sh "$(MODEL)" $(NUM_EPISODES)

gazebo-stop:
	@echo "Stopping Gazebo and ROS-Gazebo bridge processes..."
	@pkill -f 'gz sim' 2>/dev/null || true
	@pkill -f 'ros_gz_bridge|parameter_bridge' 2>/dev/null || true
	@pkill -f 'ros_gz_sim create' 2>/dev/null || true
	@sleep 1
	@pkill -9 -f 'gz sim' 2>/dev/null || true
	@pkill -9 -f 'ros_gz_bridge|parameter_bridge' 2>/dev/null || true
	@pkill -9 -f 'ros_gz_sim create' 2>/dev/null || true
	@echo "✓ Gazebo processes stopped"

rl-train: ensure-venv
	@if [ -n "$(DEVICE)" ]; then \
		$(PYTHON) -m forest_nav_rl.train_sac --config $(RL_CONFIG) --device $(DEVICE); \
	else \
		$(PYTHON) -m forest_nav_rl.train_sac --config $(RL_CONFIG); \
	fi

rl-resume: ensure-venv
	@RUN_DIR="$(RUN)"; \
	if [ -z "$$RUN_DIR" ]; then \
		RUN_DIR=$$(ls -dt outputs/runs/sac_fastsim*/ 2>/dev/null | head -1 | sed 's|/$$||'); \
		if [ -z "$$RUN_DIR" ]; then \
			echo "Error: No previous runs found. Usage: make rl-resume RUN=outputs/runs/sac_fastsim..."; \
			exit 1; \
		fi; \
		echo "Resuming from latest run: $$RUN_DIR"; \
	fi; \
	CONFIG_PATH="$(RESUME_CONFIG)"; \
	if [ -z "$$CONFIG_PATH" ]; then \
		RUN_CONFIG="$$RUN_DIR/config_used.yaml"; \
		if [ -f "$$RUN_CONFIG" ]; then \
			CONFIG_PATH="$$RUN_CONFIG"; \
			echo "Using run config: $$CONFIG_PATH"; \
		else \
			CONFIG_PATH="$(RL_CONFIG)"; \
			echo "Using RL_CONFIG fallback: $$CONFIG_PATH"; \
		fi; \
	fi; \
	if [ ! -f "$$CONFIG_PATH" ]; then \
		echo "Error: Resume config not found: $$CONFIG_PATH"; \
		echo "Usage: make rl-resume RUN=... RESUME_CONFIG=configs/training/your_finetune.yaml"; \
		exit 1; \
	fi; \
	if [ -n "$(DEVICE)" ]; then \
		$(PYTHON) -m forest_nav_rl.train_sac --config "$$CONFIG_PATH" --resume-from $$RUN_DIR --device $(DEVICE); \
	else \
		$(PYTHON) -m forest_nav_rl.train_sac --config "$$CONFIG_PATH" --resume-from $$RUN_DIR; \
	fi

rl-eval: ensure-venv
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL is required. Usage: make rl-eval MODEL=path/to/model.zip"; \
		exit 1; \
	fi
	@MODEL_CONFIG="$$(dirname "$(MODEL)")/../config_used.yaml"; \
	if [ -f "$$MODEL_CONFIG" ]; then \
		$(PYTHON) -m forest_nav_rl.eval_policy --model "$(MODEL)" --config "$$MODEL_CONFIG" --num-episodes $(NUM_EPISODES) --deterministic --device $(DEVICE); \
	elif [ -n "$(RL_CONFIG)" ] && [ -f "$(RL_CONFIG)" ]; then \
		$(PYTHON) -m forest_nav_rl.eval_policy --model "$(MODEL)" --config "$(RL_CONFIG)" --num-episodes $(NUM_EPISODES) --deterministic --device $(DEVICE); \
	else \
		$(PYTHON) -m forest_nav_rl.eval_policy --model "$(MODEL)" --num-episodes $(NUM_EPISODES) --deterministic --device $(DEVICE); \
	fi

rl-tensorboard: ensure-venv
	@echo "TensorBoard logdir: $(TB_LOGDIR)"
	@echo "Open: http://$(TB_HOST):$(TB_PORT)"
	$(PYTHON) -m tensorboard.main --logdir $(TB_LOGDIR) --host $(TB_HOST) --port $(TB_PORT)

rl-compare: ensure-venv
	$(PYTHON) -m forest_nav_rl.visualize_training --compare

rl-trajectories: ensure-venv
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL is required. Usage: make rl-trajectories MODEL=path/to/model.zip"; \
		exit 1; \
	fi
	@CONFIG_PATH="$(TRAJ_CONFIG)"; \
	if [ -z "$$CONFIG_PATH" ]; then \
		MODEL_CONFIG="$$(dirname "$(MODEL)")/../config_used.yaml"; \
		if [ -f "$$MODEL_CONFIG" ]; then \
			CONFIG_PATH="$$MODEL_CONFIG"; \
		elif [ -n "$(RL_CONFIG)" ] && [ -f "$(RL_CONFIG)" ]; then \
			CONFIG_PATH="$(RL_CONFIG)"; \
		fi; \
	fi; \
	if [ -n "$$CONFIG_PATH" ] && [ ! -f "$$CONFIG_PATH" ]; then \
		echo "Error: Trajectory config not found: $$CONFIG_PATH"; \
		echo "Usage: make rl-trajectories MODEL=... TRAJ_CONFIG=configs/training/your_config.yaml"; \
		exit 1; \
	fi; \
	if [ -n "$$CONFIG_PATH" ]; then \
		$(PYTHON) -m forest_nav_rl.visualize_trajectories --model "$(MODEL)" --config "$$CONFIG_PATH" --num-episodes $(NUM_EPISODES) --device $(DEVICE); \
	else \
		$(PYTHON) -m forest_nav_rl.visualize_trajectories --model "$(MODEL)" --num-episodes $(NUM_EPISODES) --device $(DEVICE); \
	fi

rl-gazebo-demo:
	./scripts/rl/gazebo_demo.sh 42 "$(MODEL)" $(NUM_EPISODES)

clean-generated:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -rf worldgen/outputs
	@echo "✓ Cleaned generated artifacts"

clean: clean-generated

deep-clean: clean-generated
	rm -rf .venv dist
	@echo "✓ Deep cleaned"
