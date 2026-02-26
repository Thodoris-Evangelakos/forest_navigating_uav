.PHONY: setup venv-rebuild verify worldgen worldgen-run gazebo-world gazebo-spawn-uav gazebo-agent gazebo-stop rl-train rl-resume rl-eval rl-tensorboard rl-visualize rl-compare rl-trajectories rl-gazebo-demo clean help

PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip
RL_CONFIG ?= configs/training/sac.yaml
RESUME_CONFIG ?=
TRAJ_CONFIG ?=
RUN ?=
TB_PORT ?= 6006
MODEL ?=
NUM_EPISODES ?= 5
DEVICE ?= cuda
WORLD_CONFIG ?= configs/worldgen/worldgen_run.yaml
WORLD_SEED ?= 42
SPAWN_INDEX ?= 0
SPAWN_MARGIN ?= 1.0
SPAWN_HEIGHT ?= 2.0

help:
	@echo "Forest Navigating UAV - Development Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  setup       - Create venv and install all packages (editable + deps)"
	@echo "  venv-rebuild - Recreate only .venv and reinstall packages"
	@echo "  verify      - Test imports and environment setup"
	@echo "  worldgen    - Generate a forest world with seed 42"
	@echo "  worldgen-run - Generate world and launch Gazebo"
	@echo "  gazebo-world - Generate world and launch Gazebo (WORLD_CONFIG=... WORLD_SEED=...)"
	@echo "  gazebo-spawn-uav - Spawn UAV into running Gazebo using latest world (SPAWN_INDEX=... SPAWN_MARGIN=... SPAWN_HEIGHT=...)"
	@echo "  gazebo-agent - Start ROS-Gazebo bridge and run policy control (MODEL=... NUM_EPISODES=...)"
	@echo "  gazebo-stop - Stop all Gazebo/bridge background processes"
	@echo "  rl-train    - Train SAC policy (override RL_CONFIG=... DEVICE=...)"
	@echo "  rl-resume   - Resume from run (override RUN=... RESUME_CONFIG=... DEVICE=...)"
	@echo "  rl-eval     - Evaluate trained policy (requires MODEL=..., override DEVICE=...)"
	@echo "  rl-tensorboard - Launch TensorBoard on outputs/runs (override TB_PORT=...)"
	@echo "  rl-visualize - Build single-run report (latest if RUN is empty)"
	@echo "  rl-compare  - Build multi-run comparison report"
	@echo "  rl-trajectories - Visualize trajectories (MODEL=..., TRAJ_CONFIG=... optional, DEVICE=...)"
	@echo "  rl-gazebo-demo - Launch Gazebo (paused by default), run demo (override MODEL=... NUM_EPISODES=... GAZEBO_PAUSED_START=0)"
	@echo "  clean       - Remove venv, caches, and generated outputs"
	@echo "  help        - Show this help message"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e worldgen
	$(PIP) install -e src/fastsim_forest_nav
	$(PIP) install -e src/forest_nav_rl
	@echo "Setup complete. Use '$(PYTHON)' or activate .venv"

venv-rebuild:
	rm -rf .venv
	$(MAKE) setup

verify:
	@echo "Verifying environment..."
	$(PYTHON) -c "from fastsim_forest_nav.envs.forest_nav_env import ForestNavEnv; print('✓ ForestNavEnv imported:', ForestNavEnv)"
	$(PYTHON) -c "import gymnasium; print('✓ gymnasium available')"
	$(PYTHON) -c "import stable_baselines3; print('✓ stable_baselines3 available')"
	$(PYTHON) -c "import tensorboard; print('✓ tensorboard available')"
	$(PYTHON) -c "import matplotlib; print('✓ matplotlib available')"
	$(PYTHON) -c "import yaml; print('✓ PyYAML available')"
	@echo "✓ All imports verified"

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

rl-train:
	@if [ -n "$(DEVICE)" ]; then \
		$(PYTHON) -m forest_nav_rl.train_sac --config $(RL_CONFIG) --device $(DEVICE); \
	else \
		$(PYTHON) -m forest_nav_rl.train_sac --config $(RL_CONFIG); \
	fi

rl-resume:
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

rl-eval:
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

rl-tensorboard:
	$(PYTHON) -m tensorboard.main --logdir outputs/runs --port $(TB_PORT)

rl-visualize:
	@if [ -n "$(RUN)" ]; then \
		$(PYTHON) -m forest_nav_rl.visualize_training --run-dir $(RUN); \
	else \
		$(PYTHON) -m forest_nav_rl.visualize_training; \
	fi

rl-compare:
	$(PYTHON) -m forest_nav_rl.visualize_training --compare

rl-trajectories:
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

clean:
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -rf worldgen/outputs
	@echo "✓ Cleaned"
