.PHONY: setup venv-rebuild verify worldgen worldgen-run rl-train rl-resume rl-eval rl-tensorboard rl-visualize rl-compare rl-trajectories rl-gazebo-demo clean help

PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip
RL_CONFIG ?= configs/training/sac.yaml
RUN ?=
TB_PORT ?= 6006
MODEL ?=
NUM_EPISODES ?= 5
DEVICE ?= cuda

help:
	@echo "Forest Navigating UAV - Development Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  setup       - Create venv and install all packages (editable + deps)"
	@echo "  venv-rebuild - Recreate only .venv and reinstall packages"
	@echo "  verify      - Test imports and environment setup"
	@echo "  worldgen    - Generate a forest world with seed 42"
	@echo "  worldgen-run - Generate world and launch Gazebo"
	@echo "  rl-train    - Train SAC policy (override RL_CONFIG=... DEVICE=...)"
	@echo "  rl-resume   - Resume training from latest run (or RUN=outputs/runs/sac_fastsim_XXX)"
	@echo "  rl-eval     - Evaluate trained policy (requires MODEL=..., override DEVICE=...)"
	@echo "  rl-tensorboard - Launch TensorBoard on outputs/runs (override TB_PORT=...)"
	@echo "  rl-visualize - Build single-run report (latest if RUN is empty)"
	@echo "  rl-compare  - Build multi-run comparison report"
	@echo "  rl-trajectories - Visualize agent trajectories (requires MODEL=..., override DEVICE=...)"
	@echo "  rl-gazebo-demo - Launch Gazebo, run demo with latest model (override MODEL=... NUM_EPISODES=...)"
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
	./scripts/worldgen/generate_world_and_run.sh --seed 42

rl-train:
	@if [ -n "$(DEVICE)" ]; then \
		$(PYTHON) -m forest_nav_rl.train_sac --config $(RL_CONFIG) --device $(DEVICE); \
	else \
		$(PYTHON) -m forest_nav_rl.train_sac --config $(RL_CONFIG); \
	fi

rl-resume:
	@RUN_DIR="$(RUN)"; \
	if [ -z "$$RUN_DIR" ]; then \
		RUN_DIR=$$(ls -dt outputs/runs/sac_fastsim_*/ 2>/dev/null | head -1 | sed 's|/$||'); \
		if [ -z "$$RUN_DIR" ]; then \
			echo "Error: No previous runs found. Usage: make rl-resume RUN=outputs/runs/sac_fastsim_XXX"; \
			exit 1; \
		fi; \
		echo "Resuming from latest run: $$RUN_DIR"; \
	fi; \
	if [ -n "$(DEVICE)" ]; then \
		$(PYTHON) -m forest_nav_rl.train_sac --config $(RL_CONFIG) --resume-from $$RUN_DIR --device $(DEVICE); \
	else \
		$(PYTHON) -m forest_nav_rl.train_sac --config $(RL_CONFIG) --resume-from $$RUN_DIR; \
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
	@MODEL_CONFIG="$$(dirname "$(MODEL)")/../config_used.yaml"; \
	if [ -f "$$MODEL_CONFIG" ]; then \
		$(PYTHON) -m forest_nav_rl.visualize_trajectories --model "$(MODEL)" --config "$$MODEL_CONFIG" --num-episodes $(NUM_EPISODES) --device $(DEVICE); \
	elif [ -n "$(RL_CONFIG)" ] && [ -f "$(RL_CONFIG)" ]; then \
		$(PYTHON) -m forest_nav_rl.visualize_trajectories --model "$(MODEL)" --config "$(RL_CONFIG)" --num-episodes $(NUM_EPISODES) --device $(DEVICE); \
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
