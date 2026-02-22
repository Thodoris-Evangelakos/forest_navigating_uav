.PHONY: setup verify worldgen clean help

PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip

help:
	@echo "Forest Navigating UAV - Development Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  setup       - Create venv and install all packages (editable + deps)"
	@echo "  verify      - Test imports and environment setup"
	@echo "  worldgen    - Generate a forest world with seed 42"
	@echo "  worldgen-run - Generate world and launch Gazebo"
	@echo "  clean       - Remove venv, caches, and generated outputs"
	@echo "  help        - Show this help message"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e src/fastsim_forest_nav -e src/forest_nav_rl
	$(PIP) install pyyaml matplotlib
	@echo "✓ Setup complete. Use '$(PYTHON)' or activate .venv"

verify:
	@echo "Verifying environment..."
	$(PYTHON) -c "from fastsim_forest_nav.envs.forest_nav_env import ForestNavEnv; print('✓ ForestNavEnv imported:', ForestNavEnv)"
	$(PYTHON) -c "import gymnasium; print('✓ gymnasium available')"
	$(PYTHON) -c "import stable_baselines3; print('✓ stable_baselines3 available')"
	@echo "✓ All imports verified"

worldgen:
	./scripts/worldgen/generate_world.sh --seed 42

worldgen-run:
	./scripts/worldgen/generate_world_and_run.sh --seed 42

clean:
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -rf worldgen/outputs
	@echo "✓ Cleaned"
