#!/bin/bash

# Script to generate a random forest world
# Usage: ./generate_world.sh [config_file] [--seed SEED]
#
# Config should be a worldgen_run.yaml (references world + optional layout configs)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/worldgen/worldgen_run.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file] [--seed SEED]"
    exit 1
fi

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
VENV_STAMP="${PROJECT_ROOT}/.venv/.project-root"
CURRENT_ROOT="$(cd "$PROJECT_ROOT" && pwd -P)"

if [ -x "$PYTHON" ]; then
    if [ ! -f "$VENV_STAMP" ] || [ "$(cat "$VENV_STAMP" 2>/dev/null)" != "$CURRENT_ROOT" ]; then
        echo "Detected moved/stale .venv. Rebuilding virtual environment..."
        make -C "$PROJECT_ROOT" venv-rebuild
    fi
fi

if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "Generating world from config: $CONFIG_FILE"

# build command, seed optional
CMD="$PYTHON -m forest_worldgen.generate_world \"$CONFIG_FILE\""

# check if seed is provided
if [ "$2" = "--seed" ] && [ -n "$3" ]; then
    CMD="$CMD --seed $3"
fi

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "World generation complete!"
    echo "Outputs in: $PROJECT_ROOT/worldgen/outputs/latest/"
else
    echo "World generation failed!"
    exit 1
fi
