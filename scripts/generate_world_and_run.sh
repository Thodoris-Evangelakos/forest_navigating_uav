#!/bin/bash

# Generate a forest world and launch it in Gazebo
# Usage: ./generate_world_and_run.sh [config_file] [--seed SEED]
#
# This script:
# 1. Generates a randomized world (saved to outputs/runs/)
# 2. Launches Gazebo to visualize the world

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default config file
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/worldgen/worldgen_run.yaml}"

# Parse seed argument
SEED=""
if [ "$2" = "--seed" ] && [ -n "$3" ]; then
    SEED="$3"
fi

echo "====================================="
echo "  Generate World & Run"
echo "====================================="
echo ""

# Step 1: Generate world
echo "[1/2] Generating world..."
if [ -n "$SEED" ]; then
    bash "$SCRIPT_DIR/generate_world.sh" "$CONFIG_FILE" --seed "$SEED"
else
    bash "$SCRIPT_DIR/generate_world.sh" "$CONFIG_FILE"
fi

if [ $? -ne 0 ]; then
    echo "Error: World generation failed!"
    exit 1
fi

echo ""

# Find the most recent world.sdf in outputs/runs/
LATEST_WORLD=$(find "$PROJECT_ROOT/worldgen/outputs/runs" -name "world.sdf" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

if [ ! -f "$LATEST_WORLD" ]; then
    echo "Error: Generated world file not found in outputs/runs/"
    exit 1
fi

# Step 2: Launch Gazebo
echo "[2/2] Launching Gazebo..."
echo "World file: $LATEST_WORLD"
echo ""
echo "Starting Gazebo..."

gz sim "$LATEST_WORLD"

if [ $? -ne 0 ]; then
    echo "Error: Gazebo launch failed!"
    exit 1
fi
