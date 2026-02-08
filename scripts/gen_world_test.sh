#!/bin/bash

# script to generate and launch a test world for gazebo
# usage: ./gen_world_test.sh [config_file] [--seed SEED]
# assumes script dir is always 1 level deep, be careful about making a scripts/worldgen subdir or something

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# default config file
# just directly edit the yaml for now, might make a top level script controlling everything later
# TODO: above
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/worldgen/worldgen_run.yaml}"

echo "====================================="
echo "  Forest World Generation & Test"
echo "====================================="
echo ""

# world generation
echo "[1/2] Generating world..."
if [ "$2" = "--seed" ] && [ -n "$3" ]; then
    bash "$SCRIPT_DIR/gen_world.sh" "$CONFIG_FILE" --seed "$3"
else
    bash "$SCRIPT_DIR/gen_world.sh" "$CONFIG_FILE"
fi

if [ $? -ne 0 ]; then
    echo "Error: World generation failed!"
    exit 1
fi

echo ""

# launch in gazebo
echo "[2/2] Launching world in Gazebo..."
WORLD_FILE="$PROJECT_ROOT/worldgen/outputs/latest/world.sdf"

if [ ! -f "$WORLD_FILE" ]; then
    echo "Error: Generated world file not found: $WORLD_FILE"
    exit 1
fi

echo "Starting Gazebo with world: $WORLD_FILE"
echo "Press Ctrl+C to stop the simulation"
echo ""

# gazebo launch for my specific version
gz sim "$WORLD_FILE"

