#!/bin/bash

# Script to spawn a UAV at an edge location
# Usage: ./spawn_uav.sh [config_file] [--seed SEED] [--margin MARGIN] [--height HEIGHT] [--count COUNT] [--dist DIST]
#
# Defaults:
#   - config_file: ../configs/worldgen/worldgen_run.yaml
#   - margin: 1.0 m
#   - height: 2.0 m
#   - count: 4 spawns
#   - dist: uniform

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/worldgen/worldgen_run.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file] [--seed SEED] [--margin MARGIN] [--height HEIGHT] [--count COUNT] [--dist DIST]"
    exit 1
fi

echo "Calculating spawn points from config: $CONFIG_FILE"

# Build command with remaining args
CMD="python3 -m worldgen.forest_worldgen.spawn_utils \"$CONFIG_FILE\""

# Pass through all remaining arguments
for arg in "${@:2}"; do
    CMD="$CMD \"$arg\""
done

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "Spawn points calculated successfully!"
    echo "Spawn metadata saved to: $PROJECT_ROOT/worldgen/outputs/latest/spawn_points.json"
else
    echo "Spawn point calculation failed!"
    exit 1
fi
