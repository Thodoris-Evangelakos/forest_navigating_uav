#!/bin/bash

# Script to generate a random forest world
# Usage: ./gen_world.sh [config_file] [--seed SEED]
#
# The config can be either:
#   - a worldgen_run.yaml  (references world + layout configs)
#   - a legacy world.default.yaml  (single-file, no layout)
# remember that worldgen_run references world.default

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/worldgen/worldgen_run.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file] [--seed SEED]"
    exit 1
fi

echo "Generating world from config: $CONFIG_FILE"

# build command, seed optional
CMD="python3 -m worldgen.forest_worldgen.generate_world \"$CONFIG_FILE\""

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
