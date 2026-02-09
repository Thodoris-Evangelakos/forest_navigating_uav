#!/bin/bash

# Orchestrator script: Generate world, then spawn UAV at edge
# Usage: ./run_world_with_uav.sh [config_file] [--seed SEED] [--margin MARGIN] [--height HEIGHT]
#
# This script:
# 1. Generates a randomized world via gen_world.sh
# 2. Calculates spawn points at world edges via spawn_uav.sh
# 3. Spawns the UAV at the first calculated location using ros2
# 4. (Optional) Launches Gazebo for visualization

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default config file
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/worldgen/worldgen_run.yaml}"

# Parse additional args for world gen (--seed SEED)
SEED=""
MARGIN="1.0"
HEIGHT="2.0"

i=2
while [ $i -le $# ]; do
    case "${!i}" in
        --seed)
            ((i++))
            SEED="${!i}"
            ;;
        --margin)
            ((i++))
            MARGIN="${!i}"
            ;;
        --height)
            ((i++))
            HEIGHT="${!i}"
            ;;
    esac
    ((i++))
done

echo "====================================="
echo "  Forest World + UAV Spawn"
echo "====================================="
echo ""

# Step 1: Generate world
echo "[1/3] Generating world..."
if [ -n "$SEED" ]; then
    bash "$SCRIPT_DIR/gen_world.sh" "$CONFIG_FILE" --seed "$SEED"
else
    bash "$SCRIPT_DIR/gen_world.sh" "$CONFIG_FILE"
fi

if [ $? -ne 0 ]; then
    echo "Error: World generation failed!"
    exit 1
fi

echo ""

# Step 2: Calculate spawn points
echo "[2/3] Calculating spawn points..."
CMD="bash \"$SCRIPT_DIR/spawn_uav.sh\" \"$CONFIG_FILE\" --margin $MARGIN --height $HEIGHT"
if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi
eval $CMD

if [ $? -ne 0 ]; then
    echo "Error: Spawn point calculation failed!"
    exit 1
fi

echo ""

# Step 3: Spawn UAV via ros2
echo "[3/3] Spawning UAV in simulation..."
WORLD_FILE="$PROJECT_ROOT/worldgen/outputs/latest/world.sdf"
SPAWN_META="$PROJECT_ROOT/worldgen/outputs/latest/spawn_points.json"

if [ ! -f "$WORLD_FILE" ]; then
    echo "Error: Generated world file not found: $WORLD_FILE"
    exit 1
fi

if [ ! -f "$SPAWN_META" ]; then
    echo "Error: Spawn points file not found: $SPAWN_META"
    exit 1
fi

# Extract first spawn point from JSON (using python to avoid jq dependency)
SPAWN_POINT=$(python3 -c "
import json
with open('$SPAWN_META', 'r') as f:
    data = json.load(f)
    pt = data['spawn_points'][0]
    print(f\"{pt['x']} {pt['y']} {pt['z']}\")
")

read X Y Z <<< "$SPAWN_POINT"

echo "Spawning uav1 at position: x=$X, y=$Y, z=$Z"
ros2 run ros_gz_sim create -world randomized_world -name uav1 -file models/drones/uav_simple/model.sdf -x "$X" -y "$Y" -z "$Z"

if [ $? -eq 0 ]; then
    echo ""
    echo "====================================="
    echo "  UAV spawned successfully!"
    echo "====================================="
    echo "World file:  $WORLD_FILE"
    echo "Spawn file:  $SPAWN_META"
    echo "Gazebo launch:"
    echo "  gz sim $WORLD_FILE"
else
    echo "Error: UAV spawn failed!"
    exit 1
fi
