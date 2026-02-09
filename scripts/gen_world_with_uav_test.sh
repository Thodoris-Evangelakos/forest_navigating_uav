#!/bin/bash

# Test script: Generate world, launch Gazebo, then spawn UAV
# Usage: ./gen_world_with_uav_test.sh [config_file] [--seed SEED] [--margin MARGIN] [--height HEIGHT] [--index INDEX]
#
# This script:
# 1. Generates a randomized world (saved to outputs/runs/)
# 2. Launches Gazebo with the generated world
# 3. Spawns the UAV at an edge location (calculated on-the-fly)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default config file
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/worldgen/worldgen_run.yaml}"

# Parse arguments
SEED=""
MARGIN="1.0"
HEIGHT="2.0"
INDEX="0"

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
        --index)
            ((i++))
            INDEX="${!i}"
            ;;
    esac
    ((i++))
done

echo "====================================="
echo "  Forest World + UAV Test"
echo "====================================="
echo ""

# Step 1: Generate world (always saved to outputs/runs/)
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

# Find the most recent world.sdf in outputs/runs/
LATEST_WORLD=$(find "$PROJECT_ROOT/worldgen/outputs/runs" -name "world.sdf" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

if [ ! -f "$LATEST_WORLD" ]; then
    echo "Error: Generated world file not found in outputs/runs/"
    exit 1
fi

echo "Using world: $LATEST_WORLD"
echo ""

# Step 2: Launch Gazebo with the generated world
echo "[2/3] Launching Gazebo..."
echo "Starting Gazebo in background..."
gz sim "$LATEST_WORLD" &
GZ_PID=$!

# Wait for Gazebo to initialize
echo "Waiting for Gazebo to initialize (10 seconds)..."
sleep 10

# Check if Gazebo is still running
if ! ps -p $GZ_PID > /dev/null; then
    echo "Error: Gazebo failed to start!"
    exit 1
fi

echo ""

# Step 3: Spawn UAV (calculates spawn on-the-fly)
echo "[3/3] Spawning UAV..."
bash "$SCRIPT_DIR/spawn_uav.sh" "$LATEST_WORLD" --index "$INDEX" --margin "$MARGIN" --height "$HEIGHT"

if [ $? -eq 0 ]; then
    echo ""
    echo "====================================="
    echo "  Setup Complete!"
    echo "====================================="
    echo "World file: $LATEST_WORLD"
    echo "UAV spawned at edge position: $INDEX"
    echo ""
    echo "Gazebo is running (PID: $GZ_PID)"
    echo "To stop Gazebo, run: kill $GZ_PID"
    echo ""
    echo "Press Ctrl+C to keep Gazebo running and exit script"
    echo "or wait..."
    
    # Keep script running so user can interact with Gazebo
    wait $GZ_PID
else
    echo "Error: UAV spawn failed!"
    echo "Stopping Gazebo..."
    kill $GZ_PID
    exit 1
fi
