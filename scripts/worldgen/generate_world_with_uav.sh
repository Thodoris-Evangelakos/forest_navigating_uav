#!/bin/bash

# Generate a forest world, launch Gazebo, and spawn a UAV at the edge
# Usage: ./generate_world_with_uav.sh [config_file] [--seed SEED] [--index INDEX] [--margin MARGIN] [--height HEIGHT] [--wait-spawn|--no-wait-spawn]
#
# This script:
# 1. Generates a randomized world (saved to outputs/runs/)
# 2. Launches Gazebo with the generated world
# 3. Spawns a UAV at an edge location
#
# Defaults:
#   - config_file: configs/worldgen/worldgen_run.yaml
#   - index: 0 (NW corner)
#   - margin: 1.0 m
#   - height: 2.0 m
#   - wait-spawn: enabled (press Enter before UAV spawn)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Default config file
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/worldgen/worldgen_run.yaml}"

# Parse arguments
SEED=""
INDEX="0"
MARGIN="1.0"
HEIGHT="0.3"
WAIT_SPAWN="1"

i=2
while [ $i -le $# ]; do
    case "${!i}" in
        --seed)
            ((i++))
            SEED="${!i}"
            ;;
        --index)
            ((i++))
            INDEX="${!i}"
            ;;
        --margin)
            ((i++))
            MARGIN="${!i}"
            ;;
        --height)
            ((i++))
            HEIGHT="${!i}"
            ;;
        --wait-spawn)
            WAIT_SPAWN="1"
            ;;
        --no-wait-spawn)
            WAIT_SPAWN="0"
            ;;
    esac
    ((i++))
done

echo "====================================="
echo "  Generate World + UAV"
echo "====================================="
echo ""

# Step 1: Generate world
echo "[1/3] Generating world..."
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

# Use latest world output
LATEST_WORLD="$PROJECT_ROOT/worldgen/outputs/latest/world.sdf"

if [ ! -f "$LATEST_WORLD" ]; then
    echo "Error: Generated world file not found: $LATEST_WORLD"
    exit 1
fi

echo "Using world: $LATEST_WORLD"
echo ""

# Step 2: Launch Gazebo
echo "[2/3] Launching Gazebo..."
echo "Starting Gazebo in background..."
gz sim -r "$LATEST_WORLD" &
GZ_PID=$!

# Wait for Gazebo to initialize
echo "Waiting for Gazebo to initialize (30 seconds)..."
sleep 30

# Check if Gazebo is still running
if ! ps -p $GZ_PID > /dev/null; then
    echo "Error: Gazebo failed to start!"
    exit 1
fi

WORLD_NAME=$(python3 -c "
import re
with open('$LATEST_WORLD', 'r') as f:
    content = f.read()
match = re.search(r'<world name=\"([^\"]+)\">', content)
print(match.group(1) if match else 'randomized_world')
")

if command -v gz &> /dev/null; then
    echo "Ensuring Gazebo is unpaused..."
    gz service -s "/world/${WORLD_NAME}/control" \
        --reqtype gz.msgs.WorldControl \
        --reptype gz.msgs.Boolean \
        --timeout 3000 \
        --req 'pause: false' >/tmp/gz_unpause.log 2>&1 || true
fi

echo ""

# Step 3: Spawn UAV
echo "[3/3] Spawning UAV..."
if [ "$WAIT_SPAWN" = "1" ]; then
    if [ -t 0 ]; then
        echo "Gazebo is ready."
        echo "Press Enter when you want to spawn the UAV (or type 'q' to cancel)."
        read -r USER_CONFIRM
        if [ "$USER_CONFIRM" = "q" ]; then
            echo "Spawn cancelled by user."
            echo "Gazebo is still running (PID: $GZ_PID)."
            exit 0
        fi
    else
        echo "Non-interactive shell detected; proceeding with UAV spawn."
    fi
fi

bash "$SCRIPT_DIR/spawn_uav.sh" "$LATEST_WORLD" --index "$INDEX" --margin "$MARGIN" --height "$HEIGHT"

if [ $? -eq 0 ]; then
    echo ""
    echo "====================================="
    echo "  Setup Complete!"
    echo "====================================="
    echo "World file: $LATEST_WORLD"
    echo "UAV spawned at corner: $INDEX (0=NW, 1=SW, 2=SE, 3=NE)"
    echo ""
    echo "Gazebo is running (PID: $GZ_PID)"
    echo "To stop Gazebo: kill $GZ_PID"
    echo ""
    echo "Press Ctrl+C to exit (Gazebo will keep running)"
    
    # Keep script running so user can see the message
    wait $GZ_PID
else
    echo "Error: UAV spawn failed!"
    echo "Stopping Gazebo..."
    kill $GZ_PID 2>/dev/null
    exit 1
fi
