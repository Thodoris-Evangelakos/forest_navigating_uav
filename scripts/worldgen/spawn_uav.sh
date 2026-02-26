#!/bin/bash

# Spawn a UAV at an edge location in an existing world
# Usage: ./spawn_uav.sh <world_sdf_path> [--index INDEX] [--margin MARGIN] [--height HEIGHT]
#
# Requires: Gazebo must be already running with the specified world loaded
#
# This script calculates spawn points on-the-fly from meta.json and spawns
# the UAV at the specified corner index.
#
# Defaults:
#   - index: 0 (NW corner)
#   - margin: 1.0 m (distance from world edge)
#   - height: 2.0 m (altitude above ground)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <world_sdf_path> [--index INDEX] [--margin MARGIN] [--height HEIGHT]"
    echo ""
    echo "Example:"
    echo "  $0 worldgen/outputs/runs/2026-02-09_123456_random/world.sdf --index 0"
    exit 1
fi

WORLD_FILE="$1"
shift

if [ ! -f "$WORLD_FILE" ]; then
    echo "Error: World file not found: $WORLD_FILE"
    exit 1
fi

# Get the directory containing world.sdf to find meta.json
WORLD_DIR="$(dirname "$WORLD_FILE")"
META_FILE="$WORLD_DIR/meta.json"

if [ ! -f "$META_FILE" ]; then
    echo "Error: Metadata file not found: $META_FILE"
    exit 1
fi

# Parse arguments
INDEX=0
MARGIN=1.0
HEIGHT=0.3

while [ $# -gt 0 ]; do
    case "$1" in
        --index)
            INDEX="$2"
            shift 2
            ;;
        --margin)
            MARGIN="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Calculate spawn point on-the-fly using Python
SPAWN_POINT=$(python3 -c "
import json
import random
import math

# Read metadata from meta.json
with open('$META_FILE', 'r') as f:
    meta = json.load(f)

start_goal = meta.get('start_goal')
if isinstance(start_goal, dict) and isinstance(start_goal.get('start_xy'), list) and len(start_goal.get('start_xy')) >= 2:
    sx, sy = start_goal['start_xy'][:2]
    print(f\"{float(sx):.2f} {float(sy):.2f} {float($HEIGHT):.2f}\")
    raise SystemExit(0)

area_size = meta['world']['area_size']

# Calculate edge spawn
margin = $MARGIN
height = $HEIGHT
half_side = area_size / 2.0
edge = half_side - margin

# Generate 4 corner spawn points
spawns = [
    {'x': -edge, 'y': edge, 'z': height},   # NW
    {'x': -edge, 'y': -edge, 'z': height},  # SW
    {'x': edge, 'y': -edge, 'z': height},   # SE
    {'x': edge, 'y': edge, 'z': height},    # NE
]

idx = $INDEX % len(spawns)
pt = spawns[idx]
print(f\"{pt['x']:.2f} {pt['y']:.2f} {pt['z']:.2f}\")
")

read X Y Z <<< "$SPAWN_POINT"

echo "Spawning UAV at edge position [$INDEX]: x=$X, y=$Y, z=$Z"

# Extract world name from world.sdf
WORLD_NAME=$(python3 -c "
import re
with open('$WORLD_FILE', 'r') as f:
    content = f.read()
    match = re.search(r'<world name=\"([^\"]+)\">', content)
    if match:
        print(match.group(1))
    else:
        print('randomized_world')
")

ros2 run ros_gz_sim create -world "$WORLD_NAME" -name uav1 -file models/drones/uav_simple/model.sdf -x "$X" -y "$Y" -z "$Z"

if [ $? -eq 0 ]; then
    echo "UAV spawned successfully!"
else
    echo "Error: UAV spawn failed!"
    exit 1
fi
