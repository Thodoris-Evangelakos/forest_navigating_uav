#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SEED=${1:-42}
MODEL=${2:-}
NUM_EPISODES=${3:-5}

cd "$PROJECT_ROOT"

GAZEBO_PID=""
BRIDGE_PID=""

cleanup() {
  echo ""
  echo "🛑 Cleaning up..."
  [ -n "$BRIDGE_PID" ] && kill $BRIDGE_PID 2>/dev/null || true
  [ -n "$GAZEBO_PID" ] && kill $GAZEBO_PID 2>/dev/null || true
  wait $BRIDGE_PID 2>/dev/null || true
  wait $GAZEBO_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "🚀 Starting Gazebo demo..."
echo "  Seed: $SEED"
echo "  Episodes: $NUM_EPISODES"
if [ -n "$MODEL" ]; then
  echo "  Model: $MODEL"
fi

# Source ROS2 if available
if [ -z "$ROS_DISTRO" ]; then
  if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash 2>/dev/null
  elif [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash 2>/dev/null
  fi
fi

# ── Step 1: Generate world (synchronous) ──────────────────────────
echo ""
echo "📍 Setting up Gazebo world and UAV..."
echo "[1/4] Generating world..."
bash ./scripts/worldgen/generate_world.sh configs/worldgen/worldgen_run.yaml --seed "$SEED"

LATEST_WORLD="$PROJECT_ROOT/worldgen/outputs/latest/world.sdf"
if [ ! -f "$LATEST_WORLD" ]; then
  echo "❌ World file not found: $LATEST_WORLD"
  exit 1
fi
echo "Using world: $LATEST_WORLD"

# ── Step 2: Launch Gazebo in background ───────────────────────────
echo ""
echo "[2/4] Launching Gazebo..."
gz sim -r "$LATEST_WORLD" &
GAZEBO_PID=$!
echo "Gazebo started (PID: $GAZEBO_PID)"
echo "Waiting for Gazebo to initialize (10 seconds)..."
sleep 10

if ! ps -p $GAZEBO_PID > /dev/null; then
  echo "❌ Gazebo failed to start!"
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

# ── Step 3: Spawn UAV (synchronous — must complete before demo) ───
echo ""
echo "[3/4] Spawning UAV..."
bash ./scripts/worldgen/spawn_uav.sh "$LATEST_WORLD" --index 0

# ── Step 4: Start ROS2-Gazebo bridge ──────────────────────────────
echo ""
echo "[4/4] Starting ROS2-Gazebo bridge..."
if command -v ros2 &> /dev/null; then
  # Bridge Gazebo topics to ROS2 using parameter_bridge
  # [  = Gazebo -> ROS2 direction
  # ]  = ROS2 -> Gazebo direction
  ros2 run ros_gz_bridge parameter_bridge \
    /odom@nav_msgs/msg/Odometry[gz.msgs.Odometry \
    /scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan \
    /cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
    > /tmp/bridge.log 2>&1 &
  BRIDGE_PID=$!
  echo "Bridge started (PID: $BRIDGE_PID) - logging to /tmp/bridge.log"
else
  echo "⚠️  ROS2 not found, skipping bridge"
fi

# Wait for actual data on ROS2 topics (not just topic existence)
echo "⏳ Waiting for data on ROS2 topics (/odom, /scan)..."
MAX_WAIT=30
ELAPSED=0
DATA_READY=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
  if command -v ros2 &> /dev/null; then
    # Check that at least one message has been published on each topic
    ODOM_HZ=$(timeout 2 ros2 topic hz /odom --window 1 2>/dev/null | grep -c "average rate" || true)
    SCAN_HZ=$(timeout 2 ros2 topic hz /scan --window 1 2>/dev/null | grep -c "average rate" || true)
    if [ "$ODOM_HZ" -ge 1 ] 2>/dev/null && [ "$SCAN_HZ" -ge 1 ] 2>/dev/null; then
      DATA_READY=1
      echo "✓ Topics ready — data flowing"
      break
    fi
  fi

  if [ $((ELAPSED % 5)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
    echo "  Still waiting... ($ELAPSED/$MAX_WAIT sec)"
  fi
  sleep 1
  ELAPSED=$((ELAPSED + 1))
done

if [ $DATA_READY -eq 0 ]; then
  echo "⚠️  Topic data not confirmed within ${MAX_WAIT}s — proceeding anyway..."
fi

# Run trajectory visualization
echo ""
echo "🎬 Running Gazebo demo..."
PYTHON="./.venv/bin/python"

if [ -z "$MODEL" ]; then
  LATEST=$(find outputs/runs -name "sac_final_model.zip" -type f | sort | tail -1)
  if [ -z "$LATEST" ]; then
    echo "❌ No trained model found. Run 'make rl-train' first."
    kill $GAZEBO_PID 2>/dev/null || true
    exit 1
  fi
  MODEL=$LATEST
  echo "Using latest model: $MODEL"
fi

$PYTHON -m forest_nav_rl.visualize_trajectories \
  --model "$MODEL" \
  --config configs/training/sac_gazebo.yaml \
  --num-episodes "$NUM_EPISODES" \
  --deterministic

DEMO_EXIT=$?

# Cleanup handled by trap
if [ $DEMO_EXIT -eq 0 ]; then
  echo "✓ Demo complete"
else
  echo "❌ Demo failed with exit code $DEMO_EXIT"
fi

exit $DEMO_EXIT
