#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL=${1:-}
NUM_EPISODES=${2:-5}
TOPIC_WAIT_SEC=${TOPIC_WAIT_SEC:-120}
KILL_STALE_BRIDGE=${KILL_STALE_BRIDGE:-1}
GAZEBO_DEBUG_ROOT=${GAZEBO_DEBUG_ROOT:-outputs/debug/gazebo_evals}

cd "$PROJECT_ROOT"

PYTHON="./.venv/bin/python"
VENV_STAMP="./.venv/.project-root"
CURRENT_ROOT="$(pwd -P)"

if [ -x "$PYTHON" ]; then
  if [ ! -f "$VENV_STAMP" ] || [ "$(cat "$VENV_STAMP" 2>/dev/null)" != "$CURRENT_ROOT" ]; then
    echo "Detected moved/stale .venv. Rebuilding virtual environment..."
    make venv-rebuild
  fi
fi

if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

BRIDGE_PID=""
ODOM_REC_PID=""
CMD_REC_PID=""
RESOLVED_CONFIG_TMP=""
GOAL_MARKER_SDF=""
GOAL_MARKER_LOG=""
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
DEBUG_RUN_DIR="$GAZEBO_DEBUG_ROOT/$RUN_STAMP"
BRIDGE_LOG="$DEBUG_RUN_DIR/bridge.log"
AGENT_LOG="$DEBUG_RUN_DIR/agent.log"
RUN_META="$DEBUG_RUN_DIR/run_meta.txt"
ODOM_LOG="$DEBUG_RUN_DIR/uav_odom.yaml"
CMD_LOG="$DEBUG_RUN_DIR/uav_cmd_vel.yaml"
ODOM_TOPIC=${ODOM_TOPIC:-/model/uav1/odometry}
CMD_TOPIC=${CMD_TOPIC:-/model/uav1/cmd_vel}

mkdir -p "$DEBUG_RUN_DIR"

cleanup() {
  echo ""
  echo "~~~Cleaning up agent bridge...~~~"
  [ -n "$ODOM_REC_PID" ] && kill $ODOM_REC_PID 2>/dev/null || true
  [ -n "$CMD_REC_PID" ] && kill $CMD_REC_PID 2>/dev/null || true
  sleep 1
  [ -n "$ODOM_REC_PID" ] && kill -9 $ODOM_REC_PID 2>/dev/null || true
  [ -n "$CMD_REC_PID" ] && kill -9 $CMD_REC_PID 2>/dev/null || true
  wait $ODOM_REC_PID 2>/dev/null || true
  wait $CMD_REC_PID 2>/dev/null || true
  [ -n "$BRIDGE_PID" ] && kill $BRIDGE_PID 2>/dev/null || true
  sleep 1
  [ -n "$BRIDGE_PID" ] && kill -9 $BRIDGE_PID 2>/dev/null || true
  wait $BRIDGE_PID 2>/dev/null || true
  [ -n "$RESOLVED_CONFIG_TMP" ] && rm -f "$RESOLVED_CONFIG_TMP" 2>/dev/null || true
  [ -n "$GOAL_MARKER_SDF" ] && rm -f "$GOAL_MARKER_SDF" 2>/dev/null || true
  echo "✓ Agent bridge cleanup done"
}
trap cleanup EXIT

echo "...Starting Gazebo agent control..."
echo "  Episodes: $NUM_EPISODES"
if [ -n "$MODEL" ]; then
  echo "  Model: $MODEL"
fi
echo "  Debug logs: $DEBUG_RUN_DIR"
echo "  Attitude lock (roll/pitch=0) command examples:"
echo "    ROS2: ros2 topic pub --once /model/uav1/cmd_vel geometry_msgs/msg/Twist \"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}\""
echo "    Gazebo: gz topic -t /model/uav1/cmd_vel -m gz.msgs.Twist -p 'linear: {x: 0.0, y: 0.0, z: 0.0} angular: {x: 0.0, y: 0.0, z: 0.0}'"

if [ "$KILL_STALE_BRIDGE" = "1" ]; then
  echo "  Preflight: stopping stale ROS-Gazebo bridge processes..."
  pkill -f 'ros_gz_bridge|parameter_bridge' 2>/dev/null || true
  sleep 1
  pkill -9 -f 'ros_gz_bridge|parameter_bridge' 2>/dev/null || true
fi

# Source ROS2 if available (prefer Jazzy + Harmonic)
if [ -z "$ROS_DISTRO" ]; then
  if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash 2>/dev/null
  elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash 2>/dev/null
  fi
fi

if ! command -v ros2 &> /dev/null; then
  echo "ROS2 not found. Source ROS2 first."
  exit 1
fi

{
  echo "timestamp=$RUN_STAMP"
  echo "cwd=$(pwd -P)"
  echo "model_arg=$MODEL"
  echo "num_episodes=$NUM_EPISODES"
  echo "topic_wait_sec=$TOPIC_WAIT_SEC"
  echo "kill_stale_bridge=$KILL_STALE_BRIDGE"
  echo "ros_distro=${ROS_DISTRO:-unset}"
  echo "odom_topic=$ODOM_TOPIC"
  echo "cmd_topic=$CMD_TOPIC"
  echo "bridge_log=$BRIDGE_LOG"
  echo "agent_log=$AGENT_LOG"
  echo "odom_log=$ODOM_LOG"
  echo "cmd_log=$CMD_LOG"
} > "$RUN_META"

echo "Starting ROS2-Gazebo bridge..."
ros2 run ros_gz_bridge parameter_bridge \
  ${ODOM_TOPIC}@nav_msgs/msg/Odometry[gz.msgs.Odometry \
  /scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan \
  /model/uav1/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
  > "$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!
echo "Bridge started (PID: $BRIDGE_PID) - logging to $BRIDGE_LOG"

echo "...Waiting for data on ROS2 topics (${ODOM_TOPIC}, /scan)..."
MAX_WAIT=$TOPIC_WAIT_SEC
ELAPSED=0
DATA_READY=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
  sleep 2
  ELAPSED=$((ELAPSED + 2))

  timeout 2 ros2 topic echo "$ODOM_TOPIC" --once >/dev/null 2>&1 &
  ODOM_WAIT=$!
  timeout 2 ros2 topic echo /scan --once >/dev/null 2>&1 &
  SCAN_WAIT=$!

  if wait $ODOM_WAIT; then
    ODOM_OK=0
  else
    ODOM_OK=$?
  fi
  if wait $SCAN_WAIT; then
    SCAN_OK=0
  else
    SCAN_OK=$?
  fi

  if [ $ODOM_OK -eq 0 ] && [ $SCAN_OK -eq 0 ]; then
    DATA_READY=1
    echo "✓ Topics ready — data flowing"
    break
  fi

  if [ $((ELAPSED % 6)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
    echo "  Still waiting... ($ELAPSED/$MAX_WAIT sec)"
  fi
done

if [ $DATA_READY -eq 0 ]; then
  echo "Topic data not confirmed within ${MAX_WAIT}s."
  echo "Ensure Gazebo is running, world is unpaused, and UAV is spawned."
  exit 1
fi

echo "!!!Capturing UAV telemetry...!!!"
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL ros2 topic echo "$ODOM_TOPIC" > "$ODOM_LOG" 2>&1 &
else
  ros2 topic echo "$ODOM_TOPIC" > "$ODOM_LOG" 2>&1 &
fi
ODOM_REC_PID=$!

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL ros2 topic echo "$CMD_TOPIC" > "$CMD_LOG" 2>&1 &
else
  ros2 topic echo "$CMD_TOPIC" > "$CMD_LOG" 2>&1 &
fi
CMD_REC_PID=$!

sleep 1

{
  echo "odom_rec_pid=$ODOM_REC_PID"
  echo "cmd_rec_pid=$CMD_REC_PID"
} >> "$RUN_META"

echo "  Odom stream: $ODOM_TOPIC -> $ODOM_LOG"
echo "  Command stream: $CMD_TOPIC -> $CMD_LOG"

echo ""
echo "<--Running Gazebo policy control...-->"
if [ -z "$MODEL" ]; then
  LATEST=$(find outputs/runs \( -name "sac_final_model.zip" -o -name "best_model.zip" \) -type f | sort | tail -1)
  if [ -z "$LATEST" ]; then
    echo "No trained model found. Run 'make rl-train' first."
    exit 1
  fi
  MODEL=$LATEST
  echo "Using latest model: $MODEL"
fi

DEMO_CONFIG="configs/training/sac_gazebo.yaml"
LATEST_META="$PROJECT_ROOT/worldgen/outputs/latest/meta.json"
RESOLVED_CONFIG_TMP="/tmp/sac_gazebo_resolved_${$}.yaml"

echo "Using Gazebo config as-is: $DEMO_CONFIG"

RESOLVED_CONFIG=$(
  $PYTHON "$PROJECT_ROOT/scripts/rl/resolve_gazebo_goal_config.py" \
    --config "$DEMO_CONFIG" \
    --meta "$LATEST_META" \
    --out "$RESOLVED_CONFIG_TMP"
)

if [ -n "$RESOLVED_CONFIG" ] && [ -f "$RESOLVED_CONFIG" ]; then
  DEMO_CONFIG="$RESOLVED_CONFIG"
  echo "Using resolved Gazebo config: $DEMO_CONFIG"
fi

cp "$DEMO_CONFIG" "$DEBUG_RUN_DIR/gazebo_effective_config.yaml" 2>/dev/null || true

GOAL_MARKER_SDF="$DEBUG_RUN_DIR/goal_marker.sdf"
GOAL_MARKER_LOG="$DEBUG_RUN_DIR/goal_marker.log"

GOAL_MARKER_INFO=$(
  $PYTHON "$PROJECT_ROOT/scripts/rl/get_goal_marker_info.py" \
  --config "$DEMO_CONFIG" \
  --world-sdf "$PROJECT_ROOT/worldgen/outputs/latest/world.sdf"
)

if [ -n "$GOAL_MARKER_INFO" ]; then
  read -r GOAL_X GOAL_Y GOAL_Z GOAL_WORLD <<< "$GOAL_MARKER_INFO"

  cat > "$GOAL_MARKER_SDF" <<'EOF'
<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="goal_marker_model">
    <static>true</static>
    <link name="goal_marker_link">
      <collision name="goal_marker_collision">
        <geometry>
          <sphere>
            <radius>1.0</radius>
          </sphere>
        </geometry>
      </collision>
      <visual name="goal_marker_visual">
        <geometry>
          <sphere>
            <radius>1.0</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>1 0 0 0.7</ambient>
          <diffuse>1 0 0 0.7</diffuse>
          <specular>0.2 0.2 0.2 0.7</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF

  if command -v ros2 >/dev/null 2>&1; then
    ros2 run ros_gz_sim create \
      -world "$GOAL_WORLD" \
      -name "goal_marker_${RUN_STAMP}" \
      -file "$GOAL_MARKER_SDF" \
      -x "$GOAL_X" -y "$GOAL_Y" -z "$GOAL_Z" \
      > "$GOAL_MARKER_LOG" 2>&1 || true
  fi
fi

{
  echo "effective_config=$DEMO_CONFIG"
  echo "bridge_pid=$BRIDGE_PID"
  echo "goal_marker_log=$GOAL_MARKER_LOG"
} >> "$RUN_META"

set +e
$PYTHON -m forest_nav_rl.visualize_trajectories \
  --model "$MODEL" \
  --config "$DEMO_CONFIG" \
  --num-episodes "$NUM_EPISODES" \
  --deterministic 2>&1 | tee "$AGENT_LOG"
AGENT_EXIT=${PIPESTATUS[0]}
set -e

if [ $AGENT_EXIT -eq 0 ]; then
  echo "Gazebo agent run complete"
else
  echo "Gazebo agent failed with exit code $AGENT_EXIT"
fi

{
  echo "agent_exit=$AGENT_EXIT"
  echo "finished_at=$(date +%Y%m%d_%H%M%S)"
} >> "$RUN_META"

echo "Debug artifacts saved under: $DEBUG_RUN_DIR"

exit $AGENT_EXIT
