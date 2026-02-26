#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SEED=${1:-42}
MODEL=${2:-}
NUM_EPISODES=${3:-5}
KILL_STALE_GAZEBO=${KILL_STALE_GAZEBO:-1}
GAZEBO_PAUSED_START=${GAZEBO_PAUSED_START:-1}
GAZEBO_INIT_WAIT_SEC=${GAZEBO_INIT_WAIT_SEC:-10}
TOPIC_WAIT_SEC=${TOPIC_WAIT_SEC:-120}
SPAWN_WAIT_SEC=${SPAWN_WAIT_SEC:-60}
SPAWN_RETRIES=${SPAWN_RETRIES:-20}

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

GAZEBO_PID=""
BRIDGE_PID=""
DEMO_CONFIG_TMP=""

cleanup() {
  echo ""
  echo "🛑 Cleaning up..."
  [ -n "$BRIDGE_PID" ] && kill $BRIDGE_PID 2>/dev/null || true
  [ -n "$GAZEBO_PID" ] && kill $GAZEBO_PID 2>/dev/null || true
  sleep 1
  [ -n "$BRIDGE_PID" ] && kill -9 $BRIDGE_PID 2>/dev/null || true
  [ -n "$GAZEBO_PID" ] && kill -9 $GAZEBO_PID 2>/dev/null || true
  wait $BRIDGE_PID 2>/dev/null || true
  wait $GAZEBO_PID 2>/dev/null || true
  [ -n "$DEMO_CONFIG_TMP" ] && rm -f "$DEMO_CONFIG_TMP" 2>/dev/null || true
  echo "✓ Cleanup done"
}
trap cleanup EXIT

echo "🚀 Starting Gazebo demo..."
echo "  Seed: $SEED"
echo "  Episodes: $NUM_EPISODES"
if [ -n "$MODEL" ]; then
  echo "  Model: $MODEL"
fi
if [ "$GAZEBO_PAUSED_START" = "1" ]; then
  echo "  Gazebo startup: paused (manual Play in GUI)"
else
  echo "  Gazebo startup: auto-run"
fi

if [ "$KILL_STALE_GAZEBO" = "1" ]; then
  echo "  Preflight: stopping stale Gazebo/bridge processes..."
  pkill -f 'gz sim' 2>/dev/null || true
  pkill -f 'ros_gz_bridge|parameter_bridge' 2>/dev/null || true
  pkill -f 'ros_gz_sim create' 2>/dev/null || true
  sleep 1
  pkill -9 -f 'gz sim' 2>/dev/null || true
  pkill -9 -f 'ros_gz_bridge|parameter_bridge' 2>/dev/null || true
  pkill -9 -f 'ros_gz_sim create' 2>/dev/null || true
fi

# Source ROS2 if available (prefer Jazzy + Harmonic)
if [ -z "$ROS_DISTRO" ]; then
  if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash 2>/dev/null
  elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash 2>/dev/null
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
if [ "$GAZEBO_PAUSED_START" = "1" ]; then
  gz sim "$LATEST_WORLD" &
else
  gz sim -r "$LATEST_WORLD" &
fi
GAZEBO_PID=$!
echo "Gazebo started (PID: $GAZEBO_PID)"
echo "Waiting for Gazebo to initialize (${GAZEBO_INIT_WAIT_SEC} seconds)..."
sleep "$GAZEBO_INIT_WAIT_SEC"

if ! ps -p $GAZEBO_PID > /dev/null; then
  echo "❌ Gazebo failed to start!"
  exit 1
fi

WORLD_NAME=$($PYTHON -c "
import re
with open('$LATEST_WORLD', 'r') as f:
    content = f.read()
match = re.search(r'<world name=\"([^\"]+)\">', content)
print(match.group(1) if match else 'randomized_world')
")

unpause_world() {
  if command -v gz &> /dev/null; then
    gz service -s "/world/${WORLD_NAME}/control" \
      --reqtype gz.msgs.WorldControl \
      --reptype gz.msgs.Boolean \
      --timeout 3000 \
      --req 'pause: false' >/tmp/gz_unpause.log 2>&1 || true
  fi
}

wait_for_world_create_service() {
  if ! command -v gz &> /dev/null; then
    return 0
  fi

  local max_wait="$1"
  local elapsed=0
  while [ "$elapsed" -lt "$max_wait" ]; do
    if gz service -l 2>/dev/null | grep -q "/world/${WORLD_NAME}/create"; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  return 1
}

if [ "$GAZEBO_PAUSED_START" != "1" ] && command -v gz &> /dev/null; then
  echo "Ensuring Gazebo is unpaused..."
  unpause_world
fi

# ── Step 3: Spawn UAV (synchronous — must complete before demo) ───
echo ""
echo "[3/4] Spawning UAV..."
echo "Waiting for world create service (up to ${SPAWN_WAIT_SEC}s)..."
if ! wait_for_world_create_service "$SPAWN_WAIT_SEC"; then
  echo "⚠️  World create service not detected within ${SPAWN_WAIT_SEC}s; trying spawn retries anyway..."
fi

SPAWN_OK=0
for attempt in $(seq 1 "$SPAWN_RETRIES"); do
  echo "Spawn attempt ${attempt}/${SPAWN_RETRIES}..."
  if bash ./scripts/worldgen/spawn_uav.sh "$LATEST_WORLD" --index 0; then
    SPAWN_OK=1
    break
  fi
  sleep 2
done

if [ "$SPAWN_OK" -ne 1 ]; then
  echo "❌ UAV spawn failed after ${SPAWN_RETRIES} attempts."
  exit 1
fi

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
    /model/uav1/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
    > /tmp/bridge.log 2>&1 &
  BRIDGE_PID=$!
  echo "Bridge started (PID: $BRIDGE_PID) - logging to /tmp/bridge.log"
else
  echo "⚠️  ROS2 not found, skipping bridge"
fi

# Wait for actual data on ROS2 topics (not just topic existence)
echo "⏳ Waiting for data on ROS2 topics (/odom, /scan)..."
if [ "$GAZEBO_PAUSED_START" = "1" ]; then
  echo "   Gazebo is paused. Click ▶ Play in the Gazebo GUI to start sensors."
  if [ -t 0 ]; then
    echo "   If GUI is not rendering, type 'u' then Enter to unpause from CLI."
    echo -n "   Press Enter after Play (or 'u' + Enter): "
    read -r START_INPUT || true
    if [ "$START_INPUT" = "u" ]; then
      echo "   Unpausing world from CLI..."
      unpause_world
    fi
  fi
fi
MAX_WAIT=$TOPIC_WAIT_SEC
ELAPSED=0
DATA_READY=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
  sleep 2
  ELAPSED=$((ELAPSED + 2))
  if command -v ros2 &> /dev/null; then
    # Check both topics in parallel using echo --once (faster than hz)
    timeout 2 ros2 topic echo /odom --once >/dev/null 2>&1 &
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
  fi
  if [ $((ELAPSED % 6)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
    echo "  Still waiting... ($ELAPSED/$MAX_WAIT sec)"
  fi
done

if [ $DATA_READY -eq 0 ]; then
  echo "❌ Topic data not confirmed within ${MAX_WAIT}s."
  echo "   Start Gazebo (Play) or unpause via CLI, then rerun."
  exit 1
fi

# Run trajectory visualization
echo ""
echo "🎬 Running Gazebo demo..."

if [ -z "$MODEL" ]; then
  LATEST=$(find outputs/runs \( -name "sac_final_model.zip" -o -name "best_model.zip" \) -type f | sort | tail -1)
  if [ -z "$LATEST" ]; then
    echo "❌ No trained model found. Run 'make rl-train' first."
    kill $GAZEBO_PID 2>/dev/null || true
    exit 1
  fi
  MODEL=$LATEST
  echo "Using latest model: $MODEL"
fi

DEMO_CONFIG="configs/training/sac_gazebo.yaml"
DEMO_CONFIG_TMP="/tmp/sac_gazebo_demo_${$}.yaml"
LATEST_META="$PROJECT_ROOT/worldgen/outputs/latest/meta.json"

if [ -f "$MODEL" ]; then
  ADAPTED_CONFIG=$($PYTHON - "$MODEL" "$DEMO_CONFIG" "$DEMO_CONFIG_TMP" "$LATEST_META" <<'PY'
import pickle
import sys
from pathlib import Path

import yaml
import json


def read_lidar_beams_from_config(path: Path):
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return int(data.get("env", {}).get("env_kwargs", {}).get("params", {}).get("lidar_num_beams"))
    except Exception:
        return None


def read_lidar_beams_from_vecnormalize(model_path: Path):
    candidates = [
        model_path.parent / "vecnormalize.pkl",
        model_path.parent.parent / "vecnormalize.pkl",
        model_path.parent.parent / "final" / "vecnormalize.pkl",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("rb") as handle:
                vecnorm = pickle.load(handle)
            obs_space = getattr(vecnorm, "observation_space", None)
            shape = getattr(obs_space, "shape", None)
            if shape and len(shape) == 1 and int(shape[0]) >= 7:
                return int(shape[0]) - 6
        except Exception:
            continue
    return None


model = Path(sys.argv[1]).resolve()
base_cfg = Path(sys.argv[2]).resolve()
out_cfg = Path(sys.argv[3]).resolve()
meta_path = Path(sys.argv[4]).resolve()

run_cfg_candidates = [
    model.parent.parent / "config_used.yaml",
    model.parent / "config_used.yaml",
]

lidar_beams = None
for candidate in run_cfg_candidates:
    if candidate.exists():
        lidar_beams = read_lidar_beams_from_config(candidate)
        if lidar_beams is not None:
            break

if lidar_beams is None:
    lidar_beams = read_lidar_beams_from_vecnormalize(model)

cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
params = cfg.setdefault("env", {}).setdefault("env_kwargs", {}).setdefault("params", {})
if lidar_beams is not None:
    params["lidar_num_beams"] = int(lidar_beams)

if meta_path.exists():
  try:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    world = meta.get("world", {})
    area_size = world.get("area_size")
    if area_size is not None:
      params["world_radius"] = float(area_size) / 2.0

    start_goal = meta.get("start_goal", {})
    goal_xy = start_goal.get("goal_xy") if isinstance(start_goal, dict) else None
    if isinstance(goal_xy, list) and len(goal_xy) >= 2:
      z = float(params.get("default_z_target", 2.0))
      params["fixed_goal"] = [float(goal_xy[0]), float(goal_xy[1]), z]
      params["randomize_goal_on_reset"] = False
  except Exception:
    pass

out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(str(out_cfg))
PY
)

  if [ -n "$ADAPTED_CONFIG" ] && [ -f "$ADAPTED_CONFIG" ]; then
    DEMO_CONFIG="$ADAPTED_CONFIG"
    echo "Using adapted Gazebo config: $DEMO_CONFIG"
  fi
fi

$PYTHON -m forest_nav_rl.visualize_trajectories \
  --model "$MODEL" \
  --config "$DEMO_CONFIG" \
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
