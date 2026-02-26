#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL=${1:-}
NUM_EPISODES=${2:-5}
TOPIC_WAIT_SEC=${TOPIC_WAIT_SEC:-120}
KILL_STALE_BRIDGE=${KILL_STALE_BRIDGE:-1}

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
DEMO_CONFIG_TMP=""

cleanup() {
  echo ""
  echo "🛑 Cleaning up agent bridge..."
  [ -n "$BRIDGE_PID" ] && kill $BRIDGE_PID 2>/dev/null || true
  sleep 1
  [ -n "$BRIDGE_PID" ] && kill -9 $BRIDGE_PID 2>/dev/null || true
  wait $BRIDGE_PID 2>/dev/null || true
  [ -n "$DEMO_CONFIG_TMP" ] && rm -f "$DEMO_CONFIG_TMP" 2>/dev/null || true
  echo "✓ Agent bridge cleanup done"
}
trap cleanup EXIT

echo "🚀 Starting Gazebo agent control..."
echo "  Episodes: $NUM_EPISODES"
if [ -n "$MODEL" ]; then
  echo "  Model: $MODEL"
fi

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
  echo "❌ ROS2 not found. Source ROS2 first."
  exit 1
fi

echo "Starting ROS2-Gazebo bridge..."
ros2 run ros_gz_bridge parameter_bridge \
  /odom@nav_msgs/msg/Odometry[gz.msgs.Odometry \
  /scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan \
  /model/uav1/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
  > /tmp/bridge.log 2>&1 &
BRIDGE_PID=$!
echo "Bridge started (PID: $BRIDGE_PID) - logging to /tmp/bridge.log"

echo "⏳ Waiting for data on ROS2 topics (/odom, /scan)..."
MAX_WAIT=$TOPIC_WAIT_SEC
ELAPSED=0
DATA_READY=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
  sleep 2
  ELAPSED=$((ELAPSED + 2))

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

  if [ $((ELAPSED % 6)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
    echo "  Still waiting... ($ELAPSED/$MAX_WAIT sec)"
  fi
done

if [ $DATA_READY -eq 0 ]; then
  echo "❌ Topic data not confirmed within ${MAX_WAIT}s."
  echo "   Ensure Gazebo is running, world is unpaused, and UAV is spawned."
  exit 1
fi

echo ""
echo "🎬 Running Gazebo policy control..."
if [ -z "$MODEL" ]; then
  LATEST=$(find outputs/runs \( -name "sac_final_model.zip" -o -name "best_model.zip" \) -type f | sort | tail -1)
  if [ -z "$LATEST" ]; then
    echo "❌ No trained model found. Run 'make rl-train' first."
    exit 1
  fi
  MODEL=$LATEST
  echo "Using latest model: $MODEL"
fi

DEMO_CONFIG="configs/training/sac_gazebo.yaml"
DEMO_CONFIG_TMP="/tmp/sac_gazebo_agent_${$}.yaml"
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

AGENT_EXIT=$?
if [ $AGENT_EXIT -eq 0 ]; then
  echo "✓ Gazebo agent run complete"
else
  echo "❌ Gazebo agent failed with exit code $AGENT_EXIT"
fi

exit $AGENT_EXIT
