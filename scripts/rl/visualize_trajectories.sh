#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Error: MODEL path required as first argument}"
shift

NUM_EPISODES="${NUM_EPISODES:-5}"
CONFIG="${CONFIG:-}"

if [[ -n "$CONFIG" && -f "$CONFIG" ]]; then
  forest-nav-trajectories --model "$MODEL" --config "$CONFIG" --num-episodes "$NUM_EPISODES" "$@"
else
  forest-nav-trajectories --model "$MODEL" --num-episodes "$NUM_EPISODES" "$@"
fi
