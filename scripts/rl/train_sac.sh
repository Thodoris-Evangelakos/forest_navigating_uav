#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/training/sac.yaml}"
DEVICE="${2:-auto}"

forest-nav-train-sac --config "$CONFIG_PATH" --device "$DEVICE"
