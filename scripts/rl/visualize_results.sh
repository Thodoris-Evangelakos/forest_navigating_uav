#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--compare" ]]; then
  forest-nav-visualize --compare "${@:2}"
  exit 0
fi

RUN_DIR="${1:-}"
if [[ -n "$RUN_DIR" ]]; then
  forest-nav-visualize --run-dir "$RUN_DIR" "${@:2}"
else
  forest-nav-visualize "${@:2}"
fi
