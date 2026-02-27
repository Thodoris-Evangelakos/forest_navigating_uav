#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-sdf", required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    world_sdf = Path(args.world_sdf).resolve()

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    params = cfg.get("env", {}).get("env_kwargs", {}).get("params", {})
    goal = params.get("fixed_goal", [8.0, 0.0, 0.3])

    gx, gy, gz = 8.0, 0.0, 0.3
    if isinstance(goal, list) and len(goal) >= 2:
        gx = float(goal[0])
        gy = float(goal[1])
        if len(goal) >= 3:
            gz = float(goal[2])

    world_name = "randomized_world"
    if world_sdf.exists():
        text = world_sdf.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'<world name="([^"]+)">', text)
        if match:
            world_name = match.group(1)

    print(f"{gx} {gy} {gz} {world_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
