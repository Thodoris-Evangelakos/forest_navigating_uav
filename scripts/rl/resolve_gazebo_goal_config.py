#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    meta_path = Path(args.meta).resolve()
    out_path = Path(args.out).resolve()

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    params = cfg.setdefault("env", {}).setdefault("env_kwargs", {}).setdefault("params", {})

    default_z = float(params.get("default_z_target", 0.3))

    goal_xy = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            start_goal = meta.get("start_goal", {})
            if isinstance(start_goal, dict):
                maybe_goal = start_goal.get("goal_xy")
                if isinstance(maybe_goal, list) and len(maybe_goal) >= 2:
                    goal_xy = [float(maybe_goal[0]), float(maybe_goal[1])]
        except Exception:
            goal_xy = None

    if goal_xy is not None:
        params["fixed_goal"] = [goal_xy[0], goal_xy[1], default_z]
        params["randomize_goal_on_reset"] = False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
