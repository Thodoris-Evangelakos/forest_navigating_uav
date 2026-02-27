"""Provide the command-line interface for forest world generation."""

from __future__ import annotations

import argparse

from .pipeline import run_generation


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m forest_worldgen.generate_world",
        description="Generate forest world assets (SDF + metadata + preview)",
    )
    parser.add_argument("run_config", help="Path to worldgen run config YAML")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for deterministic generation"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the world-generation CLI entry point."""
    args = build_parser().parse_args(argv)
    run_generation(args.run_config, seed=args.seed, apply_start_goal_exclusion=True)


if __name__ == "__main__":
    main()
