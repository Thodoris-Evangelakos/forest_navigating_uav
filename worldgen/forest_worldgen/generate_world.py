"""
Compatibility module for world generation.

Public API remains available here:
- generate_positions_from_config
- run_generation
- main (CLI entry)
"""

from __future__ import annotations

from .cli import main
from .pipeline import generate_positions_from_config, run_generation

__all__ = ["generate_positions_from_config", "run_generation", "main"]


if __name__ == "__main__":
    main()
