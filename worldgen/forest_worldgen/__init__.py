"""
Forest World Generation Package

A modular world generation system for creating randomized forest environments
with configurable spatial distributions and layouts.
"""

__version__ = "0.3.0"

from . import patterns
from . import layouts
from . import export
from . import spatial_stats
from .config import load_config, load_template, resolve_path

# NOTE: generate_world.main is intentionally NOT imported here.
# It is a CLI entry point executed via `python -m`, and eagerly importing it
# causes a RuntimeWarning ("found in sys.modules … prior to execution").

__all__ = [
    'patterns',
    'layouts',
    'export',
    'spatial_stats',
    'load_config',
    'load_template',
    'resolve_path',
]
