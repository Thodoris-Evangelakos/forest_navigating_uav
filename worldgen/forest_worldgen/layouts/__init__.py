"""
Layout generators for spatial arrangement of objects
"""

from .single_zone import generate_single_zone
from .mixed_zones import generate_mixed_zones
from .mixed_field import generate_mixed_field

LAYOUT_HANDLERS = {
    'single': generate_single_zone,
    'zones': generate_mixed_zones,
    'mixture_field': generate_mixed_field,
}

__all__ = [
    'generate_single_zone',
    'generate_mixed_zones',
    'generate_mixed_field',
    'LAYOUT_HANDLERS',
]
