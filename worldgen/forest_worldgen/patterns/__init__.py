"""
Spatial point patterns for world generation.
"""

from .csr import sample_csr
from .regular import sample_regular
from .clustered import sample_clustered

PATTERN_SAMPLERS = {
    'csr': sample_csr,
    'regular': sample_regular,
    'clustered': sample_clustered,
}

__all__ = [
    'sample_csr',
    'sample_regular',
    'sample_clustered',
    'PATTERN_SAMPLERS',
]
