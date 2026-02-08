"""
Spatial point patterns for world generation.
"""

from .csr import sample_csr
from .regular import sample_regular
from .clustered import sample_clustered
from .scale_dependent import sample_scale_dependent

PATTERN_SAMPLERS = {
    'csr': sample_csr,
    'regular': sample_regular,
    'clustered': sample_clustered,
    'scale_dependent': sample_scale_dependent,
}

__all__ = [
    'sample_csr',
    'sample_regular',
    'sample_clustered',
    'sample_scale_dependent',
    'PATTERN_SAMPLERS',
]
