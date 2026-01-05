"""Decomposition methods for terravector.

Ported from DIVERGE/RESIDUALS project.
"""

from .registry import (
    DECOMPOSITION_REGISTRY,
    register_decomposition,
    get_decomposition,
    list_decompositions,
    run_decomposition,
)

# Import methods to register them
from .methods import (
    decompose_gaussian,
    decompose_bilateral,
    decompose_wavelet_dwt,
    decompose_morphological,
    decompose_tophat,
    decompose_polynomial,
    DECOMPOSITION_METHODS,
    DEFAULT_PARAMS,
)

# Import extended methods
from . import methods_extended

