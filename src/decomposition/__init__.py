"""Decomposition methods for terravector.

This package is a thin facade over the ``residuals`` library (the RESIDUALS
sibling project, https://github.com/bshepp/RESIDUALS). The method
implementations, their default parameters, and the decorator-based registry
all live there; terravector only re-exports them so internal imports keep
the ``src.decomposition`` spelling.

To add a decomposition method, register it in RESIDUALS (see its
CONTRIBUTING.md) and it becomes available here automatically.

``DECOMPOSITION_METHODS`` / ``DEFAULT_PARAMS`` are the legacy core-method
dicts consumed by the "decomposition" signature family. Their membership and
ORDER define the dimension layout of that family's embedding slice, so they
are pinned explicitly rather than derived from the full registry.
"""

from residuals.decomposition import (  # noqa: F401  (re-exports)
    DECOMPOSITION_REGISTRY,
    get_all_methods_info,
    get_decomposition,
    list_decompositions,
    register_decomposition,
    run_decomposition,
)
from residuals.decomposition.methods import (  # noqa: F401  (re-exports)
    decompose_bilateral,
    decompose_gaussian,
    decompose_morphological,
    decompose_polynomial,
    decompose_tophat,
    decompose_wavelet_dwt,
)

# Legacy core-method API. Order is load-bearing: it fixes the slice layout of
# the "decomposition" signature family (6 stats x N methods) when a config
# does not list methods explicitly. Do not reorder or extend without
# treating it as an index-invalidating change.
_CORE_METHOD_NAMES = (
    "gaussian",
    "bilateral",
    "wavelet_dwt",
    "morphological",
    "tophat",
    "polynomial",
)

DECOMPOSITION_METHODS = {
    name: DECOMPOSITION_REGISTRY[name].func for name in _CORE_METHOD_NAMES
}
DEFAULT_PARAMS = {
    name: dict(DECOMPOSITION_REGISTRY[name].default_params)
    for name in _CORE_METHOD_NAMES
}

__all__ = [
    "DECOMPOSITION_REGISTRY",
    "DECOMPOSITION_METHODS",
    "DEFAULT_PARAMS",
    "get_all_methods_info",
    "get_decomposition",
    "list_decompositions",
    "register_decomposition",
    "run_decomposition",
    "decompose_bilateral",
    "decompose_gaussian",
    "decompose_morphological",
    "decompose_polynomial",
    "decompose_tophat",
    "decompose_wavelet_dwt",
]
