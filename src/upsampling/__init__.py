"""Upsampling methods for terravector.

Thin facade over ``residuals.upsampling`` (the RESIDUALS sibling project).
Implementations and the registry live there; this module only re-exports
them so internal imports keep the ``src.upsampling`` spelling.

To add an upsampling method, register it in RESIDUALS and it becomes
available here automatically.
"""

from residuals.upsampling import (  # noqa: F401  (re-exports)
    UPSAMPLING_REGISTRY,
    get_all_methods_info,
    get_upsampling,
    list_upsamplings,
    register_upsampling,
    run_upsampling,
)

__all__ = [
    "UPSAMPLING_REGISTRY",
    "get_all_methods_info",
    "get_upsampling",
    "list_upsamplings",
    "register_upsampling",
    "run_upsampling",
]
