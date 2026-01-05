"""Upsampling methods for terravector."""

from .registry import (
    UPSAMPLING_REGISTRY,
    register_upsampling,
    get_upsampling,
    list_upsamplings,
    run_upsampling,
)

# Import methods to register them
from . import methods
from . import methods_extended

