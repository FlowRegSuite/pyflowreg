"""
Core Optical Flow Computation Module
=====================================

This module provides the core variational optical flow computation engine
for PyFlowReg, implementing pyramid-based multi-scale flow estimation with
non-linear diffusion regularization.

The module includes:
- Main optical flow solver (get_displacement via backend system)
- Low-level flow computation at each pyramid level (compute_flow)
- Backend registration system for multiple flow implementations

Available Backends
------------------
flowreg : Default variational optical flow implementation
    Full-featured gradient constancy optical flow with pyramid scheme
diso : Planned numba-based reimplementation
gpu : Planned GPU-accelerated implementation

Functions
---------
compute_flow
    Low-level numba-optimized flow field solver
register_backend
    Register new optical flow backend
get_backend
    Retrieve registered backend by name
list_backends
    List all available backends
is_backend_available
    Check if a specific backend is available

See Also
--------
pyflowreg.core.optical_flow : Main optical flow implementation
pyflowreg.core.level_solver : Pyramid level solver

Notes
-----
The core implementation maintains algorithmic compatibility with the
MATLAB Flow-Registration toolbox while leveraging numba optimization
for performance.
"""

from .level_solver import compute_flow
from .backend_registry import (
    register_backend,
    get_backend,
    list_backends,
    is_backend_available,
)

__all__ = [
    "compute_flow",
    "register_backend",
    "get_backend",
    "list_backends",
    "is_backend_available",
]

# Register built-in backends
# Default flowreg backend
from .optical_flow import get_displacement as _flowreg_get


def _flowreg_factory(**kwargs):
    """Factory for the default FlowReg backend."""
    return _flowreg_get


register_backend("flowreg", _flowreg_factory)

# DISO backend (only if OpenCV is available)
try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

if CV2_AVAILABLE:
    from .diso_optical_flow import _diso_factory

    register_backend("diso", _diso_factory)
