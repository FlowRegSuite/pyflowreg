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


# PyTorch backend (works on CPU or GPU)
try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    def _flowreg_torch_factory(device=None, dtype="float64", **kwargs):
        """
        Factory for PyTorch backend (GPU or CPU).

        Parameters
        ----------
        device : str, optional
            PyTorch device string (e.g., "cuda", "cuda:0", "cpu").
            If None, defaults to "cuda" if available, otherwise "cpu".
        dtype : str
            Data type for tensors ("float32" or "float64")
        **kwargs : dict
            Additional parameters (for compatibility)
        """
        import torch
        import warnings
        from functools import partial
        from pyflowreg.torch.core.level_solver import level_solver_rbgs_torch

        # Default device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if CUDA is available when requested
        if device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                f"CUDA device '{device}' requested but CUDA is not available. "
                "Falling back to CPU. This will be slower than GPU acceleration. "
                "To use GPU, ensure CUDA and a CUDA-enabled PyTorch are installed."
            )
            device = "cpu"

        def torch_level_solver(
            J11,
            J22,
            J33,
            J12,
            J13,
            J23,
            weight,
            u,
            v,
            alpha,
            iterations,
            update_lag,
            verbose,
            a_data,
            a_smooth,
            hx,
            hy,
        ):
            # Convert to tensors
            dtype_map = {"float64": torch.float64, "float32": torch.float32}
            torch_dtype = dtype_map.get(dtype, torch.float64)

            def to_tensor(a):
                return torch.as_tensor(a, dtype=torch_dtype, device=device)

            # Call PyTorch solver
            du, dv = level_solver_rbgs_torch(
                to_tensor(J11),
                to_tensor(J22),
                to_tensor(J33),
                to_tensor(J12),
                to_tensor(J13),
                to_tensor(J23),
                to_tensor(weight),
                to_tensor(u),
                to_tensor(v),
                alpha,
                iterations,
                update_lag,
                a_data,
                a_smooth,
                hx,
                hy,
            )

            return du.cpu().numpy(), dv.cpu().numpy()

        # Return a partial function with the custom level solver
        return partial(_flowreg_get, level_solver_backend=torch_level_solver)

    register_backend(
        "flowreg_torch", _flowreg_torch_factory, supported_executors={"sequential"}
    )


# CUDA backend (only if cupy is available)
try:
    import warnings

    # Suppress CuPy CUDA path warning during availability check
    warnings.filterwarnings(
        "ignore",
        message="CUDA path could not be detected",
        category=UserWarning,
        module=r"cupy\._environment",
    )
    import cupy as cp  # noqa: F401

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

if CUPY_AVAILABLE:

    def _flowreg_cuda_factory(**kwargs):
        """
        Factory for CuPy CUDA backend.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters (for compatibility)
        """
        from functools import partial
        from pyflowreg.cuda.core.level_solver import level_solver_rbgs_cuda

        def cuda_level_solver(
            J11,
            J22,
            J33,
            J12,
            J13,
            J23,
            weight,
            u,
            v,
            alpha,
            iterations,
            update_lag,
            verbose,
            a_data,
            a_smooth,
            hx,
            hy,
        ):
            # CUDA solver handles numpy/cupy conversion internally
            return level_solver_rbgs_cuda(
                J11,
                J22,
                J33,
                J12,
                J13,
                J23,
                weight,
                u,
                v,
                alpha,
                iterations,
                update_lag,
                a_data,
                a_smooth,
                hx,
                hy,
            )

        # Return a partial function with the custom level solver
        return partial(_flowreg_get, level_solver_backend=cuda_level_solver)

    register_backend(
        "flowreg_cuda", _flowreg_cuda_factory, supported_executors={"sequential"}
    )
