"""
Backend Registry for Optical Flow Algorithms
=============================================

Provides a centralized registry for different optical flow backends,
allowing runtime selection and configuration of flow algorithms.
"""

from typing import Callable, Dict, Set, Optional

# Global registry of backend factories
_BACKENDS: Dict[str, Callable[..., Callable]] = {}

# Global registry of supported parallelization executors for each backend
# Uses exact names from RuntimeContext for direct membership testing
_BACKEND_EXECUTORS: Dict[str, Set[str]] = {}


def register_backend(
    name: str,
    factory: Callable[..., Callable],
    supported_executors: Optional[Set[str]] = None,
) -> None:
    """
    Register an optical flow backend factory with parallelization executor support.

    Args:
        name: Backend identifier (e.g., 'flowreg', 'diso')
        factory: Factory function that returns a callable for computing flow
        supported_executors: Set of supported parallelization executor names using
            exact RuntimeContext names: {'sequential', 'threading', 'multiprocessing',
            'multiprocessing_fork', 'multiprocessing_spawn'}.
            If None, assumes all executors are supported.
    """
    _BACKENDS[name] = factory

    # Default: support all common executors
    if supported_executors is None:
        supported_executors = {
            "sequential",
            "threading",
            "multiprocessing",
            "multiprocessing_fork",
            "multiprocessing_spawn",
        }

    _BACKEND_EXECUTORS[name] = supported_executors


def get_backend(name: str) -> Callable[..., Callable]:
    """
    Get a registered backend factory by name.

    Args:
        name: Backend identifier

    Returns:
        Factory function for the backend

    Raises:
        ValueError: If backend not found
    """
    if name not in _BACKENDS:
        available = list(_BACKENDS.keys())
        raise ValueError(
            f"Unknown flow backend: '{name}'. " f"Available backends: {available}"
        )
    return _BACKENDS[name]


def list_backends() -> list[str]:
    """Get list of available backend names."""
    return list(_BACKENDS.keys())


def is_backend_available(name: str) -> bool:
    """Check if a backend is registered."""
    return name in _BACKENDS


def get_backend_executors(name: str) -> Set[str]:
    """
    Get the set of supported parallelization executors for a backend.

    Args:
        name: Backend identifier

    Returns:
        Set of supported parallelization executor names

    Raises:
        ValueError: If backend not found
    """
    if name not in _BACKEND_EXECUTORS:
        if name in _BACKENDS:
            # Backend exists but no metadata, assume all executors supported
            return {
                "sequential",
                "threading",
                "multiprocessing",
                "multiprocessing_fork",
                "multiprocessing_spawn",
            }
        raise ValueError(f"Unknown flow backend: '{name}'")
    return _BACKEND_EXECUTORS[name].copy()
