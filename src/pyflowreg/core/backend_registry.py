"""
Backend Registry for Optical Flow Algorithms
=============================================

Provides a centralized registry for different optical flow backends,
allowing runtime selection and configuration of flow algorithms.
"""

from typing import Callable, Dict

# Global registry of backend factories
_BACKENDS: Dict[str, Callable[..., Callable]] = {}


def register_backend(name: str, factory: Callable[..., Callable]) -> None:
    """
    Register an optical flow backend factory.

    Args:
        name: Backend identifier (e.g., 'flowreg', 'diso')
        factory: Factory function that returns a callable for computing flow
    """
    _BACKENDS[name] = factory


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
