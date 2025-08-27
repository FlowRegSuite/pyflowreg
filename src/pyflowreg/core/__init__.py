from .level_solver import compute_flow
from .backend_registry import register_backend, get_backend, list_backends, is_backend_available

__all__ = ["compute_flow", "register_backend", "get_backend", "list_backends", "is_backend_available"]

# Register built-in backends
# Default flowreg backend
from .optical_flow import get_displacement as _flowreg_get
def _flowreg_factory(**kwargs):
    """Factory for the default FlowReg backend."""
    return _flowreg_get
register_backend("flowreg", _flowreg_factory)

# DISO backend (only if OpenCV is available)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

if CV2_AVAILABLE:
    from .diso_optical_flow import _diso_factory
    register_backend("diso", _diso_factory)