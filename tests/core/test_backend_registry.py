"""
Tests for the optical flow backend registry.
"""

import pytest

import pyflowreg.core  # noqa: F401  # triggers built-in backend registration
from pyflowreg.core.backend_registry import (
    get_backend_executors,
    is_backend_available,
)


class TestBackendExecutorSupport:
    """Test executor-support metadata of the built-in backends."""

    def test_get_backend_executors_flowreg_supports_multiprocessing(self):
        executors = get_backend_executors("flowreg")
        assert "multiprocessing" in executors

    def test_get_backend_executors_diso_excludes_multiprocessing(self):
        """The multiprocessing workers import the flowreg solver directly and
        never reconstruct registry backends, so selecting diso there would
        silently run the variational solver instead of DIS."""
        if not is_backend_available("diso"):
            pytest.skip("diso backend not available (OpenCV missing)")

        executors = get_backend_executors("diso")
        assert executors == {"sequential", "threading"}
