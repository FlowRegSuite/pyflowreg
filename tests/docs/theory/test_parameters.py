"""
Executable-documentation tests for docs/theory/parameters.md.

Each test runs the extracted snippet from ``docs/snippets/theory/parameters/``
via the ``snippet_runner`` fixture and additionally checks that the
documented recommended ``sigma`` values pass the ``OFOptions`` sigma
validator (per-channel ``[sx, sy, st]`` rows, shape ``(n_channels, 3)``).
"""

import numpy as np
import pytest

from pyflowreg.motion_correction import OFOptions

pytestmark = pytest.mark.docs_example


class TestParametersSigmaCalcium:
    """docs/theory/parameters.md, "Recommended Settings" (calcium imaging)."""

    def test_sigma_calcium_executes(self, snippet_runner):
        ns = snippet_runner("theory/parameters/sigma_calcium.py")

        sigma = ns["sigma"]
        assert sigma == [[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]]

        options = OFOptions(sigma=sigma)
        assert np.asarray(options.sigma).shape == (2, 3)
        np.testing.assert_allclose(options.get_sigma_at(0), [1.0, 1.0, 0.1])
        np.testing.assert_allclose(options.get_sigma_at(1), [1.0, 1.0, 0.1])


class TestParametersSigmaLowSnr:
    """docs/theory/parameters.md, "Recommended Settings" (low SNR)."""

    def test_sigma_low_snr_executes(self, snippet_runner):
        ns = snippet_runner("theory/parameters/sigma_low_snr.py")

        sigma = ns["sigma"]
        assert sigma == [[1.5, 1.5, 0.3], [1.5, 1.5, 0.3]]

        options = OFOptions(sigma=sigma)
        assert np.asarray(options.sigma).shape == (2, 3)
        np.testing.assert_allclose(options.get_sigma_at(0), [1.5, 1.5, 0.3])
        np.testing.assert_allclose(options.get_sigma_at(1), [1.5, 1.5, 0.3])


class TestParametersSigmaVolumetric:
    """docs/theory/parameters.md, "Recommended Settings" (volumetric)."""

    def test_sigma_volumetric_executes(self, snippet_runner):
        ns = snippet_runner("theory/parameters/sigma_volumetric.py")

        sigma = ns["sigma"]
        assert sigma == [[1.0, 1.0, 0.5], [1.0, 1.0, 0.5]]

        options = OFOptions(sigma=sigma)
        assert np.asarray(options.sigma).shape == (2, 3)
        np.testing.assert_allclose(options.get_sigma_at(0), [1.0, 1.0, 0.5])
        np.testing.assert_allclose(options.get_sigma_at(1), [1.0, 1.0, 0.5])
