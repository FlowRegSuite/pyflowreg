"""Tests for Stage 2 inter-sequence optical-flow configuration."""

import numpy as np
import pytest

from pyflowreg.session.config import SessionConfig
import pyflowreg.session.stage2_between_avgs as stage2


@pytest.fixture
def stage2_lightweight_ops(monkeypatch):
    """Patch expensive image operations so tests isolate configuration wiring."""
    monkeypatch.setattr(stage2, "gaussian_filter", lambda image, sigma: image)
    monkeypatch.setattr(
        stage2,
        "estimate_rigid_xcorr_2d",
        lambda img1, img2, up: (0, 0),
    )
    monkeypatch.setattr(
        stage2,
        "imregister_wrapper",
        lambda img, u, v, ref, interpolation_method="cubic": img,
    )


def test_compute_between_displacement_passes_stage2_constancy(
    tmp_path, monkeypatch, stage2_lightweight_ops
):
    """Stage 2 should pass its data term selector into get_displacement."""
    captured = {}

    def fake_get_backend(name):
        assert name == "flowreg"

        def factory(**backend_params):
            assert backend_params == {"sentinel": True}

            def get_displacement(fixed, moving, **kwargs):
                captured.update(kwargs)
                return np.zeros((*fixed.shape, 2), dtype=np.float32)

            return get_displacement

        return factory

    monkeypatch.setattr(stage2, "get_backend", fake_get_backend)

    config = SessionConfig(
        root=tmp_path,
        backend_params={"sentinel": True},
        stage2_constancy_assumption="census",
    )
    reference_avg = np.arange(16, dtype=np.float64).reshape(4, 4)
    current_avg = reference_avg + 1

    w = stage2.compute_between_displacement(reference_avg, current_avg, config)

    assert captured["const_assumption"] == "cs"
    assert captured["alpha"] == (25.0, 25.0)
    assert captured["iterations"] == 100
    assert w.shape == (4, 4, 2)


def test_compute_between_displacement_rejects_diso_non_default_constancy(
    tmp_path, stage2_lightweight_ops
):
    """DISO should fail explicitly for flowreg-only data terms."""
    config = SessionConfig(
        root=tmp_path,
        flow_backend="diso",
        stage2_constancy_assumption="gray",
    )
    reference_avg = np.arange(16, dtype=np.float64).reshape(4, 4)
    current_avg = reference_avg + 1

    with pytest.raises(ValueError, match="does not support"):
        stage2.compute_between_displacement(reference_avg, current_avg, config)
