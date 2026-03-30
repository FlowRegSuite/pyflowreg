"""Tests for optical flow stage dispatch and GNC plumbing."""

import numpy as np

from pyflowreg.core.optical_flow import get_displacement


def _compress_runs(values):
    """Collapse consecutive duplicate values while preserving order."""
    compressed = []
    for value in values:
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    return compressed


def test_get_displacement_without_gnc_uses_baseline_stage():
    """Test the default path does not pass a GNC stage to the solver."""
    fixed = np.zeros((12, 12), dtype=np.float64)
    moving = np.zeros((12, 12), dtype=np.float64)
    seen_betas = []

    def fake_level_solver(
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
        gnc_beta=None,
    ):
        seen_betas.append(gnc_beta)
        return np.zeros_like(u), np.zeros_like(v)

    flow = get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=fake_level_solver,
    )

    assert flow.shape == (12, 12, 2)
    assert _compress_runs(seen_betas) == [None]


def test_get_displacement_with_gnc_repeats_pyramid_per_stage():
    """Test GNC reruns the pyramid with one fixed stage weight per pass."""
    fixed = np.zeros((12, 12), dtype=np.float64)
    moving = np.zeros((12, 12), dtype=np.float64)
    seen_betas = []

    def fake_level_solver(
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
        gnc_beta=None,
    ):
        seen_betas.append(gnc_beta)
        return np.zeros_like(u), np.zeros_like(v)

    flow = get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=fake_level_solver,
        gnc_schedule=(0.0, 0.5, 1.0),
    )

    assert flow.shape == (12, 12, 2)
    assert _compress_runs(seen_betas) == [0.0, 0.5, 1.0]
