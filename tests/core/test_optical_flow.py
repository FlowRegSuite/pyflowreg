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


def test_get_displacement_without_gnc_ignores_warping_steps():
    """Test warping_steps alone does not change the default path."""
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
        warping_steps=3,
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


def test_get_displacement_with_gnc_warping_steps_repeats_solver_calls():
    """Test optional GNC warping steps repeat the per-level solver calls."""
    fixed = np.zeros((12, 12), dtype=np.float64)
    moving = np.zeros((12, 12), dtype=np.float64)
    base_seen_betas = []
    warp_seen_betas = []

    def make_fake_level_solver(sink):
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
            sink.append(gnc_beta)
            return np.zeros_like(u), np.zeros_like(v)

        return fake_level_solver

    get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=make_fake_level_solver(base_seen_betas),
        gnc_schedule=(0.0, 0.5, 1.0),
    )
    get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=make_fake_level_solver(warp_seen_betas),
        gnc_schedule=(0.0, 0.5, 1.0),
        warping_steps=3,
    )

    for beta in (0.0, 0.5, 1.0):
        assert warp_seen_betas.count(beta) == 3 * base_seen_betas.count(beta)
