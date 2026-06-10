"""Tests for low-level GNC penalty blending in the 2D solver."""

import numpy as np

from pyflowreg.core.level_solver import compute_flow_gnc


def _data_term_case(a_data=0.45):
    """Create a one-pixel data force with smoothness in the denominator."""
    shape = (3, 3, 1)
    field_shape = shape[:2]
    tensors = [np.zeros(shape, dtype=np.float64) for _ in range(6)]
    J11, J22, J33, J12, J13, J23 = tensors
    J11[1, 1, 0] = 1.0
    J13[1, 1, 0] = -1.0
    J33[1, 1, 0] = 4.0
    return dict(
        J11=J11,
        J22=J22,
        J33=J33,
        J12=J12,
        J13=J13,
        J23=J23,
        weight=np.ones(shape, dtype=np.float64),
        u=np.zeros(field_shape, dtype=np.float64),
        v=np.zeros(field_shape, dtype=np.float64),
        alpha_x=1.0,
        alpha_y=1.0,
        iterations=1,
        a_data=np.array([a_data], dtype=np.float64),
        a_smooth=1.0,
        hx=1.0,
        hy=1.0,
    )


def _smoothness_case(a_smooth):
    """Create a non-flat initial flow so smoothness weights affect the solve."""
    shape = (5, 5, 1)
    field_shape = shape[:2]
    u = np.zeros(field_shape, dtype=np.float64)
    u[2, 2] = 1.0
    return dict(
        J11=np.zeros(shape, dtype=np.float64),
        J22=np.zeros(shape, dtype=np.float64),
        J33=np.zeros(shape, dtype=np.float64),
        J12=np.zeros(shape, dtype=np.float64),
        J13=np.zeros(shape, dtype=np.float64),
        J23=np.zeros(shape, dtype=np.float64),
        weight=np.ones(shape, dtype=np.float64),
        u=u,
        v=np.zeros(field_shape, dtype=np.float64),
        alpha_x=1.0,
        alpha_y=1.0,
        iterations=1,
        a_data=np.array([1.0], dtype=np.float64),
        a_smooth=a_smooth,
        hx=1.0,
        hy=1.0,
    )


def test_compute_flow_gnc_updates_data_weights_on_matlab_cadence():
    """Test GNC data weights update on the MATLAB-style update-lag tick."""
    common = _data_term_case()

    lagged_quadratic = compute_flow_gnc(**common, update_lag=2, gnc_beta=0.0)
    lagged_robust = compute_flow_gnc(**common, update_lag=2, gnc_beta=1.0)
    np.testing.assert_allclose(lagged_quadratic, lagged_robust)

    updated_quadratic = compute_flow_gnc(**common, update_lag=1, gnc_beta=0.0)
    updated_midpoint = compute_flow_gnc(**common, update_lag=1, gnc_beta=0.5)
    updated_robust = compute_flow_gnc(**common, update_lag=1, gnc_beta=1.0)

    assert updated_quadratic[1, 1, 0] > updated_midpoint[1, 1, 0]
    assert updated_midpoint[1, 1, 0] > updated_robust[1, 1, 0]


def test_compute_flow_gnc_leaves_quadratic_data_term_beta_invariant():
    """Test ``a_data=1`` is quadratic already and ignores the GNC beta."""
    common = _data_term_case(a_data=1.0)

    quadratic_stage = compute_flow_gnc(**common, update_lag=1, gnc_beta=0.0)
    robust_stage = compute_flow_gnc(**common, update_lag=1, gnc_beta=1.0)

    np.testing.assert_allclose(quadratic_stage, robust_stage)


def test_compute_flow_gnc_blends_sublinear_smoothness_weights():
    """Test sublinear smoothness weights change with the GNC stage beta."""
    common = _smoothness_case(a_smooth=0.5)

    quadratic_stage = compute_flow_gnc(**common, update_lag=1, gnc_beta=0.0)
    robust_stage = compute_flow_gnc(**common, update_lag=1, gnc_beta=1.0)

    assert not np.allclose(quadratic_stage, robust_stage)


def test_compute_flow_gnc_leaves_quadratic_smoothness_beta_invariant():
    """Test ``a_smooth=1`` is quadratic already and ignores the GNC beta."""
    common = _smoothness_case(a_smooth=1.0)

    quadratic_stage = compute_flow_gnc(**common, update_lag=1, gnc_beta=0.0)
    robust_stage = compute_flow_gnc(**common, update_lag=1, gnc_beta=1.0)

    np.testing.assert_allclose(quadratic_stage, robust_stage)
