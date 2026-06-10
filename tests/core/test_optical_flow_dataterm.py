"""Tests for optical-flow data term dispatch."""

import numpy as np
import pytest

import pyflowreg.core.optical_flow as optical_flow


def _zero_level_solver(
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
    """Return no correction so tests isolate motion tensor dispatch."""
    return np.zeros_like(u), np.zeros_like(v)


def _make_motion_tensor(name, calls):
    def motion_tensor(f1, f2, hy, hx):
        calls.append(name)
        shape = (f1.shape[0] + 2, f1.shape[1] + 2)
        return tuple(np.zeros(shape, dtype=np.float64) for _ in range(6))

    return motion_tensor


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "gc"),
        ("gc", "gc"),
        ("gradient", "gc"),
        ("gray", "gray"),
        ("brightness", "gray"),
        ("cs", "cs"),
        ("census", "cs"),
    ],
)
def test_get_displacement_dispatches_constancy_assumption(monkeypatch, value, expected):
    """get_displacement should route each selector to its data term."""
    calls = []
    monkeypatch.setattr(
        optical_flow, "get_motion_tensor_gc", _make_motion_tensor("gc", calls)
    )
    monkeypatch.setattr(
        optical_flow, "get_motion_tensor_gray", _make_motion_tensor("gray", calls)
    )
    monkeypatch.setattr(
        optical_flow, "get_motion_tensor_cs", _make_motion_tensor("cs", calls)
    )

    fixed = np.zeros((16, 16), dtype=np.float64)
    moving = np.zeros((16, 16), dtype=np.float64)
    kwargs = {} if value is None else {"const_assumption": value}

    optical_flow.get_displacement(
        fixed,
        moving,
        levels=1,
        min_level=0,
        iterations=1,
        level_solver_backend=_zero_level_solver,
        **kwargs,
    )

    assert calls
    assert set(calls) == {expected}


def test_get_displacement_rejects_unknown_constancy_assumption():
    """Unknown data terms should fail before any solver work starts."""
    fixed = np.zeros((16, 16), dtype=np.float64)
    moving = np.zeros((16, 16), dtype=np.float64)

    with pytest.raises(ValueError, match="Unknown constancy assumption"):
        optical_flow.get_displacement(
            fixed,
            moving,
            const_assumption="invalid",
            level_solver_backend=_zero_level_solver,
        )


def test_get_displacement_with_gnc_dispatches_constancy_assumption(monkeypatch):
    """GNC should rebuild tensors with the selected data term."""
    calls = []
    monkeypatch.setattr(
        optical_flow, "get_motion_tensor_gc", _make_motion_tensor("gc", calls)
    )
    monkeypatch.setattr(
        optical_flow, "get_motion_tensor_gray", _make_motion_tensor("gray", calls)
    )
    monkeypatch.setattr(
        optical_flow, "get_motion_tensor_cs", _make_motion_tensor("cs", calls)
    )

    fixed = np.zeros((16, 16), dtype=np.float64)
    moving = np.zeros((16, 16), dtype=np.float64)

    optical_flow.get_displacement(
        fixed,
        moving,
        levels=1,
        min_level=0,
        iterations=1,
        level_solver_backend=_zero_level_solver,
        const_assumption="census",
        gnc_schedule=(0.0, 1.0),
        warping_steps=1,
    )

    assert calls
    assert set(calls) == {"cs"}
