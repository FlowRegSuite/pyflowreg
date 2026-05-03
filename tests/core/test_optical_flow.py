"""Tests for optical flow stage dispatch, GNC plumbing, and tensor helpers."""

import inspect

import numpy as np

import pyflowreg.core.optical_flow as optical_flow
from pyflowreg.core.optical_flow import get_displacement, get_motion_tensor_cs


def _compress_runs(values):
    """Collapse consecutive duplicate values while preserving order."""
    compressed = []
    for value in values:
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    return compressed


def _make_zero_level_solver(sink=None):
    """Create a fake level solver that records GNC stages."""

    def fake_level_solver(*args):
        u = args[7]
        v = args[8]
        gnc_beta = args[-1]
        if sink is not None:
            sink.append(gnc_beta)
        return np.zeros_like(u), np.zeros_like(v)

    return fake_level_solver


def _make_tensor_stub(counter=None):
    """Return zero motion tensors with the expected padded shape."""

    def fake_get_motion_tensor_gc(f1, f2, hy, hx):
        if counter is not None:
            counter.append((f1.shape, f2.shape, hy, hx))
        shape = (f1.shape[0] + 2, f1.shape[1] + 2)
        return tuple(np.zeros(shape, dtype=np.float64) for _ in range(6))

    return fake_get_motion_tensor_gc


def _sample_images(shape=(8, 9)):
    y = np.linspace(0.0, 1.0, shape[0])[:, np.newaxis]
    x = np.linspace(0.0, 1.0, shape[1])[np.newaxis, :]
    f1 = 0.25 + 0.35 * x + 0.20 * y
    f2 = 0.30 + 0.25 * x * x + 0.15 * y
    return f1.astype(np.float64), f2.astype(np.float64)


def _assert_zero_border(tensors):
    for tensor in tensors:
        assert np.array_equal(tensor[0, :], np.zeros_like(tensor[0, :]))
        assert np.array_equal(tensor[-1, :], np.zeros_like(tensor[-1, :]))
        assert np.array_equal(tensor[:, 0], np.zeros_like(tensor[:, 0]))
        assert np.array_equal(tensor[:, -1], np.zeros_like(tensor[:, -1]))


def test_get_displacement_without_gnc_uses_baseline_stage():
    """Test the default path does not pass a GNC stage to the solver."""
    fixed = np.zeros((12, 12), dtype=np.float64)
    moving = np.zeros((12, 12), dtype=np.float64)
    seen_betas = []

    flow = get_displacement(
        fixed,
        moving,
        levels=0,
        iterations=2,
        update_lag=1,
        level_solver_backend=_make_zero_level_solver(seen_betas),
    )

    assert flow.shape == (12, 12, 2)
    assert _compress_runs(seen_betas) == [None]


def test_get_displacement_without_gnc_ignores_warping_steps():
    """Test warping_steps alone does not change the default path."""
    fixed = np.zeros((12, 12), dtype=np.float64)
    moving = np.zeros((12, 12), dtype=np.float64)
    seen_betas = []

    flow = get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=_make_zero_level_solver(seen_betas),
        warping_steps=3,
    )

    assert flow.shape == (12, 12, 2)
    assert _compress_runs(seen_betas) == [None]


def test_get_displacement_with_gnc_repeats_pyramid_per_stage():
    """Test GNC reruns the pyramid with one fixed stage weight per pass."""
    fixed = np.zeros((12, 12), dtype=np.float64)
    moving = np.zeros((12, 12), dtype=np.float64)
    seen_betas = []

    flow = get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=_make_zero_level_solver(seen_betas),
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

    get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=_make_zero_level_solver(base_seen_betas),
        gnc_schedule=(0.0, 0.5, 1.0),
        warping_steps=1,
    )
    get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=_make_zero_level_solver(warp_seen_betas),
        gnc_schedule=(0.0, 0.5, 1.0),
        warping_steps=3,
    )

    assert _compress_runs(base_seen_betas) == [0.0, 0.5, 1.0]
    for beta in (0.0, 0.5, 1.0):
        assert warp_seen_betas.count(beta) == 3 * base_seen_betas.count(beta)


def test_get_displacement_with_gnc_default_warping_steps_is_ten():
    """Test omitted GNC warping steps use Sun-style ten warps per level."""
    fixed = np.zeros((12, 12), dtype=np.float64)
    moving = np.zeros((12, 12), dtype=np.float64)
    base_seen_betas = []
    seen_betas = []

    get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=_make_zero_level_solver(base_seen_betas),
        gnc_schedule=(0.0, 1.0),
        warping_steps=1,
    )
    flow = get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=_make_zero_level_solver(seen_betas),
        gnc_schedule=(0.0, 1.0),
    )

    assert flow.shape == (12, 12, 2)
    assert seen_betas.count(0.0) == 10 * base_seen_betas.count(0.0)
    assert seen_betas.count(1.0) == 10 * base_seen_betas.count(1.0)


def test_get_displacement_with_gnc_rebuilds_tensors_each_warp(monkeypatch):
    """Test each GNC warp rewarps images and rebuilds motion tensors."""
    fixed = np.zeros((12, 12, 2), dtype=np.float64)
    moving = np.zeros((12, 12, 2), dtype=np.float64)
    warp_calls = []
    tensor_calls = []
    seen_betas = []

    def fake_imregister_wrapper(f2_level, u, v, f1_level):
        warp_calls.append((f2_level.shape, u.shape, v.shape, f1_level.shape))
        return f2_level

    monkeypatch.setattr(optical_flow, "imregister_wrapper", fake_imregister_wrapper)
    monkeypatch.setattr(
        optical_flow,
        "get_motion_tensor_gc",
        _make_tensor_stub(tensor_calls),
    )

    flow = optical_flow.get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=_make_zero_level_solver(seen_betas),
        gnc_schedule=(0.0, 1.0),
        warping_steps=2,
    )

    assert flow.shape == (12, 12, 2)
    assert len(warp_calls) == len(seen_betas)
    assert len(tensor_calls) == 2 * len(warp_calls)


def test_get_displacement_with_gnc_median_filters_after_each_warp(monkeypatch):
    """Test GNC applies the median filter after each warp-level solve."""
    fixed = np.zeros((12, 12), dtype=np.float64)
    moving = np.zeros((12, 12), dtype=np.float64)
    median_calls = []
    seen_betas = []

    def fake_median_filter(arr, size, mode):
        median_calls.append((arr.shape, size, mode))
        return arr

    monkeypatch.setattr(optical_flow, "median_filter", fake_median_filter)
    monkeypatch.setattr(optical_flow, "get_motion_tensor_gc", _make_tensor_stub())

    flow = optical_flow.get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=_make_zero_level_solver(seen_betas),
        gnc_schedule=(0.0, 1.0),
        warping_steps=3,
    )

    assert flow.shape == (12, 12, 2)
    assert len(median_calls) == 2 * len(seen_betas)
    assert {(size, mode) for _, size, mode in median_calls} == {((5, 5), "mirror")}
    assert all(min(shape) > 5 for shape, _, _ in median_calls)


def test_get_displacement_with_gnc_carries_flow_between_stages():
    """Test each GNC stage starts from the previous stage displacement."""
    fixed = np.zeros((12, 12), dtype=np.float64)
    moving = np.zeros((12, 12), dtype=np.float64)
    stage_initial_means = []

    def fake_level_solver(*args):
        u = args[7]
        v = args[8]
        gnc_beta = args[-1]
        stage_initial_means.append((gnc_beta, float(u[1:-1, 1:-1].mean())))
        du = np.full_like(u, gnc_beta + 1.0)
        return du, np.zeros_like(v)

    flow = get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        level_solver_backend=fake_level_solver,
        gnc_schedule=(0.0, 1.0),
        warping_steps=1,
    )

    assert flow.shape == (12, 12, 2)
    beta0_means = [mean for beta, mean in stage_initial_means if beta == 0.0]
    beta1_means = [mean for beta, mean in stage_initial_means if beta == 1.0]
    assert beta0_means[0] == 0.0
    np.testing.assert_allclose(beta0_means[-1], len(beta0_means) - 1, atol=1e-6)
    np.testing.assert_allclose(beta1_means[0], len(beta0_means), atol=1e-6)


def test_level_solver_dispatches_to_default_solver_without_gnc(monkeypatch):
    """Test the CPU wrapper keeps the default solver branch separate."""
    shape = (3, 3, 1)
    field_shape = shape[:2]
    seen = []

    def fake_default(*args, **kwargs):
        seen.append(("default", kwargs))
        return np.zeros((*field_shape, 2), dtype=np.float64)

    def fake_gnc(*args, **kwargs):
        seen.append(("gnc", kwargs))
        return np.zeros((*field_shape, 2), dtype=np.float64)

    monkeypatch.setattr(optical_flow, "compute_flow", fake_default)
    monkeypatch.setattr(optical_flow, "compute_flow_gnc", fake_gnc)

    optical_flow.level_solver(
        *[np.zeros(shape, dtype=np.float64) for _ in range(6)],
        np.ones(shape, dtype=np.float64),
        np.zeros(field_shape, dtype=np.float64),
        np.zeros(field_shape, dtype=np.float64),
        (1.0, 1.0),
        1,
        1,
        0,
        np.array([0.45], dtype=np.float64),
        1.0,
        1.0,
        1.0,
    )

    assert len(seen) == 1
    assert seen[0][0] == "default"
    assert "gnc_beta" not in seen[0][1]


def test_level_solver_dispatches_to_gnc_solver_with_beta(monkeypatch):
    """Test the CPU wrapper only enters the GNC solver when requested."""
    shape = (3, 3, 1)
    field_shape = shape[:2]
    seen = []

    def fake_default(*args, **kwargs):
        seen.append(("default", kwargs))
        return np.zeros((*field_shape, 2), dtype=np.float64)

    def fake_gnc(*args, **kwargs):
        seen.append(("gnc", kwargs))
        return np.zeros((*field_shape, 2), dtype=np.float64)

    monkeypatch.setattr(optical_flow, "compute_flow", fake_default)
    monkeypatch.setattr(optical_flow, "compute_flow_gnc", fake_gnc)

    optical_flow.level_solver(
        *[np.zeros(shape, dtype=np.float64) for _ in range(6)],
        np.ones(shape, dtype=np.float64),
        np.zeros(field_shape, dtype=np.float64),
        np.zeros(field_shape, dtype=np.float64),
        (1.0, 1.0),
        1,
        1,
        0,
        np.array([0.45], dtype=np.float64),
        1.0,
        1.0,
        1.0,
        gnc_beta=0.5,
    )

    assert len(seen) == 1
    assert seen[0][0] == "gnc"
    assert seen[0][1]["gnc_beta"] == 0.5


def test_get_motion_tensor_cs_shape_and_zero_border():
    """Returned tensors match the solver contract."""
    f1, f2 = _sample_images()

    tensors = get_motion_tensor_cs(f1, f2, hy=1.0, hx=1.0)

    assert len(tensors) == 6
    for tensor in tensors:
        assert tensor.shape == (f1.shape[0] + 2, f1.shape[1] + 2)
    _assert_zero_border(tensors)


def test_get_motion_tensor_cs_constant_images_near_zero():
    """Constant frames should not create census tensor energy."""
    f1 = np.full((7, 6), 0.25, dtype=np.float64)
    f2 = np.full((7, 6), 0.75, dtype=np.float64)

    tensors = get_motion_tensor_cs(f1, f2, hy=1.0, hx=1.0)

    for tensor in tensors:
        np.testing.assert_allclose(tensor, 0.0, atol=1e-14)


def test_get_motion_tensor_cs_additive_shift_invariance():
    """Neighbor-center differences should cancel common additive offsets."""
    f1, f2 = _sample_images()

    base = get_motion_tensor_cs(f1, f2, hy=1.0, hx=1.0)
    shifted = get_motion_tensor_cs(f1 + 4.0, f2 + 4.0, hy=1.0, hx=1.0)

    for base_tensor, shifted_tensor in zip(base, shifted):
        np.testing.assert_allclose(shifted_tensor, base_tensor, atol=1e-12)


def test_get_motion_tensor_cs_default_eps_matches_normalized_convention():
    """Default epsilon should be the normalized 0.1 / 255.0 value."""
    f1, f2 = _sample_images()

    default = get_motion_tensor_cs(f1, f2, hy=1.0, hx=1.0)
    explicit = get_motion_tensor_cs(f1, f2, hy=1.0, hx=1.0, eps=0.1 / 255.0)

    for default_tensor, explicit_tensor in zip(default, explicit):
        np.testing.assert_allclose(default_tensor, explicit_tensor)


def test_get_motion_tensor_cs_does_not_use_np_roll():
    """Neighbor access should not use cyclic shifts."""
    source = inspect.getsource(get_motion_tensor_cs)

    assert "np.roll" not in source


def test_get_motion_tensor_cs_anisotropic_spacing():
    """Anisotropic spacing should preserve tensor shape and zero borders."""
    f1, f2 = _sample_images(shape=(6, 10))

    tensors = get_motion_tensor_cs(f1, f2, hy=0.5, hx=2.0)

    for tensor in tensors:
        assert tensor.shape == (f1.shape[0] + 2, f1.shape[1] + 2)
        assert np.all(np.isfinite(tensor))
    _assert_zero_border(tensors)
