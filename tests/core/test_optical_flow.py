"""Tests for core optical-flow tensor helpers."""

import inspect

import numpy as np

from pyflowreg.core.optical_flow import get_motion_tensor_cs


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
