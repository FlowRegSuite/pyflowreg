"""
Tests for the core optical flow pyramid and motion tensor functions.
"""

import numpy as np

from pyflowreg.core.optical_flow import get_displacement, get_motion_tensor_gc


def _make_recording_zero_solver(records):
    """Level solver stub that records grid spacings and returns zero flow.

    Matches the positional level_solver signature: args 7/8 are u/v
    (with boundary padding) and args 15/16 are hx/hy.
    """

    def solver(*args):
        u, v = args[7], args[8]
        records.append(
            {
                "rows": u.shape[0] - 2,
                "cols": u.shape[1] - 2,
                "hx": args[15],
                "hy": args[16],
            }
        )
        return np.zeros_like(u), np.zeros_like(v)

    return solver


class TestPyramidGridSpacing:
    """Test the geometric binding of per-level grid spacings."""

    def test_get_displacement_passes_geometric_spacings_to_solver(self):
        """The solver's hx scales column (x) differences, so it must receive
        the column spacing n/cols, and hy the row spacing m/rows. A swapped
        binding diverges from this on non-square images where the per-axis
        pyramid depths and rounding differ."""
        rng = np.random.default_rng(seed=7)
        m, n = 32, 96
        fixed = rng.random((m, n))
        moving = rng.random((m, n))

        records = []
        flow = get_displacement(
            fixed,
            moving,
            iterations=1,
            update_lag=1,
            level_solver_backend=_make_recording_zero_solver(records),
        )

        assert flow.shape == (m, n, 2)
        assert len(records) > 1
        for rec in records:
            np.testing.assert_allclose(rec["hx"], n / rec["cols"], rtol=1e-12)
            np.testing.assert_allclose(rec["hy"], m / rec["rows"], rtol=1e-12)
        # Sanity: the elongated input makes row and column spacings genuinely
        # differ at coarse levels, so the assertions above are discriminative.
        assert any(not np.isclose(r["hx"], r["hy"]) for r in records)


class TestMotionTensorGC:
    """Test the gradient-constancy motion tensor on anisotropic grids."""

    def test_get_motion_tensor_gc_second_derivative_axis_binding(self):
        """fxx must be divided by the column spacing hx, not the row spacing.

        For f = 1e-3 * x^2 on a grid with hx=2: fxx = 2e-3 and
        fxy = fyy = ft = 0, so J11 = fxx^2 / (fxx^2 + 1e-6) = 0.8 at interior
        pixels. A swapped binding (fxx divided by hy^2 = 1) yields fxx = 8e-3
        and J11 close to 0.985 instead.
        """
        hy, hx = 1.0, 2.0
        H, W = 17, 23
        x = np.arange(W, dtype=np.float64) * hx
        f = np.tile(1e-3 * x**2, (H, 1))

        J11, J22, J33, J12, J13, J23 = get_motion_tensor_gc(f, f, hy, hx)

        center = (H // 2, W // 2)
        np.testing.assert_allclose(J11[center], 0.8, atol=0.01)
        # Identical frames: temporal terms vanish.
        np.testing.assert_allclose(J33[center], 0.0, atol=1e-12)
        np.testing.assert_allclose(J13[center], 0.0, atol=1e-12)
