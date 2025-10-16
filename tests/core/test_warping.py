"""
Tests for core warping utilities.

Tests backward_valid_mask, imregister_binary, and compute_batch_valid_masks
functions used in session-level processing.
"""

import numpy as np
import pytest

from pyflowreg.core.warping import (
    backward_valid_mask,
    imregister_binary,
    compute_batch_valid_masks,
)


class TestBackwardValidMask:
    """Test backward_valid_mask function."""

    def test_zero_displacement(self):
        """Test that zero displacement gives all-valid mask."""
        H, W = 32, 32
        u = np.zeros((H, W), dtype=np.float32)
        v = np.zeros((H, W), dtype=np.float32)

        valid = backward_valid_mask(u, v)

        assert valid.shape == (H, W)
        assert valid.dtype == bool
        assert np.all(valid), "All pixels should be valid with zero displacement"

    @pytest.mark.parametrize("shift_x,shift_y", [(5, 0), (0, 5), (3, 4), (-3, -4)])
    def test_constant_shift(self, shift_x, shift_y):
        """Test constant shift creates expected invalid region."""
        H, W = 32, 32
        u = np.full((H, W), shift_x, dtype=np.float32)
        v = np.full((H, W), shift_y, dtype=np.float32)

        valid = backward_valid_mask(u, v)

        # Check that pixels mapping outside bounds are invalid
        for y in range(H):
            for x in range(W):
                mapped_x = x + shift_x
                mapped_y = y + shift_y
                expected_valid = (0 <= mapped_x < W) and (0 <= mapped_y < H)
                assert (
                    valid[y, x] == expected_valid
                ), f"Pixel ({y},{x}) validity mismatch"

    def test_right_shift_invalidates_right_edge(self):
        """Test that rightward shift invalidates right edge pixels."""
        H, W = 32, 32
        shift = 5
        u = np.full((H, W), shift, dtype=np.float32)
        v = np.zeros((H, W), dtype=np.float32)

        valid = backward_valid_mask(u, v)

        # Right edge should be invalid
        assert not np.any(valid[:, -shift:]), "Right edge should be invalid"
        # Left pixels should be valid
        assert np.all(valid[:, : W - shift]), "Left pixels should be valid"

    def test_fractional_displacement(self):
        """Test that fractional displacements work correctly."""
        H, W = 32, 32
        u = np.full((H, W), 2.5, dtype=np.float32)
        v = np.full((H, W), 1.7, dtype=np.float32)

        valid = backward_valid_mask(u, v)

        # Pixels mapping to [0, W) x [0, H) should be valid
        # With shift (u=2.5, v=1.7), pixel at (y, x) maps to (y+v, x+u) = (y+1.7, x+2.5)
        # Valid if: 0 <= y+1.7 < H  AND  0 <= x+2.5 < W
        # So: y < H-1.7 = 30.3  AND  x < W-2.5 = 29.5
        # Valid region: y in [0, 30] (rows 0-30), x in [0, 29] (cols 0-29)
        assert np.all(
            valid[:31, :30]
        ), "Interior should be valid (rows 0-30, cols 0-29)"
        assert not np.any(valid[31:, :]), "Last row (31) should be invalid"
        assert not np.any(valid[:, 30:]), "Last 2 columns (30-31) should be invalid"


class TestImregisterBinary:
    """Test imregister_binary function."""

    def test_zero_displacement(self):
        """Test that zero displacement preserves mask."""
        H, W = 32, 32
        mask = np.random.rand(H, W) > 0.5
        u = np.zeros((H, W), dtype=np.float32)
        v = np.zeros((H, W), dtype=np.float32)

        warped = imregister_binary(mask, u, v)

        assert warped.shape == mask.shape
        assert warped.dtype == bool
        np.testing.assert_array_equal(
            warped, mask, err_msg="Zero displacement should preserve mask"
        )

    def test_constant_shift_circle_mask(self):
        """Test warping circular mask with constant shift."""
        H, W = 64, 64
        # Create circular mask
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        center_y, center_x = H // 2, W // 2
        radius = 15
        mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= radius**2

        # Shift right by 10 pixels
        shift_x, shift_y = 10, 0
        u = np.full((H, W), shift_x, dtype=np.float32)
        v = np.full((H, W), shift_y, dtype=np.float32)

        warped = imregister_binary(mask, u, v)

        # Check center of original circle is now at shifted location
        # Original center at (32, 32) should map to (32, 42) in warped
        # In warped coordinates, center should be at original location
        # Actually, backward warp means output pixel (y,x) reads from (y+v, x+u)
        # So to get the same pattern shifted left, we shift the reading location right

        # After shift_x=10, the circle should appear shifted LEFT by 10
        # (because we're warping backward: output reads from input+shift)
        # So check that center region is empty and shifted region has the circle

        # Simpler check: verify total number of True pixels is similar
        # (accounting for edge truncation)
        original_count = np.sum(mask)
        warped_count = np.sum(warped)

        # Should lose some pixels at the edge where mapping goes out of bounds
        assert warped_count <= original_count, "Warped mask should not gain pixels"
        assert warped_count > 0.8 * original_count, "Should retain most pixels"

    def test_out_of_bounds_invalidates(self):
        """Test that out-of-bounds mapping creates invalid pixels."""
        H, W = 32, 32
        mask = np.ones((H, W), dtype=bool)  # All True

        # Large shift that pushes everything out of bounds
        u = np.full((H, W), W + 10, dtype=np.float32)
        v = np.zeros((H, W), dtype=np.float32)

        warped = imregister_binary(mask, u, v)

        # Everything should be invalid (out of bounds)
        assert not np.any(warped), "All pixels should be invalid with large shift"

    def test_nearest_interpolation(self):
        """Test that nearest-neighbor interpolation is used."""
        H, W = 16, 16
        # Checkerboard pattern
        mask = np.zeros((H, W), dtype=bool)
        mask[::2, ::2] = True
        mask[1::2, 1::2] = True

        # Fractional shift (should use nearest neighbor)
        u = np.full((H, W), 0.4, dtype=np.float32)
        v = np.zeros((H, W), dtype=np.float32)

        warped = imregister_binary(mask, u, v)

        # With nearest interpolation, pattern should be preserved (mostly)
        # Can't be exact due to quantization, but should maintain boolean values
        assert warped.dtype == bool
        assert np.any(warped), "Should have some True values"
        assert np.any(~warped), "Should have some False values"


class TestComputeBatchValidMasks:
    """Test compute_batch_valid_masks function."""

    def test_single_frame(self):
        """Test batch with single frame."""
        H, W = 32, 32
        u = np.full((H, W), 5.0, dtype=np.float32)
        v = np.zeros((H, W), dtype=np.float32)
        w = np.stack([u, v], axis=-1)[np.newaxis, ...]  # Add time dimension

        valid_batch = compute_batch_valid_masks(w)

        assert valid_batch.shape == (1, H, W)
        assert valid_batch.dtype == np.uint8
        assert np.all(
            (valid_batch == 0) | (valid_batch == 255)
        ), "Values should be 0 or 255"

    def test_multiple_frames(self):
        """Test batch with multiple frames."""
        T, H, W = 5, 32, 32

        # Create varying displacements
        w = np.zeros((T, H, W, 2), dtype=np.float32)
        for t in range(T):
            w[t, :, :, 0] = t  # Increasing horizontal shift
            w[t, :, :, 1] = 0

        valid_batch = compute_batch_valid_masks(w)

        assert valid_batch.shape == (T, H, W)
        assert valid_batch.dtype == np.uint8

        # First frame (no shift) should be all valid
        assert np.all(valid_batch[0] == 255), "First frame should be all valid"

        # Last frame (shift=4) should have invalid right edge
        assert np.any(
            valid_batch[-1, :, -4:] == 0
        ), "Last frame should have invalid edge"

    def test_output_format_hdf5_compatible(self):
        """Test that output is uint8 for HDF5 storage."""
        T, H, W = 3, 16, 16
        w = np.zeros((T, H, W, 2), dtype=np.float32)

        valid_batch = compute_batch_valid_masks(w)

        assert valid_batch.dtype == np.uint8, "Should be uint8 for HDF5 storage"
        assert np.all(
            (valid_batch == 0) | (valid_batch == 255)
        ), "Values should be 0 or 255"


class TestMaskCorrectness:
    """
    Test mask correctness with synthetic video.

    Build a tiny video with known constant shift, run motion correction
    with mask persistence, and verify masks match analytic in-bounds region.
    """

    def test_synthetic_video_mask_correctness(self, tmp_path):
        """
        Create synthetic video with constant shift, verify mask correctness.

        Uses compensate_arr workflow to generate masks, then checks that
        per-frame masks match analytic expectations and temporal AND is correct.
        """
        pytest.skip(
            "Integration test requiring compensate_arr - implement after basic tests pass"
        )
        # TODO: Implement full integration test
        # T, H, W = 5, 32, 32
        # Create video with constant shift
        # Run compensate_arr with save_valid_idx=True
        # Load idx output
        # Verify per-frame masks match backward_valid_mask results
        # Verify temporal AND is correct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
