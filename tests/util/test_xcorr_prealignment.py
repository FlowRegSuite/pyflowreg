"""
Tests for cross-correlation prealignment functionality.

Tests the xcorr_prealignment module for rigid displacement estimation
and its integration with motion compensation across different executors.
"""

import numpy as np
import pytest
from scipy.ndimage import shift as ndi_shift

from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.motion_correction.compensate_arr import compensate_arr
from pyflowreg.util.xcorr_prealignment import estimate_rigid_xcorr_2d


class TestEstimateRigidXcorr2D:
    """Test the estimate_rigid_xcorr_2d function."""

    def test_basic_rigid_shift_detection(self):
        """Test detection of simple rigid shifts."""
        # Create reference image
        H, W = 128, 128
        ref = np.zeros((H, W), dtype=np.float32)
        ref[40:60, 40:60] = 1.0  # Square feature
        ref[70:80, 70:80] = 0.8  # Small square

        # True displacement to apply to mov to align with ref (what we want to estimate)
        true_dx_dy = np.array([5.0, -3.0], dtype=np.float32)

        # Create moved image by shifting ref with the displacement
        # ndi_shift uses (dy, dx) order
        mov = ndi_shift(
            ref, shift=(true_dx_dy[1], true_dx_dy[0]), order=1, mode="constant"
        )

        # Estimate should return the displacement to apply to mov to align with ref
        estimated_shift = estimate_rigid_xcorr_2d(ref, mov, target_hw=(64, 64), up=1)

        # Check estimates match true displacement
        assert np.abs(estimated_shift[0] - true_dx_dy[0]) <= 1.0
        assert np.abs(estimated_shift[1] - true_dx_dy[1]) <= 1.0

    def test_subpixel_accuracy(self):
        """Test subpixel shift detection with upsampling."""
        # Create reference image with realistic features (not pure sinusoids)
        H, W = 128, 128
        ref = np.random.rand(H, W).astype(np.float32) * 0.1

        # Add Gaussian blobs for distinct features
        y, x = np.ogrid[:H, :W]
        ref += np.exp(-((y - 30) ** 2 + (x - 40) ** 2) / 50) * 2.0
        ref += np.exp(-((y - 80) ** 2 + (x - 90) ** 2) / 40) * 1.5
        ref += np.exp(-((y - 60) ** 2 + (x - 60) ** 2) / 60) * 1.8

        # Add some rectangles for sharp features
        ref[45:55, 70:85] = 1.2
        ref[90:100, 20:30] = 0.8

        # True displacement to apply to mov to align with ref
        true_dx_dy = np.array([2.3, -1.7], dtype=np.float32)

        # Create shifted version with subpixel shift
        # Use 'constant' mode to avoid wraparound issues
        mov = ndi_shift(
            ref, shift=(true_dx_dy[1], true_dx_dy[0]), order=3, mode="constant"
        )

        # Estimate with upsampling for subpixel accuracy
        estimated_shift = estimate_rigid_xcorr_2d(ref, mov, target_hw=(128, 128), up=10)

        # Check subpixel accuracy - should be quite accurate with realistic features
        assert np.abs(estimated_shift[0] - true_dx_dy[0]) < 0.5
        assert np.abs(estimated_shift[1] - true_dx_dy[1]) < 0.5

    def test_multichannel_input(self):
        """Test handling of multi-channel images."""
        # Create 2-channel reference with structured features
        H, W, C = 64, 64, 2
        ref = np.zeros((H, W, C), dtype=np.float32)

        # Add different features to each channel for better registration
        y, x = np.ogrid[:H, :W]
        ref[..., 0] = np.exp(-((y - 20) ** 2 + (x - 30) ** 2) / 50)
        ref[..., 1] = np.exp(-((y - 40) ** 2 + (x - 35) ** 2) / 40)
        ref[15:25, 40:50, 0] = 1.0
        ref[35:45, 10:20, 1] = 0.8
        ref += np.random.randn(H, W, C) * 0.05  # Small noise

        # True displacement to apply to mov to align with ref
        true_dx_dy = np.array([3.0, -2.0], dtype=np.float32)

        # Apply shift to create moving image
        mov = np.zeros_like(ref)
        for c in range(C):
            mov[..., c] = ndi_shift(
                ref[..., c],
                shift=(true_dx_dy[1], true_dx_dy[0]),
                order=1,
                mode="constant",
            )

        # Estimate shift - should handle multichannel
        estimated_shift = estimate_rigid_xcorr_2d(ref, mov, target_hw=(32, 32), up=1)

        # Check estimates (allow 1-pixel tolerance due to downsampling)
        assert np.abs(estimated_shift[0] - true_dx_dy[0]) <= 1.0
        assert np.abs(estimated_shift[1] - true_dx_dy[1]) <= 1.0

    def test_weighted_estimation(self):
        """Test weighted cross-correlation estimation."""
        # Create reference with non-uniform features
        H, W = 128, 128
        ref = np.zeros((H, W), dtype=np.float32)
        ref[30:50, 30:50] = 1.0  # Important region
        ref[80:90, 80:90] = 0.3  # Less important region

        # Create weight emphasizing important region
        weight = np.zeros((H, W), dtype=np.float32)
        weight[20:60, 20:60] = 1.0  # Focus on important region

        # True displacement to apply to mov to align with ref
        true_dx_dy = np.array([4.0, -3.0], dtype=np.float32)

        # Create shifted version
        mov = ndi_shift(
            ref, shift=(true_dx_dy[1], true_dx_dy[0]), order=1, mode="constant"
        )

        # Estimate with weight
        estimated_shift = estimate_rigid_xcorr_2d(
            ref, mov, weight=weight, target_hw=(64, 64), up=1
        )

        # Should still detect shift correctly
        assert np.abs(estimated_shift[0] - true_dx_dy[0]) <= 1.0
        assert np.abs(estimated_shift[1] - true_dx_dy[1]) <= 1.0


class TestCCPrealignmentIntegration:
    """Test CC prealignment integration with motion compensation."""

    @pytest.fixture
    def synthetic_rigid_data(self):
        """Create synthetic data with known rigid displacements."""
        T, H, W = 5, 128, 128

        # Create reference with structured features (similar to 3D version)
        reference = np.random.rand(H, W).astype(np.float32) * 0.2
        y, x = np.ogrid[:H, :W]
        # Add gaussian blob
        reference += (10 * np.exp(-((y - 64) ** 2 + (x - 64) ** 2) / 200)).astype(
            np.float32
        )
        # Add rectangles for better features
        reference[30:50, 30:50] = 5.0
        reference[70:90, 70:90] = 4.0
        reference[40:60, 80:100] = 3.0
        reference = reference.astype(np.float32)  # Ensure float32 after all operations

        # Generate frames with rigid shifts
        frames = np.zeros((T, H, W), dtype=np.float32)
        # Use larger, more detectable shifts
        true_shifts = np.array(
            [[10.0, -5.0], [-8.0, 7.0], [6.0, -4.0], [-5.0, -3.0], [4.0, 6.0]],
            dtype=np.float32,
        )

        for t in range(T):
            # Create moved frame by shifting reference
            frames[t] = ndi_shift(
                reference,
                shift=(true_shifts[t, 1], true_shifts[t, 0]),  # (dy, dx)
                order=1,
                mode="nearest",
            ).astype(np.float32)
            # Add small noise
            frames[t] += (np.random.randn(H, W) * 0.1).astype(np.float32)

        return reference, frames, true_shifts

    def test_cc_improves_registration(self, synthetic_rigid_data):
        """Test that CC prealignment improves registration quality."""
        reference, frames, _ = synthetic_rigid_data

        # Options with very few iterations (to make CC benefit more visible)
        options_no_cc = OFOptions(
            alpha=1.0,
            levels=5,  # Fewer levels
            eta=0.8,
            iterations=5,  # Very few iterations
            cc_initialization=False,
            quality_setting="fast",
        )

        # Options with CC prealignment
        options_with_cc = OFOptions(
            alpha=1.0,
            levels=5,  # Same levels
            eta=0.8,
            iterations=5,  # Same few iterations
            cc_initialization=True,
            cc_hw=64,
            cc_up=2,
            quality_setting="fast",
        )

        # Register without CC
        reg_no_cc, _ = compensate_arr(frames, reference, options_no_cc)
        error_no_cc = np.mean(np.abs(reg_no_cc - reference[np.newaxis, ...]))

        # Register with CC
        reg_with_cc, _ = compensate_arr(frames, reference, options_with_cc)
        error_with_cc = np.mean(np.abs(reg_with_cc - reference[np.newaxis, ...]))

        # CC should reduce error or at least be similar (with fewer iterations, CC helps more)
        # Allow for small numerical differences
        assert (
            error_with_cc <= error_no_cc * 1.05
        ), f"CC error {error_with_cc:.4f} should be <= no-CC error {error_no_cc:.4f}"
        # Final error should be reasonable
        assert error_with_cc < 0.5

    @pytest.mark.parametrize(
        "executor_name", ["sequential", "threading", "multiprocessing"]
    )
    def test_cc_with_different_executors(self, synthetic_rigid_data, executor_name):
        """Test CC prealignment works with each executor."""
        from pyflowreg.motion_correction.compensate_recording import (
            BatchMotionCorrector,
            RegistrationConfig,
        )
        from pyflowreg.motion_correction.OF_options import OutputFormat

        reference, frames, true_shifts = synthetic_rigid_data

        # Ensure arrays are contiguous for numba functions
        reference = np.ascontiguousarray(reference, dtype=np.float32)
        frames = np.ascontiguousarray(frames, dtype=np.float32)

        # Create options for this executor
        options = OFOptions(
            alpha=1.0,
            levels=10,
            eta=0.8,
            iterations=20,
            cc_initialization=True,
            cc_hw=64,
            cc_up=1,
            quality_setting="fast",
            input_file=frames,
            reference_frames=reference,
            output_format=OutputFormat.ARRAY,
            save_w=True,
        )

        # Create config for this executor
        config = RegistrationConfig(
            parallelization=executor_name,
            n_jobs=2 if executor_name != "sequential" else 1,
            verbose=False,
        )

        # Create pipeline and run
        pipeline = BatchMotionCorrector(options, config)
        pipeline.run()

        # Get results from ArrayWriter
        registered = pipeline.video_writer.get_array()
        flow = pipeline.w_writer.get_array()

        # Verify results shape - system normalizes to 4D internally
        # If input was 3D (T,H,W), output will be 4D (T,H,W,1)
        expected_shape = frames.shape if frames.ndim == 4 else (*frames.shape, 1)
        assert registered.shape == expected_shape
        assert flow.shape == (frames.shape[0], frames.shape[1], frames.shape[2], 2)

        # Check that registration improved alignment
        # Squeeze registered to match original dimensions for comparison
        registered_for_comparison = np.squeeze(registered)
        error_before = np.mean(np.abs(frames - reference))
        error_after = np.mean(np.abs(registered_for_comparison - reference))
        assert error_after < error_before * 0.5  # Should reduce error by at least 50%

    def test_cc_parameter_validation(self):
        """Test validation of CC-related parameters."""
        # cc_hw as integer
        options = OFOptions(cc_initialization=True, cc_hw=128)
        assert options.cc_hw == 128

        # cc_hw as tuple
        options = OFOptions(cc_initialization=True, cc_hw=(128, 256))
        assert options.cc_hw == (128, 256)

        # cc_up must be >= 1
        with pytest.raises(ValueError):
            OFOptions(cc_initialization=True, cc_up=0)

        # Default values
        options = OFOptions()
        assert options.cc_initialization is False
        assert options.cc_hw == 256
        assert options.cc_up == 1


if __name__ == "__main__":
    """Quick test of xcorr prealignment functionality."""
    print("Testing cross-correlation prealignment...")

    # Create simple test case
    H, W = 128, 128
    ref = np.zeros((H, W), dtype=np.float32)
    ref[40:60, 40:60] = 1.0

    # Apply known shift
    shift_x, shift_y = 5.0, -3.0
    mov = ndi_shift(ref, shift=(shift_y, shift_x), order=1, mode="constant")

    # Estimate shift
    estimated = estimate_rigid_xcorr_2d(ref, mov, target_hw=(64, 64), up=1)

    print(f"True shift: dx={shift_x:.1f}, dy={shift_y:.1f}")
    print(f"Estimated (negated): dx={estimated[0]:.1f}, dy={estimated[1]:.1f}")
    print(
        f"Error: dx={abs(estimated[0] - (-shift_x)):.2f}, dy={abs(estimated[1] - (-shift_y)):.2f}"
    )

    # Test with motion compensation
    print("\nTesting with motion compensation...")
    T = 3
    frames = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        shift = np.random.uniform(-5, 5, 2)
        frames[t] = ndi_shift(ref, shift=(shift[1], shift[0]), order=1, mode="constant")

    options = OFOptions(
        cc_initialization=True, cc_hw=64, cc_up=2, quality_setting="fast"
    )

    registered, _ = compensate_arr(frames, ref, options)
    error = np.mean(np.abs(registered - ref[np.newaxis, ...]))
    print(f"Mean registration error: {error:.4f}")

    if error < 0.15:
        print("✓ Test passed!")
    else:
        print("✗ Test failed - error too large")
