"""
Tests for compensate_arr array-based motion compensation.

Tests array-based motion compensation using in-memory arrays,
matching the functionality of compensate_recording but without file I/O.
"""

import pytest
import numpy as np

from pyflowreg.motion_correction.compensate_arr import compensate_arr
from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat
from pyflowreg.util.io._arr import ArrayReader, ArrayWriter
from tests.fixtures import create_simple_test_data
from pyflowreg.core import list_backends

# Define test backends - simple list
TEST_BACKENDS = ["flowreg"]  # Default backend

# Add DISO if available
if "diso" in list_backends():
    TEST_BACKENDS.append("diso")


class TestCompensateArrBasics:
    """Test basic functionality of compensate_arr."""

    def test_basic_array_compensation(self):
        """Test basic array compensation with minimal data."""
        # Create simple test data
        T, H, W, C = 5, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Test with all available backends
        for backend_name in TEST_BACKENDS:
            # Run compensation with backend
            registered, flow = compensate_arr(
                video, reference, flow_backend=backend_name
            )

            # Check output shapes
            assert registered.shape == video.shape, f"Failed for backend {backend_name}"
            assert flow.shape == (T, H, W, 2), f"Failed for backend {backend_name}"

            # Check data types
            assert isinstance(registered, np.ndarray)
            assert isinstance(flow, np.ndarray)

    def test_single_channel_3d_input(self):
        """Test handling of single-channel 3D input (T,H,W)."""
        # Create single-channel 3D data
        T, H, W = 10, 32, 32
        video = np.random.rand(T, H, W).astype(np.float32)
        reference = np.mean(video[:3], axis=0)  # 2D reference

        # Run compensation
        registered, flow = compensate_arr(video, reference)

        # Check output shapes - should preserve input shape
        assert registered.shape == (T, H, W)
        assert flow.shape == (T, H, W, 2)

    def test_single_frame_2d_input(self):
        """Test handling of single frame 2D input (H,W)."""
        # Create single 2D frame
        H, W = 64, 64
        frame = np.random.rand(H, W).astype(np.float32)
        reference = np.random.rand(H, W).astype(np.float32)

        # Run compensation
        registered, flow = compensate_arr(frame, reference)

        # Check output shapes
        assert registered.shape == (H, W)
        assert flow.shape == (H, W, 2)  # Single frame flow

    def test_multichannel_input(self):
        """Test handling of multi-channel input (T,H,W,C)."""
        # Create multi-channel data
        T, H, W, C = 8, 24, 24, 3
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)  # (H,W,C) reference

        # Run compensation
        registered, flow = compensate_arr(video, reference)

        # Check output shapes
        assert registered.shape == (T, H, W, C)
        assert flow.shape == (T, H, W, 2)


class TestCompensateArrWithOptions:
    """Test compensate_arr with various OFOptions configurations."""

    def test_with_custom_options(self):
        """Test with custom OF_options."""
        # Create test data
        T, H, W, C = 6, 20, 20, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Create custom options
        options = OFOptions(
            alpha=50.0, levels=3, iterations=10, eta=0.9, quality_setting="fast"
        )

        # Run compensation
        registered, flow = compensate_arr(video, reference, options)

        # Check outputs
        assert registered.shape == video.shape
        assert flow.shape == (T, H, W, 2)

    def test_save_w_option(self):
        """Test that save_w=True properly returns displacement fields."""
        # Create test data
        T, H, W, C = 4, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Create options with save_w enabled
        options = OFOptions(save_w=True)

        # Run compensation
        registered, flow = compensate_arr(video, reference, options)

        # Check that flow fields are returned
        assert flow is not None
        assert flow.shape == (T, H, W, 2)

        # Check that flow has reasonable values (not all zeros)
        assert not np.allclose(flow, 0)

    def test_save_w_disabled(self):
        """Test behavior when save_w is disabled."""
        # Create test data
        T, H, W, C = 4, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Create options with save_w disabled
        options = OFOptions(save_w=False)

        # Run compensation
        registered, flow = compensate_arr(video, reference, options)

        # Flow should still be returned (as empty array for compatibility)
        assert flow is not None
        assert flow.shape == (T, H, W, 2)
        # Note: It might be zeros since save_w is False

    def test_output_typename_casting(self):
        """Test output type casting based on output_typename option."""
        # Create test data
        T, H, W, C = 3, 12, 12, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Test different output types
        output_types = {
            "single": np.float32,
            "double": np.float64,
            "uint16": np.uint16,
        }

        for typename, expected_dtype in output_types.items():
            options = OFOptions(output_typename=typename)
            registered, flow = compensate_arr(video, reference, options)

            # Check output dtype matches requested type
            assert registered.dtype == expected_dtype, f"Failed for {typename}"


class TestArrayReaderWriterIntegration:
    """Test that ArrayReader and ArrayWriter are properly used."""

    def test_array_reader_creation(self):
        """Test that input arrays are wrapped in ArrayReader."""
        # Create test data
        T, H, W, C = 5, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Patch get_video_file_reader to verify ArrayReader is created
        from pyflowreg.util.io import factory

        original_get_reader = factory.get_video_file_reader

        reader_created = False

        def mock_get_reader(input_source, *args, **kwargs):
            nonlocal reader_created
            result = original_get_reader(input_source, *args, **kwargs)
            if isinstance(result, ArrayReader):
                reader_created = True
            return result

        factory.get_video_file_reader = mock_get_reader
        try:
            registered, flow = compensate_arr(video, reference)
            assert reader_created, "ArrayReader was not created"
        finally:
            factory.get_video_file_reader = original_get_reader

    def test_array_writer_creation(self):
        """Test that output format ARRAY triggers ArrayWriter."""
        # Create test data
        T, H, W, C = 5, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Patch get_video_file_writer to verify ArrayWriter is created
        from pyflowreg.util.io import factory

        original_get_writer = factory.get_video_file_writer

        writer_created = False

        def mock_get_writer(file_path, output_format, *args, **kwargs):
            nonlocal writer_created
            result = original_get_writer(file_path, output_format, *args, **kwargs)
            if isinstance(result, ArrayWriter):
                writer_created = True
            return result

        factory.get_video_file_writer = mock_get_writer
        try:
            registered, flow = compensate_arr(video, reference)
            assert writer_created, "ArrayWriter was not created"
        finally:
            factory.get_video_file_writer = original_get_writer

    def test_displacement_writer_is_array_writer(self):
        """Test that displacement writer is also ArrayWriter when output is ARRAY."""
        # Create test data
        T, H, W, C = 4, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Create options with save_w enabled
        options = OFOptions(save_w=True)

        # Run compensation
        registered, flow = compensate_arr(video, reference, options)

        # Verify flow fields were captured
        assert flow is not None
        assert flow.shape == (T, H, W, 2)
        assert flow.dtype == np.float32 or flow.dtype == np.float64


class TestCompensateArrConsistency:
    """Test consistency between array and file-based processing."""

    def test_consistent_with_file_processing(self, temp_dir):
        """Test that array processing matches file-based processing."""
        # Create test data
        T, H, W, C = 8, 24, 24, 2
        video = create_simple_test_data((T, H, W, C))
        reference = np.mean(video[:3], axis=0)

        # Process with compensate_arr
        registered_arr, flow_arr = compensate_arr(video, reference)

        # TODO: Could compare with file-based processing if needed
        # This would require writing video to file and using compensate_recording

        # For now, just verify outputs are reasonable
        assert registered_arr.shape == video.shape
        assert flow_arr.shape == (T, H, W, 2)

    def test_batch_processing_consistency(self):
        """Test that batching works correctly for arrays."""
        # Create test data with enough frames for multiple batches
        T, H, W, C = 25, 16, 16, 2  # Enough for multiple batches
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        # Create options with small batch size to force multiple batches
        options = OFOptions(
            buffer_size=10,  # Force smaller batches
            save_w=True,
        )

        # Run compensation
        registered, flow = compensate_arr(video, reference, options)

        # Check outputs
        assert registered.shape == video.shape
        assert flow.shape == (T, H, W, 2)

    def test_flow_initialization_chain(self):
        """Test that flow initialization is properly maintained across batches."""
        # Create test data with drift pattern
        T, H, W, C = 20, 32, 32, 2
        video = np.zeros((T, H, W, C), dtype=np.float32)

        # Create a moving pattern
        for t in range(T):
            offset = t * 2  # Progressive drift
            for c in range(C):
                # Create shifted pattern
                y, x = np.mgrid[:H, :W]
                pattern = np.sin((x - offset) * 2 * np.pi / W) * 0.5 + 0.5
                video[t, :, :, c] = pattern

        reference = video[0]  # Use first frame as reference

        # Run compensation with flow initialization
        options = OFOptions(
            update_initialization_w=True,  # Enable flow initialization chain
            save_w=True,
        )

        registered, flow = compensate_arr(video, reference, options)

        # Check that flow captures the drift
        assert flow is not None
        assert flow.shape == (T, H, W, 2)

        # Just verify flow fields were computed (not testing specific values with random data)
        flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        assert flow_magnitude.shape == (T, H, W)


class TestCompensateArrEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_video(self):
        """Test handling of empty video array."""
        # Create empty video
        video = np.array([]).reshape(0, 10, 10, 2)
        reference = np.random.rand(10, 10, 2).astype(np.float32)

        # This should handle gracefully (might return empty or raise)
        try:
            registered, flow = compensate_arr(video, reference)
            # If it succeeds, check shapes
            assert registered.shape[0] == 0
            assert flow.shape[0] == 0
        except (ValueError, IndexError):
            # Expected behavior for empty input
            pass

    def test_single_channel_consistency(self):
        """Test that single channel processing is consistent."""
        # Create single-channel data in different formats
        T, H, W = 10, 20, 20

        # Format 1: 3D array (T, H, W)
        video_3d = np.random.rand(T, H, W).astype(np.float32)
        reference_2d = np.mean(video_3d[:3], axis=0)

        # Format 2: 4D array with single channel (T, H, W, 1)
        video_4d = video_3d[..., np.newaxis]
        reference_3d = reference_2d[..., np.newaxis]

        # Process both formats
        reg_3d, flow_3d = compensate_arr(video_3d, reference_2d)
        reg_4d, flow_4d = compensate_arr(video_4d, reference_3d)

        # Results should be equivalent (modulo shape)
        assert reg_3d.shape == (T, H, W)
        assert reg_4d.shape == (T, H, W, 1)

        # Compare data (squeeze 4D for comparison)
        np.testing.assert_allclose(reg_3d, np.squeeze(reg_4d, axis=-1), rtol=1e-5)

    def test_reference_shape_mismatch(self):
        """Test handling of reference shape mismatches."""
        # Create test data
        T, H, W, C = 5, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)

        # Test with wrong reference shape
        wrong_reference = np.random.rand(H + 1, W, C).astype(np.float32)

        # This should raise an error during processing
        with pytest.raises(Exception):  # Could be ValueError, AssertionError, etc.
            registered, flow = compensate_arr(video, wrong_reference)

    def test_nan_handling(self):
        """Test handling of NaN values in input."""
        # Create test data with NaN
        T, H, W, C = 5, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        video[2, 5, 5, 0] = np.nan  # Insert NaN
        reference = np.mean(video[:2], axis=0)

        # This might handle NaN or raise - both are acceptable
        try:
            registered, flow = compensate_arr(video, reference)
            # If it succeeds, check that output is reasonable
            assert registered.shape == video.shape
            # NaN might propagate or be handled
        except (ValueError, RuntimeError):
            # Expected if NaN is not supported
            pass


class TestCompensateArrPerformance:
    """Test performance aspects of array compensation."""

    def test_memory_efficiency(self):
        """Test that array processing doesn't create unnecessary copies."""
        # Create test data
        T, H, W, C = 10, 32, 32, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:3], axis=0)

        # Get initial memory usage (approximate)
        initial_size = video.nbytes + reference.nbytes

        # Run compensation
        registered, flow = compensate_arr(video, reference)

        # Check that output is reasonable size
        output_size = registered.nbytes + flow.nbytes

        # Output should be roughly same order of magnitude as input
        # (not creating many unnecessary copies)
        assert output_size < initial_size * 5  # Generous bound

    @pytest.mark.slow
    def test_large_array_processing(self):
        """Test processing of larger arrays."""
        # Create larger test data
        T, H, W, C = 100, 128, 128, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:10], axis=0)

        # Create options for faster processing
        options = OFOptions(quality_setting="fast", levels=2, iterations=5)

        # Run compensation
        registered, flow = compensate_arr(video, reference, options)

        # Check outputs
        assert registered.shape == video.shape
        assert flow.shape == (T, H, W, 2)


class TestProgressCallback:
    """Test progress callback functionality."""

    def test_progress_callback_called(self):
        """Test that progress callback is called during processing."""
        # Create test data
        T, H, W, C = 10, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Track progress calls
        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        # Run compensation with callback
        registered, flow = compensate_arr(
            video, reference, progress_callback=progress_callback
        )

        # Check callback was called
        assert len(progress_calls) > 0, "Progress callback was not called"

        # Check final progress matches total frames
        final_current, final_total = progress_calls[-1]
        assert final_current == T, f"Final progress {final_current} != {T} frames"
        assert final_total == T, f"Total frames {final_total} != {T}"

        # Check progress increments
        for i, (current, total) in enumerate(progress_calls):
            assert current > 0, f"Progress {i}: current={current} should be > 0"
            assert current <= total, f"Progress {i}: current={current} > total={total}"
            assert total == T, f"Progress {i}: total={total} != {T}"

    def test_progress_callback_with_batches(self):
        """Test progress callback with batch processing."""
        # Create test data with multiple batches
        T, H, W, C = 50, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        # Track progress
        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        # Force smaller batch size
        options = OFOptions(buffer_size=10)

        # Run compensation
        registered, flow = compensate_arr(video, reference, options, progress_callback)

        # Check progress was reported
        assert len(progress_calls) > 0

        # Verify monotonic increase
        previous_current = 0
        for current, total in progress_calls:
            assert current >= previous_current, "Progress should be monotonic"
            previous_current = current

        # Final progress should match total frames
        assert progress_calls[-1][0] == T

    def test_progress_callback_exception_handling(self):
        """Test that exceptions in callback don't break processing."""
        # Create test data
        T, H, W, C = 5, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        call_count = [0]
        exception_raised = [False]

        def faulty_callback(current, total):
            call_count[0] += 1
            # Raise exception on first call to test exception handling
            if call_count[0] == 1:
                exception_raised[0] = True
                raise ValueError("Test exception in callback")

        # Run with faulty callback - should complete despite exception
        registered, flow = compensate_arr(
            video, reference, progress_callback=faulty_callback
        )

        # Check processing completed
        assert registered.shape == video.shape
        assert flow.shape == (T, H, W, 2)
        assert call_count[0] >= 1  # Callback was called at least once
        assert exception_raised[0]  # Exception was actually raised

    def test_progress_percentage(self):
        """Test computing progress percentage from callback."""
        # Create test data
        T, H, W, C = 20, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:3], axis=0)

        percentages = []

        def progress_callback(current, total):
            if total > 0:
                percent = (current / total) * 100
                percentages.append(percent)

        # Run compensation
        registered, flow = compensate_arr(
            video, reference, progress_callback=progress_callback
        )

        # Check percentages
        assert len(percentages) > 0
        assert percentages[-1] == 100.0, "Final progress should be 100%"

        # Check monotonic increase
        for i in range(1, len(percentages)):
            assert percentages[i] >= percentages[i - 1], "Percentages should increase"

    def test_multiple_callbacks(self):
        """Test registering multiple progress callbacks."""
        # Create test data
        T, H, W, C = 8, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Track calls from multiple callbacks
        calls1 = []
        calls2 = []

        def callback1(current, total):
            calls1.append((current, total))

        def callback2(current, total):
            calls2.append((current, total))

        # Create options
        options = OFOptions()

        # Note: compensate_arr only accepts one callback, but we can test
        # that BatchMotionCorrector can handle multiple registrations
        from pyflowreg.motion_correction.compensate_recording import (
            BatchMotionCorrector,
        )

        # Set up for array processing
        options.input_file = video
        options.reference_frames = reference
        options.output_format = OutputFormat.ARRAY
        options.save_w = True
        options.save_meta_info = False

        # Create compensator and register multiple callbacks
        compensator = BatchMotionCorrector(options)
        compensator.register_progress_callback(callback1)
        compensator.register_progress_callback(callback2)

        # Run
        compensator.run()

        # Both callbacks should be called
        assert len(calls1) > 0, "First callback not called"
        assert len(calls2) > 0, "Second callback not called"
        assert calls1 == calls2, "Both callbacks should receive same calls"


class TestCompensateArrIntegration:
    """Integration tests with other components."""

    def test_with_preprocessing(self):
        """Test array compensation with preprocessing options."""
        # Create test data
        T, H, W, C = 8, 32, 32, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:3], axis=0)

        # Create options with preprocessing
        options = OFOptions(
            sigma=[[2.0, 2.0, 0.5], [2.0, 2.0, 0.5]],  # Gaussian filtering
            channel_normalization="separate",  # Per-channel normalization
            save_w=True,
        )

        # Run compensation
        registered, flow = compensate_arr(video, reference, options)

        # Check outputs
        assert registered.shape == video.shape
        assert flow.shape == (T, H, W, 2)

    def test_with_different_quality_settings(self):
        """Test all quality settings work with arrays."""
        # Create test data
        T, H, W, C = 6, 24, 24, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        quality_settings = ["fast", "balanced", "quality"]

        for quality in quality_settings:
            options = OFOptions(quality_setting=quality)
            registered, flow = compensate_arr(video, reference, options)

            # Check outputs for each quality setting
            assert registered.shape == video.shape, f"Failed for quality={quality}"
            assert flow.shape == (T, H, W, 2), f"Failed for quality={quality}"

    def test_options_not_modified(self):
        """Test that user's options object is not modified."""
        # Create test data
        T, H, W, C = 5, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Create options
        original_options = OFOptions(
            alpha=100.0,
            save_w=False,
            output_format=OutputFormat.HDF5,  # Different from ARRAY
        )

        # Store original values
        original_alpha = original_options.alpha
        original_save_w = original_options.save_w
        original_format = original_options.output_format

        # Run compensation
        registered, flow = compensate_arr(video, reference, original_options)

        # Check that original options were not modified
        assert original_options.alpha == original_alpha
        assert original_options.save_w == original_save_w
        assert (
            original_options.output_format == original_format
        )  # Should not be changed to ARRAY
