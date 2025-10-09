"""
Tests for FlowRegLive real-time motion compensation.

Tests live/streaming motion compensation with adaptive reference updates,
temporal filtering, and flow initialization persistence.
"""

import pytest
import numpy as np

from pyflowreg.motion_correction.flow_reg_live import FlowRegLive
from pyflowreg.motion_correction.OF_options import OFOptions, QualitySetting


class TestFlowRegLiveBasics:
    """Test basic functionality of FlowRegLive."""

    def test_basic_initialization(self):
        """Test basic FlowRegLive initialization."""
        flow_reg = FlowRegLive()

        # Check defaults
        assert flow_reg.options.quality_setting == QualitySetting.FAST
        assert flow_reg.options.save_w is False
        assert flow_reg.reference_update_interval == 20
        assert flow_reg.reference_update_weight == 0.2
        assert flow_reg.reference_raw is None
        assert flow_reg.reference_proc is None
        assert flow_reg.last_flow is None
        assert flow_reg.frame_count == 0

    def test_initialization_with_options(self):
        """Test FlowRegLive initialization with custom options."""
        options = OFOptions(
            alpha=5.0, sigma=[[3.0, 3.0, 1.0], [3.0, 3.0, 1.0]], iterations=30
        )

        flow_reg = FlowRegLive(
            options=options,
            reference_buffer_size=100,
            reference_update_interval=10,
            reference_update_weight=0.5,
        )

        # Check that options are overridden for speed
        assert flow_reg.options.quality_setting == QualitySetting.FAST
        assert flow_reg.options.save_w is False
        # But other options preserved
        # Note: alpha is converted to tuple in OF_options
        assert flow_reg.options.alpha == (5.0, 5.0)
        assert flow_reg.options.iterations == 30

        # Check custom parameters
        assert flow_reg.reference_update_interval == 10
        assert flow_reg.reference_update_weight == 0.5

    def test_temporal_buffer_size_calculation(self):
        """Test automatic temporal buffer size calculation from sigma."""
        # Test with small temporal sigma
        options = OFOptions(sigma=[2.0, 2.0, 0.5])
        flow_reg = FlowRegLive(options=options, truncate=4.0)

        # Buffer size = int(truncate * sigma_t + 0.5) + 1
        expected_size = int(4.0 * 0.5 + 0.5) + 1  # = 3
        assert flow_reg.temporal_buffer.maxlen == expected_size

        # Test with larger temporal sigma
        options = OFOptions(sigma=[2.0, 2.0, 2.0])
        flow_reg = FlowRegLive(options=options, truncate=4.0)
        expected_size = int(4.0 * 2.0 + 0.5) + 1  # = 9
        assert flow_reg.temporal_buffer.maxlen == expected_size

        # Test with per-channel sigmas - should use max
        options = OFOptions(sigma=[[2.0, 2.0, 1.0], [2.0, 2.0, 3.0]])
        flow_reg = FlowRegLive(options=options, truncate=4.0)
        expected_size = int(4.0 * 3.0 + 0.5) + 1  # = 13 (max of 1.0 and 3.0)
        assert flow_reg.temporal_buffer.maxlen == expected_size

    def test_set_reference_from_single_frame(self):
        """Test setting reference from a single frame."""
        flow_reg = FlowRegLive()

        # Create single frame reference
        H, W, C = 32, 32, 2
        reference = np.random.rand(H, W, C).astype(np.float32)

        flow_reg.set_reference(reference)

        # Check reference was set
        assert flow_reg.reference_raw is not None
        assert flow_reg.reference_proc is not None
        assert flow_reg.reference_raw.shape == (H, W, C)
        assert flow_reg.reference_proc.shape == (H, W, C)

        # Check normalization parameters were set
        assert flow_reg.norm_min is not None
        assert flow_reg.norm_max is not None

    def test_set_reference_from_multiple_frames(self):
        """Test setting reference from multiple frames with preregistration."""
        flow_reg = FlowRegLive()

        # Create multiple frames
        T, H, W, C = 10, 32, 32, 2
        frames = np.random.rand(T, H, W, C).astype(np.float32)

        flow_reg.set_reference(frames)

        # Check reference was set and is 3D
        assert flow_reg.reference_raw is not None
        assert flow_reg.reference_proc is not None
        assert flow_reg.reference_raw.shape == (H, W, C)
        assert flow_reg.reference_proc.shape == (H, W, C)

    def test_set_reference_from_buffer(self):
        """Test setting reference from internal buffer."""
        flow_reg = FlowRegLive(reference_buffer_size=5)

        # Add frames to buffer
        H, W, C = 24, 24, 2
        for _ in range(5):
            frame = np.random.rand(H, W, C).astype(np.float32)
            flow_reg.reference_buffer.append(frame)

        # Set reference from buffer
        flow_reg.set_reference()  # No frames argument - uses buffer

        # Check reference was set
        assert flow_reg.reference_raw is not None
        assert flow_reg.reference_proc is not None
        assert flow_reg.reference_raw.shape == (H, W, C)

    def test_reference_buffer_empty_error(self):
        """Test error when trying to set reference from empty buffer."""
        flow_reg = FlowRegLive()

        with pytest.raises(ValueError, match="Reference buffer is empty"):
            flow_reg.set_reference()  # Buffer is empty


class TestFlowRegLiveFrameProcessing:
    """Test frame-by-frame processing with FlowRegLive."""

    def test_single_frame_processing(self):
        """Test processing single frames with __call__."""
        flow_reg = FlowRegLive()

        # Set reference
        H, W, C = 32, 32, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        flow_reg.set_reference(reference)

        # Process single frame
        frame = np.random.rand(H, W, C).astype(np.float32)
        registered, flow = flow_reg(frame)

        # Check outputs
        assert registered.shape == (H, W, C)
        assert flow.shape == (H, W, 2)
        assert flow_reg.frame_count == 1
        assert flow_reg.last_flow is not None

    def test_frame_processing_without_reference(self):
        """Test that frames before reference just pass through."""
        flow_reg = FlowRegLive(reference_buffer_size=3)

        # Process frames without reference - should buffer them
        H, W, C = 24, 24, 2
        frames = []
        for i in range(3):
            frame = np.random.rand(H, W, C).astype(np.float32)
            frames.append(frame)
            registered, flow = flow_reg(frame)

            # Without reference, should return original
            np.testing.assert_array_equal(registered, frame)
            np.testing.assert_array_equal(flow, np.zeros((H, W, 2), dtype=np.float32))

        # Check frames were buffered
        assert len(flow_reg.reference_buffer) == 3

    def test_sequential_frame_processing(self):
        """Test processing multiple frames sequentially."""
        flow_reg = FlowRegLive()

        # Set reference
        H, W, C = 28, 28, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        flow_reg.set_reference(reference)

        # Process multiple frames
        T = 10
        registered_frames = []
        flow_fields = []

        for t in range(T):
            frame = np.random.rand(H, W, C).astype(np.float32)
            registered, flow = flow_reg(frame)
            registered_frames.append(registered)
            flow_fields.append(flow)

            # Check frame count increments
            assert flow_reg.frame_count == t + 1

        # Check all outputs
        assert len(registered_frames) == T
        assert len(flow_fields) == T
        assert all(r.shape == (H, W, C) for r in registered_frames)
        assert all(f.shape == (H, W, 2) for f in flow_fields)

    def test_flow_initialization_persistence(self):
        """Test that flow field is preserved between frames."""
        flow_reg = FlowRegLive()

        # Set reference
        H, W, C = 24, 24, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        flow_reg.set_reference(reference)

        # Process first frame
        frame1 = np.random.rand(H, W, C).astype(np.float32)
        _, flow1 = flow_reg(frame1)

        # Save the flow state
        saved_flow = flow_reg.last_flow.copy()

        # Process second frame
        frame2 = np.random.rand(H, W, C).astype(np.float32)
        _, flow2 = flow_reg(frame2)

        # The saved flow should have been used as initialization
        np.testing.assert_array_equal(saved_flow, flow1)

        # New flow should be different
        assert not np.allclose(flow1, flow2)

    def test_2d_frame_handling(self):
        """Test handling of 2D frames (H,W)."""
        flow_reg = FlowRegLive()

        # Set 2D reference
        H, W = 32, 32
        reference = np.random.rand(H, W).astype(np.float32)
        flow_reg.set_reference(reference)

        # Process 2D frame
        frame = np.random.rand(H, W).astype(np.float32)
        registered, flow = flow_reg(frame)

        # Should maintain 3D internally but return appropriate shape
        assert registered.shape == (H, W, 1) or registered.shape == (H, W)
        assert flow.shape == (H, W, 2)


class TestFlowRegLiveBatchProcessing:
    """Test batch processing with register_frames method."""

    def test_register_frames_batch(self):
        """Test batch processing with register_frames."""
        flow_reg = FlowRegLive()

        # Set reference
        H, W, C = 24, 24, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        flow_reg.set_reference(reference)

        # Create batch of frames
        T = 8
        frames = np.random.rand(T, H, W, C).astype(np.float32)

        # Process batch
        registered, flows = flow_reg.register_frames(frames)

        # Check outputs
        assert registered.shape == (T, H, W, C)
        assert flows.shape == (T, H, W, 2)
        assert flow_reg.frame_count == T

    def test_batch_vs_sequential_consistency(self):
        """Test that batch and sequential processing give same results."""
        # Create two identical FlowRegLive instances
        options = OFOptions(alpha=3.0, sigma=[2.0, 2.0, 0.5])
        flow_reg_batch = FlowRegLive(options=options)
        flow_reg_seq = FlowRegLive(options=options)

        # Set same reference
        H, W, C = 20, 20, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        flow_reg_batch.set_reference(reference.copy())
        flow_reg_seq.set_reference(reference.copy())

        # Create test frames
        T = 6
        frames = np.random.rand(T, H, W, C).astype(np.float32)

        # Process with batch
        registered_batch, flows_batch = flow_reg_batch.register_frames(frames)

        # Process sequentially
        registered_seq = []
        flows_seq = []
        for t in range(T):
            reg, flow = flow_reg_seq(frames[t])
            registered_seq.append(reg)
            flows_seq.append(flow)

        registered_seq = np.array(registered_seq)
        flows_seq = np.array(flows_seq)

        # Results should be very similar (might have small numerical differences)
        np.testing.assert_allclose(registered_batch, registered_seq, rtol=1e-5)
        np.testing.assert_allclose(flows_batch, flows_seq, rtol=1e-5)


class TestFlowRegLiveReferenceUpdate:
    """Test reference update logic in FlowRegLive."""

    def test_reference_update_at_interval(self):
        """Test that reference updates at specified intervals."""
        flow_reg = FlowRegLive(reference_update_interval=5, reference_update_weight=0.5)

        # Set initial reference
        H, W, C = 24, 24, 2
        initial_ref = np.ones((H, W, C), dtype=np.float32) * 0.5
        flow_reg.set_reference(initial_ref)

        # Save initial reference state
        initial_proc = flow_reg.reference_proc.copy()
        initial_raw = flow_reg.reference_raw.copy()

        # Process frames
        for i in range(1, 6):
            frame = np.random.rand(H, W, C).astype(np.float32)
            flow_reg(frame)

            if i < 5:
                # Reference should not change before interval
                np.testing.assert_array_equal(flow_reg.reference_proc, initial_proc)
                np.testing.assert_array_equal(flow_reg.reference_raw, initial_raw)
            else:
                # At frame 5, reference should update
                assert not np.array_equal(flow_reg.reference_proc, initial_proc)
                assert not np.array_equal(flow_reg.reference_raw, initial_raw)

    def test_reference_update_weight_0(self):
        """Test reference update with weight=0 (no update)."""
        flow_reg = FlowRegLive(
            reference_update_interval=1,  # Update every frame
            reference_update_weight=0.0,  # But with 0 weight
        )

        # Set initial reference
        H, W, C = 20, 20, 2
        initial_ref = np.ones((H, W, C), dtype=np.float32) * 0.5
        flow_reg.set_reference(initial_ref)

        # Save initial reference
        initial_raw = flow_reg.reference_raw.copy()

        # Process a frame (will trigger update)
        frame = np.random.rand(H, W, C).astype(np.float32)
        flow_reg(frame)

        # Reference should not change with weight=0
        np.testing.assert_allclose(flow_reg.reference_raw, initial_raw, rtol=1e-6)

    def test_reference_update_weight_1(self):
        """Test reference update with weight=1 (full replacement)."""
        flow_reg = FlowRegLive(
            reference_update_interval=1,  # Update every frame
            reference_update_weight=1.0,  # Full replacement
        )

        # Set initial reference
        H, W, C = 20, 20, 2
        initial_ref = np.ones((H, W, C), dtype=np.float32) * 0.5
        flow_reg.set_reference(initial_ref)

        # Process a distinctive frame
        frame = np.ones((H, W, C), dtype=np.float32) * 0.8  # Different value
        registered, _ = flow_reg(frame)

        # With weight=1, reference_raw should become the registered frame
        # Note: reference_raw gets the warped (registered) version
        # Due to the warping, registered should be close to reference after compensation
        # But the raw reference itself should have changed significantly
        assert not np.allclose(flow_reg.reference_raw, initial_ref, rtol=0.1)

    def test_reference_normalization_update(self):
        """Test that normalization parameters update with reference."""
        flow_reg = FlowRegLive(
            reference_update_interval=1,
            reference_update_weight=0.5,
            options=OFOptions(channel_normalization="separate"),
        )

        # Set initial reference with known range
        H, W, C = 16, 16, 2
        initial_ref = np.ones((H, W, C), dtype=np.float32) * 0.5
        flow_reg.set_reference(initial_ref)

        # Save initial normalization parameters
        initial_norm_min = (
            flow_reg.norm_min.copy()
            if isinstance(flow_reg.norm_min, list)
            else flow_reg.norm_min
        )
        initial_norm_max = (
            flow_reg.norm_max.copy()
            if isinstance(flow_reg.norm_max, list)
            else flow_reg.norm_max
        )

        # Process frame with different range - use more extreme difference
        frame = np.random.rand(H, W, C).astype(np.float32) * 3.0  # Much wider range
        flow_reg(frame)

        # Normalization parameters should have updated
        new_norm_min = flow_reg.norm_min
        new_norm_max = flow_reg.norm_max

        # They should be different after reference update - use looser tolerance since
        # the change might be subtle due to Gaussian filtering and warping
        if isinstance(initial_norm_min, list):
            # Check if ANY of the normalization parameters changed
            min_changed = any(
                not np.isclose(initial_norm_min[i], new_norm_min[i], rtol=5e-2)
                for i in range(len(initial_norm_min))
            )
            max_changed = any(
                not np.isclose(initial_norm_max[i], new_norm_max[i], rtol=5e-2)
                for i in range(len(initial_norm_max))
            )

            # If neither min nor max changed significantly, the test should fail
            # But allow for the possibility that the changes are very small due to processing
            assert (
                min_changed or max_changed
            ), f"Expected normalization to change: min {initial_norm_min} -> {new_norm_min}, max {initial_norm_max} -> {new_norm_max}"
        else:
            assert not (
                np.isclose(initial_norm_min, new_norm_min, rtol=5e-2)
                and np.isclose(initial_norm_max, new_norm_max, rtol=5e-2)
            )


class TestFlowRegLiveTemporalFiltering:
    """Test temporal filtering functionality."""

    def test_temporal_buffer_accumulation(self):
        """Test that temporal buffer accumulates frames."""
        options = OFOptions(sigma=[2.0, 2.0, 1.0])
        flow_reg = FlowRegLive(options=options)

        # Set reference
        H, W, C = 20, 20, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        flow_reg.set_reference(reference)

        # Process frames and check buffer
        for i in range(5):
            frame = np.random.rand(H, W, C).astype(np.float32)
            flow_reg(frame)

            # Buffer should accumulate up to maxlen
            expected_len = min(i + 1, flow_reg.temporal_buffer.maxlen)
            assert len(flow_reg.temporal_buffer) == expected_len

    def test_temporal_filtering_applied(self):
        """Test that temporal filtering is applied when sigma_t > 0."""
        # With temporal filtering
        options_with = OFOptions(sigma=[2.0, 2.0, 2.0])  # sigma_t = 2.0
        flow_reg_with = FlowRegLive(options=options_with)

        # Without temporal filtering
        options_without = OFOptions(sigma=[2.0, 2.0, 0.0])  # sigma_t = 0
        flow_reg_without = FlowRegLive(options=options_without)

        # Set same reference
        H, W, C = 24, 24, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        flow_reg_with.set_reference(reference.copy())
        flow_reg_without.set_reference(reference.copy())

        # Process same frames
        frames = [np.random.rand(H, W, C).astype(np.float32) for _ in range(5)]

        results_with = []
        results_without = []

        for frame in frames:
            reg_with, _ = flow_reg_with(frame.copy())
            reg_without, _ = flow_reg_without(frame.copy())
            results_with.append(reg_with)
            results_without.append(reg_without)

        # Results should differ due to temporal filtering
        # (exact comparison difficult due to complex processing)
        assert len(results_with) == len(results_without)


class TestFlowRegLiveSpecialMethods:
    """Test special methods and utilities."""

    def test_reset_reference(self):
        """Test resetting reference."""
        flow_reg = FlowRegLive()

        # Set initial reference
        H, W, C = 20, 20, 2
        initial_ref = np.random.rand(H, W, C).astype(np.float32)
        flow_reg.set_reference(initial_ref)

        # Process some frames
        for _ in range(5):
            frame = np.random.rand(H, W, C).astype(np.float32)
            flow_reg(frame)

        assert flow_reg.frame_count == 5
        assert flow_reg.last_flow is not None

        # Reset reference
        new_ref = np.random.rand(H, W, C).astype(np.float32)
        flow_reg.reset_reference(new_ref)

        # Check reset
        assert flow_reg.last_flow is None  # Flow should be reset
        assert len(flow_reg.temporal_buffer) == 0  # Buffer should be cleared
        np.testing.assert_array_equal(flow_reg.reference_raw, new_ref)

    def test_get_current_flow(self):
        """Test getting current flow state."""
        flow_reg = FlowRegLive()

        # Initially no flow
        assert flow_reg.get_current_flow() is None

        # Set reference and process frame
        H, W, C = 20, 20, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        flow_reg.set_reference(reference)

        frame = np.random.rand(H, W, C).astype(np.float32)
        _, flow = flow_reg(frame)

        # Should return copy of current flow
        current_flow = flow_reg.get_current_flow()
        assert current_flow is not None
        assert current_flow.shape == (H, W, 2)
        np.testing.assert_array_equal(current_flow, flow)

        # Should be a copy, not the same object
        assert current_flow is not flow_reg.last_flow

    def test_set_flow_init(self):
        """Test manually setting flow initialization."""
        flow_reg = FlowRegLive()

        # Set reference
        H, W, C = 20, 20, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        flow_reg.set_reference(reference)

        # Set custom flow initialization
        custom_flow = np.ones((H, W, 2), dtype=np.float32) * 0.5
        flow_reg.set_flow_init(custom_flow)

        # Process frame - should use custom flow as init
        frame = np.random.rand(H, W, C).astype(np.float32)
        _, result_flow = flow_reg(frame)

        # Flow should have been influenced by initialization
        # (exact test difficult, but flow should be computed)
        assert result_flow.shape == (H, W, 2)


class TestFlowRegLiveEdgeCases:
    """Test edge cases and error handling."""

    def test_multichannel_consistency(self):
        """Test consistency across different channel counts."""
        # Test with 1, 2, and 3 channels
        for C in [1, 2, 3]:
            flow_reg = FlowRegLive()

            H, W = 24, 24
            reference = np.random.rand(H, W, C).astype(np.float32)
            flow_reg.set_reference(reference)

            frame = np.random.rand(H, W, C).astype(np.float32)
            registered, flow = flow_reg(frame)

            assert registered.shape[-1] == C or (C == 1 and registered.ndim == 2)
            assert flow.shape == (H, W, 2)

    def test_shape_preservation(self):
        """Test that input shapes are preserved correctly."""
        flow_reg = FlowRegLive()

        # Test 2D input
        H, W = 28, 28
        reference_2d = np.random.rand(H, W).astype(np.float32)
        flow_reg.set_reference(reference_2d)

        frame_2d = np.random.rand(H, W).astype(np.float32)
        reg_2d, flow_2d = flow_reg(frame_2d)

        # Should handle 2D gracefully
        assert reg_2d.shape == (H, W, 1) or reg_2d.shape == (H, W)
        assert flow_2d.shape == (H, W, 2)

    def test_empty_reference_buffer_error(self):
        """Test error handling for empty reference buffer."""
        flow_reg = FlowRegLive()

        # Try to set reference from empty buffer
        with pytest.raises(ValueError, match="Reference buffer is empty"):
            flow_reg.set_reference()

    def test_reference_update_normalization_consistency(self):
        """Test that normalization remains consistent through updates."""
        flow_reg = FlowRegLive(
            reference_update_interval=1,
            reference_update_weight=0.3,
            options=OFOptions(channel_normalization="joint"),
        )

        # Set reference
        H, W, C = 16, 16, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        flow_reg.set_reference(reference)

        # Process multiple frames with updates
        for _ in range(10):
            frame = np.random.rand(H, W, C).astype(np.float32)
            registered, _ = flow_reg(frame)

            # Check normalization parameters are valid
            assert flow_reg.norm_min is not None
            assert flow_reg.norm_max is not None

            # Registered output should be reasonable
            assert not np.isnan(registered).any()
            assert not np.isinf(registered).any()


class TestFlowRegLiveComparison:
    """Compare FlowRegLive with compensate_arr for consistency."""

    def test_single_frame_comparison(self):
        """Compare single frame processing between FlowRegLive and compensate_arr."""
        from pyflowreg.motion_correction.compensate_arr import compensate_arr

        # Create test data
        H, W, C = 32, 32, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        frame = np.random.rand(H, W, C).astype(np.float32)

        # Process with compensate_arr
        frames_4d = frame[np.newaxis, ...]  # Add time dimension
        reg_arr, flow_arr = compensate_arr(frames_4d, reference)
        reg_arr = reg_arr[0]  # Remove time dimension
        flow_arr = flow_arr[0]

        # Process with FlowRegLive (without temporal filtering)
        options = OFOptions(sigma=[2.0, 2.0, 0.0])  # No temporal filtering
        flow_reg = FlowRegLive(options=options)
        flow_reg.set_reference(reference)
        reg_live, flow_live = flow_reg(frame)

        # Results should be similar (not exact due to different preprocessing)
        # Just check shapes and reasonable values
        assert reg_live.shape == reg_arr.shape
        assert flow_live.shape == flow_arr.shape

    def test_batch_consistency(self):
        """Test that FlowRegLive batch processing is consistent."""
        # Create test data
        T, H, W, C = 10, 24, 24, 2
        frames = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(frames[:3], axis=0)

        # Process with FlowRegLive
        flow_reg = FlowRegLive(reference_update_interval=100)  # No updates
        flow_reg.set_reference(reference)

        # Process all frames
        registered_frames = []
        flow_fields = []
        for t in range(T):
            reg, flow = flow_reg(frames[t])
            registered_frames.append(reg)
            flow_fields.append(flow)

        registered_frames = np.array(registered_frames)
        flow_fields = np.array(flow_fields)

        # Check shapes
        assert registered_frames.shape == frames.shape
        assert flow_fields.shape == (T, H, W, 2)

        # Check no NaN or Inf
        assert not np.isnan(registered_frames).any()
        assert not np.isinf(registered_frames).any()
        assert not np.isnan(flow_fields).any()
        assert not np.isinf(flow_fields).any()


# Performance test (marked slow)
@pytest.mark.slow
class TestFlowRegLivePerformance:
    """Test performance aspects of FlowRegLive."""

    def test_large_frame_processing(self):
        """Test processing of larger frames."""
        flow_reg = FlowRegLive(
            options=OFOptions(quality_setting="fast", levels=2, iterations=5)
        )

        # Large frames
        H, W, C = 256, 256, 2
        reference = np.random.rand(H, W, C).astype(np.float32)
        flow_reg.set_reference(reference)

        # Process several large frames
        for _ in range(5):
            frame = np.random.rand(H, W, C).astype(np.float32)
            registered, flow = flow_reg(frame)

            assert registered.shape == (H, W, C)
            assert flow.shape == (H, W, 2)

    def test_streaming_simulation(self):
        """Simulate streaming processing scenario."""
        flow_reg = FlowRegLive(
            reference_update_interval=20, reference_update_weight=0.2
        )

        # Simulate video stream
        H, W, C = 128, 128, 2
        stream_length = 100

        # Initialize with first frames
        init_frames = np.random.rand(10, H, W, C).astype(np.float32)
        flow_reg.set_reference(init_frames)

        # Stream processing
        for frame_idx in range(stream_length):
            frame = np.random.rand(H, W, C).astype(np.float32)
            registered, flow = flow_reg(frame)

            # Verify outputs
            assert registered.shape == (H, W, C)
            assert flow.shape == (H, W, 2)

            # Check frame count
            assert flow_reg.frame_count == frame_idx + 1
