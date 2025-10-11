"""
Tests for callback API extensions in motion compensation.

Tests the w_callback and registered_callback functionality for exposing
intermediate batch data during motion correction processing.
"""

import pytest
import numpy as np
from typing import List, Tuple

from pyflowreg.motion_correction.compensate_arr import compensate_arr
from pyflowreg.motion_correction.compensate_recording import (
    BatchMotionCorrector,
)
from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat


class CallbackTracker:
    """Helper class to track callback invocations."""

    def __init__(self):
        self.w_calls: List[Tuple[np.ndarray, int, int]] = []
        self.registered_calls: List[Tuple[np.ndarray, int, int]] = []
        self.progress_calls: List[Tuple[int, int]] = []

    def w_callback(self, w_batch: np.ndarray, start_idx: int, end_idx: int):
        """Track displacement field callbacks."""
        self.w_calls.append((w_batch.copy(), start_idx, end_idx))

    def registered_callback(
        self, registered_batch: np.ndarray, start_idx: int, end_idx: int
    ):
        """Track registered frame callbacks."""
        self.registered_calls.append((registered_batch.copy(), start_idx, end_idx))

    def progress_callback(self, current: int, total: int):
        """Track progress callbacks."""
        self.progress_calls.append((current, total))


class TestWCallback:
    """Test displacement field (w) callback functionality."""

    def test_w_callback_invoked(self):
        """Test that w_callback is invoked during processing."""
        tracker = CallbackTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
            save_w=True,
        )

        registered, w = compensate_arr(
            video, reference, options=options, w_callback=tracker.w_callback
        )

        # Callback should have been invoked
        assert len(tracker.w_calls) > 0

    def test_w_callback_receives_correct_data(self):
        """Test that w_callback receives displacement fields with correct shape."""
        tracker = CallbackTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
            save_w=True,
        )

        registered, w = compensate_arr(
            video, reference, options=options, w_callback=tracker.w_callback
        )

        # Check each callback invocation
        for w_batch, start_idx, end_idx in tracker.w_calls:
            # Displacement field should have shape (T, H, W, 2)
            assert w_batch.ndim == 4
            assert w_batch.shape[1:3] == (16, 32)  # H, W
            assert w_batch.shape[3] == 2  # u, v components

            # Batch size should match indices
            assert w_batch.shape[0] == end_idx - start_idx

    def test_w_callback_batch_indices(self):
        """Test that w_callback receives correct batch indices."""
        tracker = CallbackTracker()

        video = np.random.rand(25, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,  # Force multiple batches
            save_w=True,
        )

        registered, w = compensate_arr(
            video, reference, options=options, w_callback=tracker.w_callback
        )

        # Verify indices are sequential and cover all frames
        all_indices = []
        for _, start_idx, end_idx in tracker.w_calls:
            all_indices.extend(range(start_idx, end_idx))

        assert all_indices == list(range(25)), "Indices should cover all frames"

    def test_w_callback_multiple_batches(self):
        """Test w_callback with multiple batches."""
        tracker = CallbackTracker()

        video = np.random.rand(30, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=8,  # Should create multiple batches
            save_w=True,
        )

        compensate_arr(video, reference, options=options, w_callback=tracker.w_callback)

        # Should have multiple batches
        assert len(tracker.w_calls) > 1

        # Total frames should sum to 30
        total_frames = sum(
            end_idx - start_idx for _, start_idx, end_idx in tracker.w_calls
        )
        assert total_frames == 30


class TestRegisteredCallback:
    """Test registered frame callback functionality."""

    def test_registered_callback_invoked(self):
        """Test that registered_callback is invoked during processing."""
        tracker = CallbackTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
        )

        registered, w = compensate_arr(
            video,
            reference,
            options=options,
            registered_callback=tracker.registered_callback,
        )

        # Callback should have been invoked
        assert len(tracker.registered_calls) > 0

    def test_registered_callback_receives_correct_data(self):
        """Test that registered_callback receives frames with correct shape."""
        tracker = CallbackTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
        )

        registered, w = compensate_arr(
            video,
            reference,
            options=options,
            registered_callback=tracker.registered_callback,
        )

        # Check each callback invocation
        for batch, start_idx, end_idx in tracker.registered_calls:
            # Should have shape (T, H, W, C)
            assert batch.ndim == 4
            assert batch.shape[1:] == (16, 32, 2)  # H, W, C

            # Batch size should match indices
            assert batch.shape[0] == end_idx - start_idx

    def test_registered_callback_batch_indices(self):
        """Test that registered_callback receives correct batch indices."""
        tracker = CallbackTracker()

        video = np.random.rand(25, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
        )

        compensate_arr(
            video,
            reference,
            options=options,
            registered_callback=tracker.registered_callback,
        )

        # Verify indices are sequential and cover all frames
        all_indices = []
        for _, start_idx, end_idx in tracker.registered_calls:
            all_indices.extend(range(start_idx, end_idx))

        assert all_indices == list(range(25)), "Indices should cover all frames"


class TestCombinedCallbacks:
    """Test using multiple callbacks together."""

    def test_all_callbacks_together(self):
        """Test using progress, w, and registered callbacks together."""
        tracker = CallbackTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
            save_w=True,
        )

        registered, w = compensate_arr(
            video,
            reference,
            options=options,
            progress_callback=tracker.progress_callback,
            w_callback=tracker.w_callback,
            registered_callback=tracker.registered_callback,
        )

        # All callbacks should have been invoked
        assert len(tracker.progress_calls) > 0
        assert len(tracker.w_calls) > 0
        assert len(tracker.registered_calls) > 0

        # Number of w and registered callbacks should match
        assert len(tracker.w_calls) == len(tracker.registered_calls)

    def test_callbacks_with_null_writer(self):
        """Test callbacks work with NULL output format."""
        tracker = CallbackTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.NULL,  # No output storage
            levels=1,
            iterations=2,
            buffer_size=10,
            save_w=True,
        )

        registered, w = compensate_arr(
            video,
            reference,
            options=options,
            w_callback=tracker.w_callback,
            registered_callback=tracker.registered_callback,
        )

        # Callbacks should still work even with NULL writer
        assert len(tracker.w_calls) > 0
        assert len(tracker.registered_calls) > 0

        # Results should still be returned
        assert registered.shape == video.shape
        assert w.shape == (20, 16, 32, 2)


class TestCallbackDirectRegistration:
    """Test direct callback registration on BatchMotionCorrector."""

    def test_register_w_callback(self):
        """Test register_w_callback method."""
        tracker = CallbackTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            input_file=video,
            reference_frames=reference,
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
            save_w=True,
        )

        compensator = BatchMotionCorrector(options)
        compensator.register_w_callback(tracker.w_callback)
        compensator.run()

        assert len(tracker.w_calls) > 0

    def test_register_registered_callback(self):
        """Test register_registered_callback method."""
        tracker = CallbackTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            input_file=video,
            reference_frames=reference,
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
        )

        compensator = BatchMotionCorrector(options)
        compensator.register_registered_callback(tracker.registered_callback)
        compensator.run()

        assert len(tracker.registered_calls) > 0

    def test_multiple_callback_registration(self):
        """Test registering multiple callbacks of same type."""
        tracker1 = CallbackTracker()
        tracker2 = CallbackTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            input_file=video,
            reference_frames=reference,
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
            save_w=True,
        )

        compensator = BatchMotionCorrector(options)
        compensator.register_w_callback(tracker1.w_callback)
        compensator.register_w_callback(tracker2.w_callback)
        compensator.run()

        # Both trackers should receive callbacks
        assert len(tracker1.w_calls) > 0
        assert len(tracker2.w_calls) > 0
        assert len(tracker1.w_calls) == len(tracker2.w_calls)


class TestCallbackErrorHandling:
    """Test error handling in callbacks."""

    def test_callback_exception_handled(self):
        """Test that exceptions in callbacks are caught and logged."""

        def failing_callback(data, start, end):
            raise RuntimeError("Intentional callback error")

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
            save_w=True,
        )

        # Should not raise - error should be caught and warned
        with pytest.warns(UserWarning, match="callback error"):
            registered, w = compensate_arr(
                video,
                reference,
                options=options,
                w_callback=failing_callback,
            )

        # Processing should complete despite callback error
        assert registered.shape == video.shape

    def test_partial_callback_failure(self):
        """Test that one failing callback doesn't affect others."""
        tracker = CallbackTracker()

        def failing_w_callback(data, start, end):
            raise RuntimeError("W callback error")

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
            save_w=True,
        )

        # W callback fails, but registered callback should still work
        with pytest.warns(UserWarning):
            registered, w = compensate_arr(
                video,
                reference,
                options=options,
                w_callback=failing_w_callback,
                registered_callback=tracker.registered_callback,
            )

        # Registered callback should have been invoked
        assert len(tracker.registered_calls) > 0


class TestCallbackDataIntegrity:
    """Test that callback data is correct and complete."""

    def test_w_callback_data_matches_output(self):
        """Test that w_callback data matches final w output."""
        tracker = CallbackTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
            save_w=True,
        )

        registered, w_final = compensate_arr(
            video, reference, options=options, w_callback=tracker.w_callback
        )

        # Reconstruct w from callbacks
        w_from_callbacks = []
        for w_batch, start_idx, end_idx in sorted(tracker.w_calls, key=lambda x: x[1]):
            w_from_callbacks.append(w_batch)

        w_reconstructed = np.concatenate(w_from_callbacks, axis=0)

        # Should match final output
        np.testing.assert_array_almost_equal(w_reconstructed, w_final)

    def test_registered_callback_data_matches_output(self):
        """Test that registered_callback data matches final registered output."""
        tracker = CallbackTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.ARRAY,
            levels=1,
            iterations=2,
            buffer_size=10,
        )

        registered_final, w = compensate_arr(
            video,
            reference,
            options=options,
            registered_callback=tracker.registered_callback,
        )

        # Reconstruct registered from callbacks
        registered_from_callbacks = []
        for batch, start_idx, end_idx in sorted(
            tracker.registered_calls, key=lambda x: x[1]
        ):
            registered_from_callbacks.append(batch)

        registered_reconstructed = np.concatenate(registered_from_callbacks, axis=0)

        # Should match final output
        np.testing.assert_array_almost_equal(registered_reconstructed, registered_final)


class TestCallbackUseCases:
    """Test real-world callback use cases."""

    def test_online_statistics_collection(self):
        """Test using callbacks for online statistics collection."""

        class StatsCollector:
            def __init__(self):
                self.mean_displacements = []
                self.max_displacements = []

            def collect_stats(self, w_batch, start_idx, end_idx):
                for t in range(w_batch.shape[0]):
                    mag = np.sqrt(w_batch[t, :, :, 0] ** 2 + w_batch[t, :, :, 1] ** 2)
                    self.mean_displacements.append(np.mean(mag))
                    self.max_displacements.append(np.max(mag))

        collector = StatsCollector()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.NULL,  # Don't need output
            levels=1,
            iterations=2,
            buffer_size=10,
            save_w=True,
        )

        compensate_arr(
            video, reference, options=options, w_callback=collector.collect_stats
        )

        # Should have collected stats for all frames
        assert len(collector.mean_displacements) == 20
        assert len(collector.max_displacements) == 20

    def test_trajectory_tracking(self):
        """Test using callbacks for trajectory tracking."""

        class TrajectoryTracker:
            def __init__(self):
                self.trajectories = []

            def track(self, w_batch, start_idx, end_idx):
                # Track mean displacement trajectory
                for t in range(w_batch.shape[0]):
                    mean_u = np.mean(w_batch[t, :, :, 0])
                    mean_v = np.mean(w_batch[t, :, :, 1])
                    self.trajectories.append((start_idx + t, mean_u, mean_v))

        tracker = TrajectoryTracker()

        video = np.random.rand(20, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.NULL,
            levels=1,
            iterations=2,
            buffer_size=10,
            save_w=True,
        )

        compensate_arr(video, reference, options=options, w_callback=tracker.track)

        # Should have trajectory for all frames
        assert len(tracker.trajectories) == 20

        # Frame indices should be sequential
        frame_indices = [t[0] for t in tracker.trajectories]
        assert frame_indices == list(range(20))
