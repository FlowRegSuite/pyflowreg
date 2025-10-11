"""
Tests for workflows.md code examples.

This module contains manual mirror tests that validate all code examples
in the workflows guide execute correctly using existing test fixtures.
"""

import numpy as np

from pyflowreg.motion_correction import compensate_arr, OFOptions
from pyflowreg.motion_correction.OF_options import OutputFormat
from pyflowreg.util.io import NullVideoWriter


class TestOutputFormats:
    """Test output format examples from workflows.md."""

    def test_output_format_array(self, test_data_array, fast_of_options):
        """Test memory accumulation with OutputFormat.ARRAY."""
        video, _ = test_data_array
        reference = np.mean(video[:10], axis=0)

        # Memory accumulation (default-like behavior)
        options_memory = OFOptions(
            output_format=OutputFormat.ARRAY, quality_setting="fast"
        )

        registered, w = compensate_arr(video, reference, options_memory)

        # Verify output
        assert registered.shape == video.shape
        assert w.shape == (video.shape[0], video.shape[1], video.shape[2], 2)

    def test_output_format_null(self, test_data_array, fast_of_options):
        """Test callback-only processing with OutputFormat.NULL."""
        video, _ = test_data_array
        reference = np.mean(video[:10], axis=0)

        # No storage (for real-time processing via callbacks)
        options_realtime = OFOptions(
            output_format=OutputFormat.NULL, quality_setting="fast", save_w=True
        )

        registered, w = compensate_arr(video, reference, options_realtime)

        # Still get outputs from compensate_arr
        assert registered.shape == video.shape
        assert w.shape == (video.shape[0], video.shape[1], video.shape[2], 2)


class TestCallbackAPI:
    """Test real-time data access via callbacks from workflows.md."""

    def test_basic_callback_usage(self, test_data_array, fast_of_options):
        """Test basic callback usage example from workflows.md."""
        video, _ = test_data_array
        reference = np.mean(video[:10], axis=0)

        # Track callback invocations
        motion_data = []
        display_data = []

        def monitor_motion(w_batch, start_idx, end_idx):
            """Monitor displacement fields during processing."""
            for t in range(w_batch.shape[0]):
                magnitude = np.sqrt(w_batch[t, :, :, 0] ** 2 + w_batch[t, :, :, 1] ** 2)
                motion_data.append(
                    {"frame": start_idx + t, "magnitude": np.mean(magnitude)}
                )

        def display_frames(batch, start_idx, end_idx):
            """Display corrected frames as they're computed."""
            display_data.append(
                {"start": start_idx, "end": end_idx, "shape": batch.shape}
            )

        # Use callbacks during processing
        registered, w = compensate_arr(
            video,
            reference,
            options=fast_of_options,
            w_callback=monitor_motion,
            registered_callback=display_frames,
        )

        # Verify callbacks were called
        assert len(motion_data) == video.shape[0]
        assert len(display_data) > 0
        assert registered.shape == video.shape

    def test_processing_without_storage(self, test_data_array):
        """Test callback-only processing example from workflows.md."""

        class LiveProcessor:
            def __init__(self):
                self.statistics = []

            def process_batch(self, w_batch, start_idx, end_idx):
                # Process displacement fields without storing video
                batch_stats = {
                    "start": start_idx,
                    "end": end_idx,
                    "mean_motion": np.mean(
                        np.sqrt(w_batch[..., 0] ** 2 + w_batch[..., 1] ** 2)
                    ),
                }
                self.statistics.append(batch_stats)

        video, _ = test_data_array
        reference = np.mean(video[:10], axis=0)
        processor = LiveProcessor()

        options = OFOptions(
            output_format=OutputFormat.NULL,  # No storage
            save_w=True,
            buffer_size=20,  # Process 20 frames at a time
            quality_setting="fast",
        )

        # Run motion correction without storing output
        registered, w = compensate_arr(
            video, reference, options=options, w_callback=processor.process_batch
        )

        # Access collected statistics
        assert len(processor.statistics) > 0
        assert all("mean_motion" in s for s in processor.statistics)


class TestAdvancedExamples:
    """Test advanced example patterns from workflows.md."""

    def test_motion_tracker(self, test_data_array):
        """Test complete motion tracking system from workflows.md."""

        class MotionTracker:
            """Track motion throughout video using the motion correction API."""

            def __init__(self):
                self.trajectory = []
                self.quality_metrics = []
                self.current_batch = 0

            def track_displacement(self, w_batch, start_idx, end_idx):
                """Track displacement trajectory."""
                for t in range(w_batch.shape[0]):
                    frame_idx = start_idx + t
                    # Calculate mean displacement vector
                    mean_u = np.mean(w_batch[t, :, :, 0])
                    mean_v = np.mean(w_batch[t, :, :, 1])

                    self.trajectory.append(
                        {
                            "frame": frame_idx,
                            "dx": mean_u,
                            "dy": mean_v,
                            "magnitude": np.sqrt(mean_u**2 + mean_v**2),
                        }
                    )

            def assess_quality(self, batch, start_idx, end_idx):
                """Assess quality of corrected frames."""
                for t in range(batch.shape[0]):
                    frame_idx = start_idx + t
                    # Calculate SNR as quality metric
                    signal = np.mean(batch[t])
                    noise = np.std(batch[t])
                    snr = signal / noise if noise > 0 else 0

                    self.quality_metrics.append(
                        {"frame": frame_idx, "snr": snr, "brightness": signal}
                    )

                self.current_batch += 1

            def get_summary(self):
                """Get motion summary statistics."""
                if not self.trajectory:
                    return None

                magnitudes = [t["magnitude"] for t in self.trajectory]
                return {
                    "total_frames": len(self.trajectory),
                    "mean_motion": np.mean(magnitudes),
                    "max_motion": np.max(magnitudes),
                    "std_motion": np.std(magnitudes),
                    "mean_snr": np.mean([q["snr"] for q in self.quality_metrics]),
                }

        # Create tracker
        tracker = MotionTracker()

        # Load video
        video, _ = test_data_array
        reference = np.mean(video[:10], axis=0)

        # Configure for tracking
        options = OFOptions(
            output_format=OutputFormat.NULL,  # Don't store output
            save_w=True,  # Compute displacement fields
            buffer_size=25,  # 25 frames per batch
            levels=2,
            iterations=5,
            quality_setting="fast",
        )

        # Run motion correction with tracking
        registered, w = compensate_arr(
            video,
            reference,
            options=options,
            w_callback=tracker.track_displacement,
            registered_callback=tracker.assess_quality,
        )

        # Get results
        summary = tracker.get_summary()

        # Verify tracking worked
        assert summary is not None
        assert summary["total_frames"] == video.shape[0]
        assert "mean_motion" in summary
        assert "max_motion" in summary
        assert "std_motion" in summary
        assert "mean_snr" in summary


class TestPerformanceOptimization:
    """Test performance optimization examples from workflows.md."""

    def test_buffer_size_selection(self, test_data_array):
        """Test buffer size configuration example."""
        video, _ = test_data_array
        reference = np.mean(video[:10], axis=0)

        # Real-time display - smaller buffers for responsive updates
        options_realtime = OFOptions(
            buffer_size=10,  # Update every 10 frames
            output_format=OutputFormat.NULL,
            quality_setting="fast",
        )

        registered, _ = compensate_arr(video, reference, options_realtime)
        assert registered.shape == video.shape

    def test_callback_performance_pattern(self, test_data_array):
        """Test callback performance guideline pattern."""
        import threading
        from queue import Queue

        class AsyncProcessor:
            def __init__(self):
                self.queue = Queue()
                self.processed_count = 0
                self.running = True
                self.worker = threading.Thread(target=self._process_queue)
                self.worker.daemon = True
                self.worker.start()

            def callback(self, batch, start_idx, end_idx):
                # Quick copy and queue
                self.queue.put((batch.copy(), start_idx, end_idx))

            def _process_queue(self):
                import queue

                while self.running:
                    try:
                        batch, start, end = self.queue.get(timeout=0.1)
                        # Simulate heavy processing
                        self.processed_count += batch.shape[0]
                    except queue.Empty:
                        continue

            def stop(self):
                self.running = False
                self.worker.join(timeout=1)

        video, _ = test_data_array
        reference = np.mean(video[:10], axis=0)

        processor = AsyncProcessor()
        options = OFOptions(output_format=OutputFormat.NULL, quality_setting="fast")

        registered, _ = compensate_arr(
            video, reference, options=options, registered_callback=processor.callback
        )

        # Give worker time to process queue
        import time

        time.sleep(0.5)

        processor.stop()

        # Verify async processing worked
        assert processor.processed_count > 0
        assert registered.shape == video.shape


class TestAPIReference:
    """Test API reference examples from workflows.md."""

    def test_callback_signatures(self, test_data_array, fast_of_options):
        """Test that callback signatures match API reference."""

        # Track signature validation
        progress_calls = []
        w_calls = []
        registered_calls = []

        def progress(current: int, total: int) -> None:
            """Called with current frame number and total frames."""
            assert isinstance(current, int)
            assert isinstance(total, int)
            progress_calls.append((current, total))

        def w_callback(w_batch: np.ndarray, start_idx: int, end_idx: int) -> None:
            """Called with batch of displacement fields."""
            assert isinstance(w_batch, np.ndarray)
            assert w_batch.ndim == 4
            assert w_batch.shape[-1] == 2
            assert isinstance(start_idx, int)
            assert isinstance(end_idx, int)
            w_calls.append((start_idx, end_idx))

        def registered_callback(
            batch: np.ndarray, start_idx: int, end_idx: int
        ) -> None:
            """Called with batch of corrected frames."""
            assert isinstance(batch, np.ndarray)
            assert batch.ndim == 4
            assert isinstance(start_idx, int)
            assert isinstance(end_idx, int)
            registered_calls.append((start_idx, end_idx))

        video, _ = test_data_array
        reference = np.mean(video[:10], axis=0)

        registered, w = compensate_arr(
            video,
            reference,
            options=fast_of_options,
            progress_callback=progress,
            w_callback=w_callback,
            registered_callback=registered_callback,
        )

        # Verify all callbacks were called
        assert len(w_calls) > 0
        assert len(registered_calls) > 0
        # progress_callback might be called depending on executor

    def test_key_configuration_options(self):
        """Test key configuration options from API reference."""
        options = OFOptions(
            output_format=OutputFormat.NULL,  # NULL, ARRAY, HDF5, TIFF, etc.
            buffer_size=20,  # Frames per batch
            save_w=True,  # Compute displacement fields
            levels=5,  # Pyramid levels
            iterations=50,  # Iterations per level
            quality_setting="balanced",  # fast, balanced, accurate
        )

        # Verify configuration
        assert options.output_format == OutputFormat.NULL
        assert options.buffer_size == 20
        assert options.save_w is True
        assert options.levels == 5
        assert options.iterations == 50
        assert options.quality_setting.value == "balanced"


class TestNullWriterIntegration:
    """Test NullVideoWriter integration examples."""

    def test_null_writer_basic_usage(self):
        """Test NullVideoWriter basic usage pattern."""
        writer = NullVideoWriter()
        batch = np.random.rand(5, 16, 32, 2).astype(np.float32)

        writer.write_frames(batch)

        assert writer.frames_written == 5
        assert writer.batches_written == 1

    def test_null_writer_with_output_format(self, test_data_array):
        """Test using NULL output format with compensate_arr."""
        video, _ = test_data_array
        reference = np.mean(video[:10], axis=0)

        options = OFOptions(
            output_format=OutputFormat.NULL,
            levels=1,
            iterations=2,
            save_w=True,
            buffer_size=5,
            quality_setting="fast",
        )

        # Should still return registered array and flow fields
        registered, w = compensate_arr(video, reference, options=options)

        # Verify output
        assert registered.shape == video.shape
        assert w.shape == (video.shape[0], video.shape[1], video.shape[2], 2)
