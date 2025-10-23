"""Tests for parallelization executors and progress callbacks."""

import numpy as np
import pytest
from typing import List, Tuple

from pyflowreg.motion_correction.compensate_arr import compensate_arr
from pyflowreg.motion_correction.compensate_recording import (
    BatchMotionCorrector,
    RegistrationConfig,
)
from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat
from pyflowreg._runtime import RuntimeContext


class TestParallelizationExecutors:
    """Test different parallelization strategies."""

    def test_all_executors_available(self):
        """Test that all expected executors are registered."""
        available = RuntimeContext.get("available_parallelization", set())

        # At minimum, sequential should be available
        assert "sequential" in available, "Sequential executor not available"

        # Check if other executors are registered
        print(f"Available executors: {available}")

    @pytest.mark.parametrize("n_channels", [1, 2, 3])
    def test_sequential_executor(self, n_channels):
        """Test sequential executor processing with different channel counts."""

        # Create test data
        T, H, W, C = 5, 16, 16, n_channels
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Configure for sequential processing
        config = RegistrationConfig(parallelization="sequential")
        options = OFOptions(quality_setting="fast")

        # Setup for array processing
        options.input_file = video
        options.reference_frames = reference
        options.output_format = OutputFormat.ARRAY
        options.save_w = True
        options.save_meta_info = False

        # Create compensator
        compensator = BatchMotionCorrector(options, config)

        # Verify sequential executor is used
        assert compensator.executor.__class__.__name__ == "SequentialExecutor"

        # Run
        compensator.run()

        # Get results
        registered = compensator.video_writer.get_array()
        flow = compensator.w_writer.get_array() if compensator.w_writer else None

        # Verify results
        assert registered.shape == video.shape
        assert flow is not None and flow.shape == (T, H, W, 2)

    @pytest.mark.parametrize("n_channels", [1, 2, 3])
    def test_threading_executor(self, n_channels):
        """Test threading executor with different channel counts."""
        available = RuntimeContext.get("available_parallelization", set())

        if "threading" not in available:
            pytest.skip("Threading executor not available")

        # Create test data
        T, H, W, C = 10, 16, 16, n_channels
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:3], axis=0)

        # Configure for threading
        config = RegistrationConfig(parallelization="threading", n_jobs=2)
        options = OFOptions(quality_setting="fast")

        # Setup for array processing
        options.input_file = video
        options.reference_frames = reference
        options.output_format = OutputFormat.ARRAY
        options.save_w = True
        options.save_meta_info = False

        # Create compensator
        compensator = BatchMotionCorrector(options, config)

        # Verify threading executor is used
        assert compensator.executor.__class__.__name__ == "ThreadingExecutor"

        # Run
        compensator.run()

        # Get results
        registered = compensator.video_writer.get_array()
        flow = compensator.w_writer.get_array() if compensator.w_writer else None

        # Verify results
        assert registered.shape == video.shape
        assert flow is not None and flow.shape == (T, H, W, 2)

    @pytest.mark.parametrize("n_channels", [1, 2, 3])
    def test_multiprocessing_executor(self, n_channels):
        """Test multiprocessing executor with different channel counts."""
        available = RuntimeContext.get("available_parallelization", set())

        if "multiprocessing" not in available:
            pytest.skip("Multiprocessing executor not available")

        # Create test data
        T, H, W, C = 10, 16, 16, n_channels
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:3], axis=0)

        # Configure for multiprocessing
        config = RegistrationConfig(parallelization="multiprocessing", n_jobs=2)
        options = OFOptions(quality_setting="fast")

        # Setup for array processing
        options.input_file = video
        options.reference_frames = reference
        options.output_format = OutputFormat.ARRAY
        options.save_w = True
        options.save_meta_info = False

        # Create compensator
        compensator = BatchMotionCorrector(options, config)

        # Verify multiprocessing executor is used
        assert compensator.executor.__class__.__name__ == "MultiprocessingExecutor"

        # Run
        compensator.run()

        # Get results
        registered = compensator.video_writer.get_array()
        flow = compensator.w_writer.get_array() if compensator.w_writer else None

        # Verify results
        assert registered.shape == video.shape
        assert flow is not None and flow.shape == (T, H, W, 2)

    @pytest.mark.parametrize("n_channels", [1, 2])
    def test_executor_consistency(self, n_channels):
        """Test that all executors produce consistent results with different channel counts."""
        available = RuntimeContext.get("available_parallelization", set())

        # Create test data
        T, H, W, C = 8, 16, 16, n_channels
        np.random.seed(42)  # Fixed seed for reproducibility
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        results = {}

        for executor_name in available:
            # Configure
            config = RegistrationConfig(parallelization=executor_name, n_jobs=2)
            options = OFOptions(quality_setting="fast", levels=2, iterations=5)

            # Setup for array processing
            options.input_file = video.copy()  # Copy to avoid side effects
            options.reference_frames = reference.copy()
            options.output_format = OutputFormat.ARRAY
            options.save_w = True
            options.save_meta_info = False

            # Create compensator
            compensator = BatchMotionCorrector(options, config)

            # Run
            compensator.run()

            # Store results
            results[executor_name] = compensator.video_writer.get_array()

        # Compare results (allowing small numerical differences)
        if len(results) > 1:
            reference_result = list(results.values())[0]
            for name, result in results.items():
                np.testing.assert_allclose(
                    result,
                    reference_result,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Results from {name} differ from reference",
                )


class TestExecutorRegistration:
    """
    Test executor registration to prevent regressions.

    This test class specifically guards against the bug introduced in commit
    cedb82eb where pre-commit hooks removed the critical import that triggers
    executor registration, breaking multiprocessing support in production
    while tests still passed.
    """

    def test_executors_registered_in_runtime_context(self):
        """Test that all expected executors are registered with RuntimeContext."""
        RuntimeContext.init(force=True)

        registry = RuntimeContext._config.get("parallelization_registry", {})

        # These executors must always be registered
        assert "sequential" in registry, "Sequential executor not registered"
        assert "threading" in registry, "Threading executor not registered"
        assert "multiprocessing" in registry, "Multiprocessing executor not registered"

        # Verify registry contains valid dotted paths
        for name, path in registry.items():
            assert isinstance(path, str), f"Executor {name} path is not string: {path}"
            assert "." in path, f"Executor {name} has invalid path: {path}"

    def test_get_parallelization_executor_returns_classes(self):
        """Test that RuntimeContext can retrieve executor classes."""
        RuntimeContext.init(force=True)

        for executor_name in ["sequential", "threading", "multiprocessing"]:
            executor_class = RuntimeContext.get_parallelization_executor(executor_name)
            assert executor_class is not None, f"Failed to get {executor_name} executor"
            assert callable(executor_class), f"{executor_name} executor is not callable"

    def test_compensate_recording_imports_parallelization(self):
        """
        CRITICAL: Verify compensate_recording.py contains the executor import.

        This import triggers executor registration and MUST NOT be removed by
        pre-commit hooks or code cleanup tools. It must have '# noqa: F401'
        to prevent automatic removal.
        """
        import inspect
        import pyflowreg.motion_correction.compensate_recording as cr_module

        source_file = inspect.getfile(cr_module)

        with open(source_file, "r") as f:
            content = f.read()

        # Check for the critical import
        assert "import pyflowreg.motion_correction.parallelization" in content, (
            "CRITICAL: Import missing from compensate_recording.py!\n"
            "This import triggers executor registration and must not be removed.\n"
            "See commit cedb82eb for the bug this test prevents."
        )

        # Verify it has noqa comment to prevent removal
        assert (
            "# noqa" in content
        ), "Import should have '# noqa: F401' to prevent pre-commit removal"

    @pytest.mark.integration
    def test_executors_available_without_explicit_import(self):
        """
        Test executors work when importing only compensate_recording.

        This simulates real-world usage and would have caught the cedb82eb bug
        where pre-commit hooks broke production while tests still passed.
        """
        import sys
        import subprocess

        test_script = """
import sys
from pyflowreg._runtime import RuntimeContext
from pyflowreg.motion_correction import compensate_recording

# User code should NOT need to import parallelization module explicitly
# It should be imported by compensate_recording

RuntimeContext.init()

# Verify executors are registered
registry = RuntimeContext._config.get("parallelization_registry", {})
assert "multiprocessing" in registry, f"Multiprocessing not in registry: {registry}"
assert "threading" in registry, f"Threading not in registry: {registry}"

# Verify we can get executor classes
mp_executor = RuntimeContext.get_parallelization_executor("multiprocessing")
assert mp_executor is not None, "Could not get multiprocessing executor"

print("SUCCESS: Executors available")
"""

        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
        )

        assert (
            result.returncode == 0
        ), f"Fresh process test failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "SUCCESS" in result.stdout


class TestProgressCallbacks:
    """Test progress callback functionality with parallelization."""

    def test_progress_callback_with_compensate_arr(self):
        """Test progress callback through compensate_arr interface."""
        # Create test data
        T, H, W, C = 15, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:3], axis=0)

        # Track progress
        progress_calls: List[Tuple[int, int]] = []

        def progress_callback(current: int, total: int):
            progress_calls.append((current, total))

        # Run with callback
        options = OFOptions(quality_setting="fast", buffer_size=5)
        registered, flow = compensate_arr(video, reference, options, progress_callback)

        # Verify callback was called
        assert len(progress_calls) > 0, "Progress callback was not called"

        # Check final progress
        final_current, final_total = progress_calls[-1]
        actual_frames_processed = registered.shape[0]
        assert (
            final_current == final_total
        ), f"Final progress {final_current} != final_total {final_total}"
        assert (
            final_current == actual_frames_processed
        ), f"Progress {final_current} != actual frames {actual_frames_processed}"
        assert final_total == T, f"Total frames {final_total} != input frames {T}"

        # Check monotonic increase
        for i in range(1, len(progress_calls)):
            assert (
                progress_calls[i][0] >= progress_calls[i - 1][0]
            ), "Progress should increase"

    def test_progress_with_different_executors(self):
        """Test progress callbacks work with each executor type."""
        available = RuntimeContext.get("available_parallelization", set())

        # Create test data
        T, H, W, C = 20, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:3], axis=0)

        for executor_name in available:
            progress_calls = []

            def progress_callback(current, total):
                progress_calls.append((current, total))

            # Configure
            config = RegistrationConfig(parallelization=executor_name, n_jobs=2)
            options = OFOptions(
                quality_setting="fast",
                buffer_size=5,  # Small batches for more progress updates
            )

            # Setup for array processing
            options.input_file = video
            options.reference_frames = reference
            options.output_format = OutputFormat.ARRAY
            options.save_w = True
            options.save_meta_info = False

            # Create compensator and register callback
            compensator = BatchMotionCorrector(options, config)
            compensator.register_progress_callback(progress_callback)

            # Run
            compensator.run()

            # Verify progress was tracked
            assert len(progress_calls) > 0, f"No progress for {executor_name}"

            # Check that final progress matches total - the actual processed frames may differ from input
            final_current, final_total = progress_calls[-1]
            assert (
                final_current == final_total
            ), f"Wrong final count for {executor_name}: {final_current} != {final_total}"

            # Different executors have different update patterns
            if executor_name in ["sequential", "threading"]:
                # Frame-by-frame updates possible
                print(f"{executor_name}: {len(progress_calls)} progress updates")
            else:  # multiprocessing
                # Batch-wise updates only
                print(f"{executor_name}: {len(progress_calls)} batch updates")

    def test_multiple_callbacks_registered(self):
        """Test that multiple callbacks can be registered and called."""
        # Create test data
        T, H, W, C = 10, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Track calls from multiple callbacks
        calls1 = []
        calls2 = []
        calls3 = []

        def callback1(current, total):
            calls1.append((current, total))

        def callback2(current, total):
            calls2.append((current, total))

        def callback3(current, total):
            # This one calculates percentage
            percent = (current / total * 100) if total > 0 else 0
            calls3.append(percent)

        # Setup
        options = OFOptions(quality_setting="fast")
        options.input_file = video
        options.reference_frames = reference
        options.output_format = OutputFormat.ARRAY
        options.save_w = True
        options.save_meta_info = False

        # Create compensator and register multiple callbacks
        compensator = BatchMotionCorrector(options)
        compensator.register_progress_callback(callback1)
        compensator.register_progress_callback(callback2)
        compensator.register_progress_callback(callback3)

        # Run
        compensator.run()

        # All callbacks should be called
        assert len(calls1) > 0, "First callback not called"
        assert len(calls2) > 0, "Second callback not called"
        assert len(calls3) > 0, "Third callback not called"

        # First two should have identical calls
        assert calls1 == calls2, "Callbacks should receive same calls"

        # Third should end at 100%
        final_percent = calls3[-1]
        assert (
            abs(final_percent - 100.0) < 0.01
        ), f"Final progress should be 100%, got {final_percent}%"

    def test_callback_exception_handling(self):
        """Test that exceptions in callbacks don't break processing."""
        # Create test data
        T, H, W, C = 8, 16, 16, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)

        # Track successful calls
        good_calls = []
        exception_raised = [False]

        def good_callback(current, total):
            good_calls.append((current, total))

        def bad_callback(current, total):
            # Raise exception on first call to test handling
            if not exception_raised[0]:
                exception_raised[0] = True
                raise ValueError("Test exception")

        # Setup
        options = OFOptions(quality_setting="fast")
        options.input_file = video
        options.reference_frames = reference
        options.output_format = OutputFormat.ARRAY
        options.save_w = True
        options.save_meta_info = False

        # Create compensator with both callbacks
        compensator = BatchMotionCorrector(options)
        compensator.register_progress_callback(good_callback)
        compensator.register_progress_callback(bad_callback)

        # Run - should complete despite exception
        with pytest.warns(UserWarning, match="Progress callback error"):
            compensator.run()

        # Good callback should still work
        assert len(good_calls) > 0, "Good callback should still be called"

        # Check that processing completed correctly
        final_current, final_total = good_calls[-1]
        assert (
            final_current == final_total
        ), f"Processing should complete: {final_current} != {final_total}"

        # Get results to verify processing completed
        registered = compensator.video_writer.get_array()
        assert registered.shape == video.shape

    def test_progress_callback_performance(self):
        """Test that callbacks don't significantly impact performance."""
        # Create test data
        T, H, W, C = 50, 32, 32, 2
        video = np.random.rand(T, H, W, C).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        import time

        # Time without callback
        options = OFOptions(quality_setting="fast", levels=2, iterations=3)
        start = time.time()
        registered1, flow1 = compensate_arr(video, reference, options)
        time_without = time.time() - start

        # Time with callback
        call_count = [0]

        def progress_callback(current, total):
            call_count[0] += 1

        start = time.time()
        registered2, flow2 = compensate_arr(
            video, reference, options, progress_callback
        )
        time_with = time.time() - start

        print(f"Time without callback: {time_without:.3f}s")
        print(f"Time with callback: {time_with:.3f}s ({call_count[0]} calls)")
        print(f"Overhead: {(time_with - time_without):.3f}s")

        # Callback overhead should be minimal (< 10% for this test size)
        assert time_with < time_without * 1.1, "Callback overhead too high"

        # Results should be identical
        np.testing.assert_allclose(registered1, registered2, rtol=1e-6)


if __name__ == "__main__":
    # Run basic tests
    print("Testing parallelization executors...")
    test_exec = TestParallelizationExecutors()
    test_exec.test_all_executors_available()
    test_exec.test_sequential_executor()

    print("\nTesting progress callbacks...")
    test_prog = TestProgressCallbacks()
    test_prog.test_progress_callback_with_compensate_arr()
    test_prog.test_multiple_callbacks_registered()

    print("\n✓ All parallelization tests passed!")
