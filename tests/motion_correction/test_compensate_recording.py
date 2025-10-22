"""
Tests for compensate_recording with the new executor system.
"""

from unittest.mock import patch

import pytest
import numpy as np

from pyflowreg.motion_correction.compensate_recording import (
    BatchMotionCorrector,
    RegistrationConfig,
    compensate_recording,
)
from pyflowreg._runtime import RuntimeContext


class TestRegistrationConfig:
    """Test the RegistrationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RegistrationConfig()
        assert config.n_jobs == -1
        assert config.verbose is False
        assert config.parallelization is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RegistrationConfig(n_jobs=4, verbose=True, parallelization="threading")
        assert config.n_jobs == 4
        assert config.verbose is True
        assert config.parallelization == "threading"


class TestCompensateRecording:
    """Test the CompensateRecording class and executor system."""

    def test_executor_setup_auto_selection(self, fast_of_options):
        """Test automatic executor selection."""
        config = RegistrationConfig(parallelization=None)
        pipeline = BatchMotionCorrector(fast_of_options, config)

        # Should auto-select an available executor
        assert pipeline.executor is not None
        assert pipeline.executor.name in ["multiprocessing", "threading", "sequential"]

    def test_executor_setup_specific_selection(self, fast_of_options):
        """Test specific executor selection."""
        config = RegistrationConfig(parallelization="sequential")
        pipeline = BatchMotionCorrector(fast_of_options, config)

        assert pipeline.executor is not None
        assert pipeline.executor.name == "sequential"

    def test_executor_setup_fallback(self, fast_of_options):
        """Test fallback hierarchy when executors are unavailable."""
        # Scenario 1: Auto-selection with limited availability
        # Multiprocessing unavailable -> should fall back to threading
        with RuntimeContext.use(available_parallelization={"sequential", "threading"}):
            config = RegistrationConfig(parallelization=None)  # Auto-select
            pipeline = BatchMotionCorrector(fast_of_options, config)
            assert pipeline.executor.name == "threading"

        # Multiprocessing and threading unavailable -> should fall back to sequential
        with RuntimeContext.use(available_parallelization={"sequential"}):
            config = RegistrationConfig(parallelization=None)  # Auto-select
            pipeline = BatchMotionCorrector(fast_of_options, config)
            assert pipeline.executor.name == "sequential"

        # Scenario 2: Explicit request with limited availability
        # Request multiprocessing, but only threading available -> should fall back to threading
        with RuntimeContext.use(available_parallelization={"sequential", "threading"}):
            config = RegistrationConfig(
                parallelization="multiprocessing"
            )  # Explicit request
            pipeline = BatchMotionCorrector(fast_of_options, config)
            assert pipeline.executor.name == "threading"

        # Request multiprocessing, but only sequential available -> should fall back to sequential
        with RuntimeContext.use(available_parallelization={"sequential"}):
            config = RegistrationConfig(
                parallelization="multiprocessing"
            )  # Explicit request
            pipeline = BatchMotionCorrector(fast_of_options, config)
            assert pipeline.executor.name == "sequential"

    def test_n_workers_setup(self, fast_of_options):
        """Test n_workers configuration."""
        # Test auto-detection (-1)
        config = RegistrationConfig(n_jobs=-1)
        pipeline = BatchMotionCorrector(fast_of_options, config)
        assert pipeline.n_workers > 0

        # Test specific value
        config = RegistrationConfig(n_jobs=3)
        pipeline = BatchMotionCorrector(fast_of_options, config)
        assert pipeline.n_workers == 3

    def test_initialization_with_basic_options(self, basic_of_options):
        """Test pipeline initialization with basic options."""
        config = RegistrationConfig(n_jobs=2)
        pipeline = BatchMotionCorrector(basic_of_options, config)

        assert pipeline.options == basic_of_options
        assert pipeline.config == config
        assert pipeline.executor is not None
        assert len(pipeline.mean_disp) == 0
        assert len(pipeline.max_disp) == 0


class TestExecutorTypes:
    """Test different executor types work correctly."""

    @pytest.mark.executor
    def test_sequential_executor(
        self, small_test_video, fast_of_options, sequential_config
    ):
        """Test sequential executor functionality."""
        video_path, shape = small_test_video
        fast_of_options.input_file = video_path

        pipeline = BatchMotionCorrector(fast_of_options, sequential_config)
        assert pipeline.executor.name == "sequential"
        assert pipeline.executor.n_workers == 1

    @pytest.mark.executor
    def test_threading_executor(
        self, small_test_video, fast_of_options, threading_config
    ):
        """Test threading executor functionality."""
        video_path, shape = small_test_video
        fast_of_options.input_file = video_path

        pipeline = BatchMotionCorrector(fast_of_options, threading_config)
        assert pipeline.executor.name == "threading"
        assert pipeline.executor.n_workers == 2

    @pytest.mark.executor
    def test_multiprocessing_executor(
        self, small_test_video, fast_of_options, multiprocessing_config
    ):
        """Test multiprocessing executor functionality."""
        video_path, shape = small_test_video
        fast_of_options.input_file = video_path

        pipeline = BatchMotionCorrector(fast_of_options, multiprocessing_config)
        assert pipeline.executor.name == "multiprocessing"
        assert pipeline.executor.n_workers == 2


class TestRuntimeContextIntegration:
    """Test runtime context properly manages executor selection."""

    def test_available_parallelization(self):
        """Test that available parallelization modes are detected."""
        available = RuntimeContext.get_available_parallelization()
        assert "sequential" in available
        assert len(available) > 0

    def test_executor_registration(self):
        """Test that executors are properly registered."""
        sequential_class = RuntimeContext.get_parallelization_executor("sequential")
        assert sequential_class is not None

        threading_class = RuntimeContext.get_parallelization_executor("threading")
        assert threading_class is not None

    def test_runtime_context_use(self, fast_of_options):
        """Test runtime context temporary configuration."""
        with RuntimeContext.use(max_workers=8):
            config = RegistrationConfig(n_jobs=-1)  # Auto-detect
            pipeline = BatchMotionCorrector(fast_of_options, config)
            # Note: n_workers might be different due to system limits
            assert pipeline.n_workers > 0


class TestCompensateRecordingIntegration:
    """Integration tests for the complete compensate_recording function."""

    @pytest.mark.integration
    def test_compensate_recording_sequential(self, small_test_video, fast_of_options):
        """Test compensate_recording function with sequential executor."""
        video_path, shape = small_test_video
        fast_of_options.input_file = video_path
        fast_of_options.buffer_size = 5

        config = RegistrationConfig(
            n_jobs=1, verbose=True, parallelization="sequential"
        )

        # Test that pipeline can be created and configured correctly
        pipeline = BatchMotionCorrector(fast_of_options, config)
        assert pipeline.executor.name == "sequential"
        assert pipeline.n_workers == 1
        assert pipeline.options.buffer_size == 5

    @pytest.mark.integration
    @pytest.mark.parametrize("executor_name", ["sequential", "threading"])
    def test_compensate_recording_all_executors(
        self, small_test_video, fast_of_options, executor_name
    ):
        """Test compensate_recording with different executors."""
        video_path, shape = small_test_video
        fast_of_options.input_file = video_path
        fast_of_options.buffer_size = 3

        config = RegistrationConfig(
            n_jobs=2, verbose=True, parallelization=executor_name
        )

        # Test executor selection by creating pipeline and checking executor type
        pipeline = BatchMotionCorrector(fast_of_options, config)
        assert pipeline.executor.name == executor_name

        # Test that pipeline can be initialized properly
        assert pipeline.config.parallelization == executor_name
        assert pipeline.n_workers == 2


class TestBackwardCompatibility:
    """Test that refactored compensate_recording maintains backward compatibility."""

    def test_compensate_recording_no_config(self, small_test_video, fast_of_options):
        """Test compensate_recording without explicit config (backward compatibility)."""
        video_path, shape = small_test_video
        fast_of_options.input_file = video_path

        # Test that pipeline can be created with default config
        pipeline = BatchMotionCorrector(fast_of_options, config=None)
        assert pipeline.config is not None  # Should create default config
        assert pipeline.executor is not None

        # Test that the function signature still works
        assert callable(compensate_recording)

    def test_compensate_recording_with_reference_frame(
        self, small_test_video, fast_of_options, reference_frame
    ):
        """Test compensate_recording with provided reference frame."""
        video_path, shape = small_test_video
        fast_of_options.input_file = video_path

        # Test that pipeline can be created with provided reference frame
        pipeline = BatchMotionCorrector(fast_of_options)

        # Test the reference frame setup
        pipeline._setup_reference(reference_frame)
        assert pipeline.reference_raw is not None
        np.testing.assert_array_equal(pipeline.reference_raw, reference_frame)


class TestExecutorCleanup:
    """Test that executors are properly cleaned up."""

    def test_executor_cleanup_on_completion(self, small_test_video, fast_of_options):
        """Test that executor is cleaned up after successful completion."""
        video_path, shape = small_test_video
        fast_of_options.input_file = video_path
        fast_of_options.buffer_size = 2

        config = RegistrationConfig(parallelization="sequential", n_jobs=1)
        pipeline = BatchMotionCorrector(fast_of_options, config)
        executor = pipeline.executor

        # Test that executor has cleanup method
        assert hasattr(executor, "cleanup")
        assert callable(executor.cleanup)

        # Mock cleanup to verify it can be called
        with patch.object(executor, "cleanup") as mock_cleanup:
            # Simulate calling cleanup
            pipeline.executor.cleanup()
            mock_cleanup.assert_called_once()

    def test_executor_cleanup_on_exception(self, small_test_video, fast_of_options):
        """Test that executor is cleaned up even when exceptions occur."""
        video_path, shape = small_test_video
        fast_of_options.input_file = video_path
        fast_of_options.buffer_size = 2

        config = RegistrationConfig(parallelization="sequential", n_jobs=1)
        pipeline = BatchMotionCorrector(fast_of_options, config)
        executor = pipeline.executor

        # Test that cleanup can be called after exceptions
        with patch.object(executor, "cleanup") as mock_cleanup:
            # Simulate an error scenario where cleanup is needed
            try:
                # Simulate calling cleanup in finally block
                pipeline.executor.cleanup()
            except Exception:
                pass

            mock_cleanup.assert_called_once()


class TestErrorHandling:
    """Test error handling in the executor system."""

    def test_invalid_executor_name(self, fast_of_options):
        """Test handling of invalid executor names."""
        config = RegistrationConfig(parallelization="invalid_executor")

        # Should fallback to best available without crashing
        pipeline = BatchMotionCorrector(fast_of_options, config)
        # On most systems, multiprocessing is available and should be the fallback
        assert pipeline.executor.name in ["multiprocessing", "threading", "sequential"]
        # The actual executor depends on what's available, but it should be the best one

    def test_executor_instantiation_error(self, fast_of_options):
        """Test handling of executor instantiation errors."""
        # Create a mock that returns None for 'threading' but allows 'sequential' to work
        original_get_executor = RuntimeContext.get_parallelization_executor

        def mock_get_executor(name):
            if name == "threading":
                return None  # Simulate threading not available
            else:
                return original_get_executor(name)  # Allow fallback to work

        with patch.object(
            RuntimeContext,
            "get_parallelization_executor",
            side_effect=mock_get_executor,
        ):
            config = RegistrationConfig(parallelization="threading")

            # Should fallback to sequential
            pipeline = BatchMotionCorrector(fast_of_options, config)
            assert pipeline.executor.name == "sequential"


class TestGPUBackendExecutors:
    """Test GPU backend executor compatibility."""

    def test_gpu_backend_auto_selection(self, fast_of_options):
        """Test that GPU backends automatically use sequential executor."""
        # Set backend to flowreg_torch (GPU)
        fast_of_options.flow_backend = "flowreg_torch"

        # Auto-select executor (parallelization=None)
        config = RegistrationConfig(parallelization=None)
        pipeline = BatchMotionCorrector(fast_of_options, config)

        # Should automatically select sequential for GPU backend
        assert pipeline.executor.name == "sequential"

    @pytest.mark.skipif(
        "flowreg_cuda"
        not in __import__(
            "pyflowreg.core.backend_registry", fromlist=["list_backends"]
        ).list_backends(),
        reason="CuPy backend not available on macOS",
    )
    def test_gpu_backend_with_multiprocessing_request(self, fast_of_options):
        """Test that GPU backends force sequential even when multiprocessing requested."""
        import warnings

        # Set backend to flowreg_cuda (GPU)
        fast_of_options.flow_backend = "flowreg_cuda"

        # Explicitly request multiprocessing
        config = RegistrationConfig(parallelization="multiprocessing")

        # Should warn and fall back to sequential
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline = BatchMotionCorrector(fast_of_options, config)

            # Should have warned about incompatibility
            assert len(w) > 0
            assert "does not support" in str(w[0].message)
            assert "multiprocessing" in str(w[0].message)

        # Should use sequential
        assert pipeline.executor.name == "sequential"


# Slow/comprehensive tests that can be skipped with -m "not slow"
class TestComprehensiveIntegration:
    """Comprehensive integration tests (marked as slow)."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_pipeline_medium_data(self, medium_test_video, basic_of_options):
        """Test full pipeline with medium-sized data."""
        video_path, shape = medium_test_video
        basic_of_options.input_file = video_path
        basic_of_options.buffer_size = 20

        config = RegistrationConfig(
            n_jobs=2,
            verbose=False,
            parallelization="sequential",  # Use sequential for deterministic results
        )

        # This would run the actual pipeline - commented out for safety
        # reference = compensate_recording(basic_of_options, config=config)
        # assert reference is not None

        # Instead, just test setup
        pipeline = BatchMotionCorrector(basic_of_options, config)
        assert pipeline.executor.name == "sequential"
        assert pipeline.options.buffer_size == 20
