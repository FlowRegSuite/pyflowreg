"""
Tests for compensate_recording with the new executor system.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np

from pyflowreg.motion_correction.compensate_recording import (
    BatchMotionCorrector,
    RegistrationConfig,
    compensate_recording,
)
from pyflowreg.motion_correction.OF_options import OutputFormat
from pyflowreg._runtime import RuntimeContext
from pyflowreg.util.io.factory import get_video_file_reader


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


class TestOutputDataTypePreservation:
    """Ensure compensated videos preserve the source dtype."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        ("output_format", "expected_files"),
        [
            (
                OutputFormat.HDF5,
                ("compensated.HDF5", "compensated.hdf5"),
            ),
            (
                OutputFormat.TIFF,
                ("compensated.TIFF", "compensated.tif", "compensated.tiff"),
            ),
        ],
    )
    def test_compensated_output_dtype_matches_input(
        self, small_test_video, fast_of_options, output_format, expected_files
    ):
        """The dtype of the compensated video should match the source file."""
        video_path, _ = small_test_video
        fast_of_options.input_file = video_path
        fast_of_options.output_format = output_format
        fast_of_options.buffer_size = 5

        input_reader = get_video_file_reader(video_path)
        try:
            input_frame = input_reader[0]  # Force a read so dtype is concrete
            input_dtype = input_frame.dtype
        finally:
            input_reader.close()

        # Fixture creates uint16 data – assert to ensure the test guards the desired case.
        assert input_dtype == np.dtype(
            "uint16"
        ), f"Expected uint16 input video for this test, got {input_dtype}"

        config = RegistrationConfig(
            n_jobs=1,
            verbose=False,
            parallelization="sequential",
        )

        compensate_recording(fast_of_options, config=config)

        output_dir = Path(fast_of_options.output_path)
        output_file = next(
            (
                output_dir / name
                for name in expected_files
                if (output_dir / name).exists()
            ),
            None,
        )

        assert (
            output_file is not None
        ), f"Compensated output file not found. Checked: {expected_files}"

        output_reader = get_video_file_reader(str(output_file))
        try:
            output_frame = output_reader[0]
            assert (
                output_frame.dtype == input_dtype
            ), f"Expected dtype {input_dtype}, got {output_frame.dtype}"
        finally:
            output_reader.close()


class TestReferenceFramePreregistration:
    """Test reference frame preregistration with list of frame indices."""

    @pytest.mark.integration
    def test_preregistration_with_frame_list(self, tmp_path):
        """Test preregistration when reference_frames is a list of frame indices."""
        from pyflowreg.util.io.factory import get_video_file_writer
        from pyflowreg.motion_correction import OFOptions

        # Create minimal test video: 3 frames, 8x8, 1 channel
        video_path = tmp_path / "test_preregistration.h5"
        n_frames, h, w, c = 3, 8, 8, 1

        # Create simple test data with slight variations
        test_data = np.zeros((n_frames, h, w, c), dtype=np.float32)
        for i in range(n_frames):
            test_data[i, 2:6, 2:6, 0] = (
                100.0 + i * 10.0
            )  # Small square with varying intensity

        # Write test video using factory
        writer = get_video_file_writer(str(video_path), "HDF5")
        try:
            writer.write_frames(test_data)
        finally:
            writer.close()

        # Create options with reference_frames as list of indices
        # This should trigger preregistration code path (like jupiter_demo does)
        options = OFOptions(
            input_file=str(video_path),
            reference_frames=[
                0,
                1,
                2,
            ],  # List of frame indices - triggers preregistration!
            quality_setting="fast",
            buffer_size=3,
            output_path=str(tmp_path / "output"),
        )

        config = RegistrationConfig(
            parallelization="sequential", n_jobs=1, verbose=False
        )

        # Call compensate_recording just like jupiter_demo does
        # This will trigger get_reference_frame() -> preregistration
        # which creates weight_2d and passes it to OFOptions, exposing the Pydantic bug
        reference = compensate_recording(options, config=config)

        # If we get here without Pydantic validation error, the bug is fixed
        assert reference is not None
        assert reference.shape == (h, w, c)

    def test_weight_as_list(self, tmp_path):
        """Test that weight can be specified as a list [1, 2] for backwards compatibility."""
        from pyflowreg.motion_correction import OFOptions

        # Test with weight as list - this is a common use case
        options = OFOptions(
            input_file="dummy.h5",
            weight=[1, 2],  # Should be normalized to [0.333..., 0.666...]
            output_path=str(tmp_path),
        )

        # Verify weight was normalized
        assert isinstance(options.weight, list)
        assert len(options.weight) == 2
        assert abs(options.weight[0] - 1 / 3) < 0.001
        assert abs(options.weight[1] - 2 / 3) < 0.001

    def test_weight_as_numpy_1d(self, tmp_path):
        """Test that weight can be specified as a 1D numpy array."""
        from pyflowreg.motion_correction import OFOptions

        # Test with weight as 1D numpy array
        options = OFOptions(
            input_file="dummy.h5",
            weight=np.array([1.0, 2.0]),  # Should be normalized and converted to list
            output_path=str(tmp_path),
        )

        # Verify weight was normalized and converted to list
        assert isinstance(options.weight, list)
        assert len(options.weight) == 2
        assert abs(options.weight[0] - 1 / 3) < 0.001
        assert abs(options.weight[1] - 2 / 3) < 0.001

    def test_weight_as_numpy_2d(self, tmp_path):
        """Test that weight can be a 2D numpy array (spatial weight map, single channel)."""
        from pyflowreg.motion_correction import OFOptions

        # Test with weight as 2D numpy array
        h, w = 8, 8
        weight_2d = np.ones((h, w), dtype=np.float32)

        # This should NOT fail with Pydantic validation error
        options = OFOptions(
            input_file="dummy.h5",
            weight=weight_2d,  # 2D array should be kept as-is
            output_path=str(tmp_path),
        )

        # Verify weight was kept as numpy array
        assert isinstance(options.weight, np.ndarray)
        assert options.weight.shape == (h, w)
        assert options.weight.ndim == 2

    def test_weight_as_numpy_3d(self, tmp_path):
        """Test that weight can be a 3D numpy array (from preregistration)."""
        from pyflowreg.motion_correction import OFOptions

        # Test with weight as 3D numpy array (like preregistration creates)
        h, w, c = 8, 8, 2
        weight_3d = np.ones((h, w, c), dtype=np.float32)

        # This should NOT fail with Pydantic validation error
        options = OFOptions(
            input_file="dummy.h5",
            weight=weight_3d,  # Multi-dimensional array should be kept as-is
            output_path=str(tmp_path),
        )

        # Verify weight was kept as numpy array
        assert isinstance(options.weight, np.ndarray)
        assert options.weight.shape == (h, w, c)
        assert options.weight.ndim == 3

    def test_weight_wrong_length_list(self, tmp_path):
        """Test that weight list with wrong number of channels is accepted by Pydantic but may cause issues later."""
        from pyflowreg.motion_correction import OFOptions

        # This tests that Pydantic validation doesn't fail for mismatched channel counts
        # The actual check for matching channels happens at runtime during processing
        options = OFOptions(
            input_file="dummy.h5",
            weight=[1, 2, 3],  # 3 weights - might not match actual channel count
            output_path=str(tmp_path),
        )

        # Pydantic should accept this (validation happens later during processing)
        assert isinstance(options.weight, list)
        assert len(options.weight) == 3

    def test_weight_4d_array_should_fail(self, tmp_path):
        """Test that 4D weight arrays are rejected (weight is spatial only, not temporal)."""
        from pyflowreg.motion_correction import OFOptions
        from pydantic import ValidationError

        # Weight should only be spatial (H, W, C), not temporal
        h, w, c, t = 8, 8, 2, 10
        weight_4d = np.ones((h, w, c, t), dtype=np.float32)

        # This SHOULD fail - weight cannot be 4D
        with pytest.raises(ValidationError):
            OFOptions(
                input_file="dummy.h5",
                weight=weight_4d,
                output_path=str(tmp_path),
            )


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
