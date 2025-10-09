"""
Tests for quickstart.md code examples.

This module contains manual mirror tests that validate all code examples
in the quickstart guide execute correctly using existing test fixtures.
"""

import pytest
import numpy as np

from pyflowreg.motion_correction import compensate_arr, compensate_recording, OFOptions
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig


class TestQuickstartImports:
    """Test that all imports in quickstart.md work correctly."""

    def test_array_workflow_imports(self):
        """Test imports for array-based workflow example."""
        from pyflowreg.motion_correction import compensate_arr, OFOptions

        assert callable(compensate_arr)
        assert OFOptions is not None

    def test_file_workflow_imports(self):
        """Test imports for file-based workflow example."""
        from pyflowreg.motion_correction import OFOptions

        assert callable(compensate_recording)
        assert OFOptions is not None

    def test_parallel_processing_imports(self):
        """Test imports for parallel processing configuration."""
        from pyflowreg.motion_correction import OFOptions
        from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

        assert callable(compensate_recording)
        assert OFOptions is not None
        assert RegistrationConfig is not None


class TestQuickstartArrayWorkflow:
    """Test the basic array-based workflow from quickstart.md."""

    def test_compensate_arr_basic_workflow(self, test_data_array, fast_of_options):
        """Test basic compensate_arr workflow from quickstart example."""
        # Simulate the quickstart example workflow
        video, _ = test_data_array  # Unpack tuple (array, shape)
        reference = np.mean(video[:10], axis=0)

        # Configure motion correction
        options = fast_of_options  # Using fast for testing speed

        # Run compensation
        registered, flow = compensate_arr(video, reference, options)

        # Verify outputs
        assert registered.shape == video.shape
        assert flow.shape == (video.shape[0], video.shape[1], video.shape[2], 2)

    @pytest.mark.parametrize("quality_setting", ["fast", "balanced", "quality"])
    def test_compensate_arr_quality_settings(self, test_data_array, quality_setting):
        """Test different quality settings from quickstart guide."""
        video, _ = test_data_array  # Unpack tuple (array, shape)
        reference = np.mean(video[:10], axis=0)

        # Configure with different quality settings
        options = OFOptions(quality_setting=quality_setting)

        # Run compensation
        registered, flow = compensate_arr(video, reference, options)

        # Verify outputs have correct shapes
        assert registered.shape == video.shape
        assert flow.shape == (video.shape[0], video.shape[1], video.shape[2], 2)


class TestQuickstartFileWorkflow:
    """Test the file-based workflow from quickstart.md."""

    def test_compensate_recording_options_creation(self, tmp_path, small_test_video):
        """Test OFOptions configuration for file-based workflow."""
        # small_test_video is already a file path, not an array
        video_path, shape = small_test_video

        # Configure options as in quickstart example
        options = OFOptions(
            input_file=video_path,
            output_path=str(tmp_path / "results"),
            output_format="HDF5",
            quality_setting="fast",  # Use fast for testing
            reference_frames=list(range(min(10, shape[0]))),
            save_w=True,
        )

        # Verify options are correctly set
        assert options.input_file == video_path
        assert options.output_format.value == "HDF5"
        assert options.save_w is True


class TestQuickstartParallelProcessing:
    """Test parallel processing configuration from quickstart.md."""

    @pytest.mark.parametrize("parallelization", ["sequential", "threading"])
    def test_registration_config_creation(self, parallelization):
        """Test RegistrationConfig creation with different backends."""
        # Manual executor selection as shown in quickstart
        config = RegistrationConfig(n_jobs=-1, parallelization=parallelization)

        # Verify config is correctly set
        assert config.n_jobs == -1
        assert config.parallelization == parallelization

    def test_registration_config_auto_selection(self):
        """Test auto-selection of parallelization backend."""
        # Default behavior - auto-select best available backend
        config = RegistrationConfig(n_jobs=-1)

        # Verify config uses default auto-selection
        assert config.n_jobs == -1
        # parallelization will be auto-selected
