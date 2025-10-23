"""
Tests for OF_options configuration class.
"""

import pytest
import numpy as np

from pyflowreg.motion_correction.OF_options import OFOptions


class TestWeightValidation:
    """Test weight field validation with different input types."""

    def test_weight_as_list(self, tmp_path):
        """Test that weight can be specified as a list [1, 2] for backwards compatibility."""
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

    def test_weight_4d_array_should_fail(self, tmp_path):
        """Test that 4D weight arrays are rejected (weight is spatial only, not temporal)."""
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


class TestExampleConfigurations:
    """Test that all example configurations can be created without errors."""

    def test_jupiter_demo_config(self, tmp_path):
        """Test configuration from examples/jupiter_demo.py"""
        options = OFOptions(
            input_file="dummy.h5",
            output_path=str(tmp_path),
            output_format="HDF5",
            alpha=4,
            quality_setting="balanced",
            output_typename="",
            reference_frames=list(
                range(100, 201)
            ),  # This would trigger preregistration
        )

        assert options.alpha == (4.0, 4.0)
        assert options.quality_setting.value == "balanced"
        assert options.reference_frames == list(range(100, 201))

    def test_jupiter_demo_arr_config(self, tmp_path):
        """Test configuration from examples/jupiter_demo_arr.py"""
        options = OFOptions(
            input_file="dummy.h5",
            output_path=str(tmp_path),
            alpha=4,
            quality_setting="balanced",
            levels=100,
            iterations=50,
            eta=0.8,
            save_w=True,
            output_typename="double",
        )

        assert options.alpha == (4.0, 4.0)
        assert options.levels == 100
        assert options.iterations == 50
        assert options.save_w is True

    def test_jupiter_demo_live_config(self, tmp_path):
        """Test configuration from examples/jupiter_demo_live.py"""
        from pyflowreg.motion_correction.OF_options import QualitySetting

        options = OFOptions(
            input_file="dummy.h5",
            output_path=str(tmp_path),
            alpha=4,
            quality_setting=QualitySetting.FAST,
            sigma=[[2.0, 2.0, 0.5], [2.0, 2.0, 0.5]],
            levels=100,
            iterations=50,
            eta=0.8,
            channel_normalization="separate",
        )

        assert options.alpha == (4.0, 4.0)
        assert options.quality_setting == QualitySetting.FAST
        assert options.levels == 100

    def test_synth_evaluation_configs(self, tmp_path):
        """Test configurations from examples/synth_evaluation.py"""
        # First config with 1D numpy array weights
        options1 = OFOptions(
            input_file="dummy.h5",
            output_path=str(tmp_path),
            alpha=(2, 2),
            levels=50,
            min_level=5,
            iterations=50,
            a_data=0.45,
            a_smooth=1,
            weight=np.array([0.6, 0.4]),
        )

        assert options1.alpha == (2.0, 2.0)
        assert isinstance(options1.weight, list)  # Should be converted to list
        assert len(options1.weight) == 2

        # Second config
        options2 = OFOptions(
            input_file="dummy.h5",
            output_path=str(tmp_path),
            alpha=(8, 8),
            iterations=100,
            a_data=0.45,
            a_smooth=1.0,
            weight=np.array([0.5, 0.5], np.float32),
            levels=50,
            eta=0.8,
            update_lag=5,
        )

        assert options2.alpha == (8.0, 8.0)
        assert options2.iterations == 100

    def test_jupyter_notebook_config(self, tmp_path):
        """Test configuration from notebooks/jupiter_demo.ipynb"""
        options = OFOptions(
            input_file="dummy.h5",
            output_path=str(tmp_path),
            output_format="HDF5",
            alpha=4,
            min_level=3,
            bin_size=1,
            buffer_size=500,
            reference_frames=list(range(100, 201)),  # Would trigger preregistration
            save_meta_info=True,
            save_w=False,
        )

        assert options.alpha == (4.0, 4.0)
        assert options.min_level == 3
        assert options.buffer_size == 500
        assert options.save_w is False


class TestGetWeightAt:
    """Test the get_weight_at method with different weight configurations."""

    def test_get_weight_at_with_1d_list(self, tmp_path):
        """Test get_weight_at with 1D list weights."""
        options = OFOptions(
            input_file="dummy.h5",
            weight=[1, 2],  # 2 channels with weights 1:2
            output_path=str(tmp_path),
        )

        # Get weights for 2 channels
        w0 = options.get_weight_at(0, n_channels=2)
        w1 = options.get_weight_at(1, n_channels=2)

        # Should return floats, normalized
        assert isinstance(w0, float)
        assert isinstance(w1, float)
        assert abs(w0 - 1 / 3) < 0.001
        assert abs(w1 - 2 / 3) < 0.001

    def test_get_weight_at_with_2d_array(self, tmp_path):
        """Test get_weight_at with 2D spatial weight map."""
        # Create 2D weight map (H, W) for single channel
        h, w = 8, 8
        weight_2d = np.ones((h, w), dtype=np.float32) * 0.5

        options = OFOptions(
            input_file="dummy.h5",
            weight=weight_2d,
            output_path=str(tmp_path),
        )

        # Get weight for channel 0
        w0 = options.get_weight_at(0, n_channels=1)

        # Should return 2D array
        assert isinstance(w0, np.ndarray)
        assert w0.shape == (h, w)
        assert np.allclose(w0, 0.5)

    def test_get_weight_at_with_3d_array(self, tmp_path):
        """Test get_weight_at with 3D spatial weight map (H, W, C)."""
        # Create 3D weight map (H, W, C) - channel-last format
        h, w, c = 8, 8, 2
        weight_3d = np.ones((h, w, c), dtype=np.float32)
        weight_3d[:, :, 0] = 0.3  # Channel 0 weight
        weight_3d[:, :, 1] = 0.7  # Channel 1 weight

        options = OFOptions(
            input_file="dummy.h5",
            weight=weight_3d,
            output_path=str(tmp_path),
        )

        # Get weights for both channels
        w0 = options.get_weight_at(0, n_channels=2)
        w1 = options.get_weight_at(1, n_channels=2)

        # Should return 2D arrays with correct values
        assert isinstance(w0, np.ndarray)
        assert isinstance(w1, np.ndarray)
        assert w0.shape == (h, w)
        assert w1.shape == (h, w)
        assert np.allclose(w0, 0.3)
        assert np.allclose(w1, 0.7)
