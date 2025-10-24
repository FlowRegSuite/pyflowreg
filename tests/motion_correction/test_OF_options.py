"""
Tests for OF_options configuration class.

Tests validation, normalization, and edge cases for optical flow options.
"""

import pytest
import numpy as np

from pyflowreg.motion_correction.OF_options import (
    OFOptions,
    QualitySetting,
)


class TestWeightValidation:
    """Test weight parameter validation and normalization."""

    def test_weight_1d_list(self):
        """Test 1D list of weights is normalized."""
        opts = OFOptions(weight=[0.3, 0.7])
        # Should be normalized to sum to 1
        assert isinstance(opts.weight, list)
        assert len(opts.weight) == 2
        assert np.isclose(sum(opts.weight), 1.0)

    def test_weight_1d_array(self):
        """Test 1D numpy array is converted to list and normalized."""
        opts = OFOptions(weight=np.array([0.3, 0.7]))
        # Should be converted to list and normalized
        assert isinstance(opts.weight, list)
        assert len(opts.weight) == 2
        assert np.isclose(sum(opts.weight), 1.0)

    def test_weight_2d_array_spatial(self):
        """Test 2D spatial weight array is preserved as numpy array."""
        # Create a 2D spatial weight map (H, W)
        H, W = 64, 64
        weight_2d = np.ones((H, W), dtype=np.float64)

        opts = OFOptions(weight=weight_2d)

        # Should be kept as numpy array, not converted to nested list
        assert isinstance(opts.weight, np.ndarray)
        assert opts.weight.shape == (H, W)
        assert opts.weight.ndim == 2

    def test_weight_3d_array_spatial(self):
        """Test 3D spatial weight array is preserved as numpy array."""
        # Create a 3D spatial weight map (H, W, C)
        H, W, C = 32, 32, 2
        weight_3d = np.random.rand(H, W, C).astype(np.float64)

        opts = OFOptions(weight=weight_3d)

        # Should be kept as numpy array
        assert isinstance(opts.weight, np.ndarray)
        assert opts.weight.shape == (H, W, C)
        assert opts.weight.ndim == 3

    def test_weight_4d_array_should_fail(self):
        """Test that 4D weight arrays are rejected (weight is spatial only, not temporal)."""
        from pydantic import ValidationError

        # Weight should only be spatial (H, W, C), not temporal
        h, w, c, t = 8, 8, 2, 10
        weight_4d = np.ones((h, w, c, t), dtype=np.float32)

        # This SHOULD fail - weight cannot be 4D
        with pytest.raises(ValidationError):
            OFOptions(weight=weight_4d)

    def test_weight_default(self):
        """Test default weight value."""
        opts = OFOptions()
        assert isinstance(opts.weight, list)
        assert len(opts.weight) == 2
        assert np.isclose(sum(opts.weight), 1.0)


class TestGetReferenceFrameWithSpatialWeights:
    """Test get_reference_frame with multi-frame preregistration."""

    def test_preregistration_creates_spatial_weights(self, tmp_path):
        """Test that multi-frame preregistration creates valid OFOptions with spatial weights."""
        import tifffile

        # Create synthetic test data
        T, H, W, C = 10, 64, 64, 1
        frames = (np.random.rand(T, H, W, C) * 1000).astype(np.uint16)

        # Save as TIFF
        video_path = tmp_path / "test_video.tif"
        tifffile.imwrite(str(video_path), frames)

        # Create OFOptions with multi-frame reference
        opts = OFOptions(
            input_file=str(video_path),
            reference_frames=list(range(T)),  # Use all frames for preregistration
            quality_setting=QualitySetting.FAST,
        )

        # Get video reader
        reader = opts.get_video_reader()

        # This should NOT raise a validation error
        # The internal call creates OFOptions(weight=weight_2d) where weight_2d is (H,W,C)
        reference = opts.get_reference_frame(reader)

        # Should return a valid reference
        assert reference.shape == (H, W, C)


class TestAlphaValidation:
    """Test alpha parameter validation."""

    def test_alpha_scalar(self):
        """Test scalar alpha is converted to tuple."""
        opts = OFOptions(alpha=2.5)
        assert opts.alpha == (2.5, 2.5)

    def test_alpha_tuple(self):
        """Test tuple alpha is preserved."""
        opts = OFOptions(alpha=(1.5, 2.5))
        assert opts.alpha == (1.5, 2.5)

    def test_alpha_list_single(self):
        """Test single-element list alpha."""
        opts = OFOptions(alpha=[3.0])
        assert opts.alpha == (3.0, 3.0)

    def test_alpha_list_two(self):
        """Test two-element list alpha."""
        opts = OFOptions(alpha=[1.5, 2.5])
        assert opts.alpha == (1.5, 2.5)

    def test_alpha_negative_raises(self):
        """Test negative alpha raises error."""
        with pytest.raises(ValueError, match="Alpha must be positive"):
            OFOptions(alpha=-1.0)


class TestSigmaValidation:
    """Test sigma parameter validation."""

    def test_sigma_1d_converted_to_2d(self):
        """Test 1D sigma is converted to 2D."""
        opts = OFOptions(sigma=[1.0, 1.0, 0.5])
        assert isinstance(opts.sigma, list)
        assert len(opts.sigma) == 1
        assert len(opts.sigma[0]) == 3

    def test_sigma_2d_preserved(self):
        """Test 2D sigma is preserved."""
        sigma = [[1.0, 1.0, 0.5], [2.0, 2.0, 1.0]]
        opts = OFOptions(sigma=sigma)
        assert opts.sigma == sigma

    def test_sigma_wrong_length_raises(self):
        """Test wrong sigma length raises error."""
        with pytest.raises(ValueError, match="1D sigma must be"):
            OFOptions(sigma=[1.0, 1.0])  # Missing temporal component


class TestQualitySettingEffectiveMinLevel:
    """Test quality setting affects effective_min_level."""

    def test_quality_effective_min_level(self):
        """Test quality setting gives min_level=0."""
        opts = OFOptions(quality_setting=QualitySetting.QUALITY)
        assert opts.effective_min_level == 0

    def test_balanced_effective_min_level(self):
        """Test balanced setting gives min_level=4."""
        opts = OFOptions(quality_setting=QualitySetting.BALANCED)
        assert opts.effective_min_level == 4

    def test_fast_effective_min_level(self):
        """Test fast setting gives min_level=6."""
        opts = OFOptions(quality_setting=QualitySetting.FAST)
        assert opts.effective_min_level == 6

    def test_custom_min_level_override(self):
        """Test explicit min_level overrides quality setting."""
        opts = OFOptions(quality_setting=QualitySetting.QUALITY, min_level=8)
        assert opts.effective_min_level == 8


class TestGetWeightAt:
    """Test get_weight_at method."""

    def test_get_weight_at_1d_weights(self):
        """Test get_weight_at with 1D channel weights."""
        opts = OFOptions(weight=[0.3, 0.7])

        w0 = opts.get_weight_at(0, n_channels=2)
        w1 = opts.get_weight_at(1, n_channels=2)

        assert np.isclose(w0, 0.3)
        assert np.isclose(w1, 0.7)

    def test_get_weight_at_2d_spatial_weights(self):
        """Test get_weight_at with 2D spatial weights."""
        H, W = 32, 32
        weight_2d = np.ones((H, W), dtype=np.float64) * 0.5

        opts = OFOptions(weight=weight_2d)

        w0 = opts.get_weight_at(0, n_channels=1)

        # Should return 2D array
        assert isinstance(w0, np.ndarray)
        assert w0.shape == (H, W)
        assert np.allclose(w0, 0.5)

    def test_get_weight_at_3d_spatial_weights(self):
        """Test get_weight_at with 3D spatial weights."""
        H, W, C = 32, 32, 2
        weight_3d = np.ones((H, W, C), dtype=np.float64)
        weight_3d[:, :, 0] = 0.3
        weight_3d[:, :, 1] = 0.7

        opts = OFOptions(weight=weight_3d)

        w0 = opts.get_weight_at(0, n_channels=2)
        w1 = opts.get_weight_at(1, n_channels=2)

        # Should return 2D arrays
        assert isinstance(w0, np.ndarray)
        assert isinstance(w1, np.ndarray)
        assert w0.shape == (H, W)
        assert w1.shape == (H, W)
        assert np.allclose(w0, 0.3)
        assert np.allclose(w1, 0.7)


class TestModelCopy:
    """Test model_copy and copy methods."""

    def test_model_copy_with_spatial_weights(self):
        """Test model_copy preserves spatial weights."""
        H, W, C = 32, 32, 2
        weight_2d = np.ones((H, W, C), dtype=np.float64)

        opts = OFOptions(weight=weight_2d)
        opts_copy = opts.model_copy()

        assert isinstance(opts_copy.weight, np.ndarray)
        assert opts_copy.weight.shape == (H, W, C)

    def test_copy_method(self):
        """Test copy() method."""
        opts = OFOptions(alpha=2.5, weight=[0.3, 0.7])
        opts_copy = opts.copy()

        assert opts_copy.alpha == (2.5, 2.5)
        assert opts_copy.weight == opts.weight
        assert opts_copy is not opts


class TestToDict:
    """Test to_dict method for backend parameters."""

    def test_to_dict_basic(self):
        """Test to_dict returns correct parameters."""
        opts = OFOptions(
            alpha=2.5,
            weight=[0.4, 0.6],
            levels=100,
            min_level=4,
            iterations=50,
        )

        params = opts.to_dict()

        assert params["alpha"] == (2.5, 2.5)
        assert params["levels"] == 100
        assert params["min_level"] == 4
        assert params["iterations"] == 50

    def test_to_dict_with_spatial_weights(self):
        """Test to_dict with spatial weights."""
        H, W, C = 16, 16, 1
        weight_2d = np.ones((H, W, C))

        opts = OFOptions(weight=weight_2d)
        params = opts.to_dict()

        # Weight should be preserved as numpy array
        assert isinstance(params["weight"], np.ndarray)
        assert params["weight"].shape == (H, W, C)


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
