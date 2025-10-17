"""
Tests for Dense Inverse Search Optical Flow (DIS) backend.
"""

import pytest
import numpy as np
import cv2
import pickle
from pyflowreg.core.diso_optical_flow import DisoOF, _diso_factory


class TestDisoOFBasics:
    """Test basic functionality of DisoOF class."""

    def test_diso_initialization(self):
        """Test that DisoOF initializes correctly with default parameters."""
        diso = DisoOF()
        params = diso.get_params()

        assert params["preset"] == cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
        assert params["finest_scale"] == 2
        assert params["gradient_descent_iterations"] == 12
        assert params["patch_size"] == 8
        assert params["patch_stride"] == 4
        assert params["use_mean_normalization"] is True
        assert params["use_spatial_propagation"] is True

    def test_diso_custom_initialization(self):
        """Test DisoOF initialization with custom parameters."""
        diso = DisoOF(
            preset=cv2.DISOPTICAL_FLOW_PRESET_FAST,
            finest_scale=1,
            gradient_descent_iterations=24,
            patch_size=16,
            patch_stride=8,
            use_mean_normalization=False,
            use_spatial_propagation=False,
        )
        params = diso.get_params()

        assert params["preset"] == cv2.DISOPTICAL_FLOW_PRESET_FAST
        assert params["finest_scale"] == 1
        assert params["gradient_descent_iterations"] == 24
        assert params["patch_size"] == 16
        assert params["patch_stride"] == 8
        assert params["use_mean_normalization"] is False
        assert params["use_spatial_propagation"] is False

    def test_diso_set_params(self):
        """Test parameter updating."""
        diso = DisoOF()
        diso.set_params(finest_scale=3, patch_size=12)
        params = diso.get_params()

        assert params["finest_scale"] == 3
        assert params["patch_size"] == 12

    def test_diso_set_preset(self):
        """Test preset updating."""
        diso = DisoOF()
        diso.set_preset(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        params = diso.get_params()

        assert params["preset"] == cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST


class TestDisoOFFlow:
    """Test optical flow computation."""

    @pytest.fixture
    def simple_translation_images(self):
        """Create simple translated images for testing."""
        H, W = 128, 128
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H // 2, W // 2

        # Create circular pattern
        fixed = ((x - center_x) ** 2 + (y - center_y) ** 2) < 30**2
        fixed = fixed.astype(np.float32)

        # Create translated version
        shift_x, shift_y = 5, 3
        moving = np.roll(fixed, shift=(shift_y, shift_x), axis=(0, 1))

        return fixed, moving, shift_x, shift_y

    def test_diso_simple_translation(self, simple_translation_images):
        """Test flow computation on simple translation."""
        fixed, moving, shift_x, shift_y = simple_translation_images

        diso = DisoOF(preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = diso(fixed, moving)

        assert flow.shape == (fixed.shape[0], fixed.shape[1], 2)
        assert flow.dtype == np.float32

        # Check flow in the center region where pattern exists
        center_mask = fixed > 0.5
        if center_mask.any():
            mean_u = flow[center_mask, 0].mean()
            mean_v = flow[center_mask, 1].mean()

            # Allow some tolerance for discrete pixel shifts
            assert abs(mean_u - shift_x) < 2.0
            assert abs(mean_v - shift_y) < 2.0

    def test_diso_with_initial_flow(self, simple_translation_images):
        """Test flow computation with warm start."""
        fixed, moving, shift_x, shift_y = simple_translation_images

        # Create initial flow guess (80% of true displacement)
        H, W = fixed.shape
        w_init = np.zeros((H, W, 2), dtype=np.float32)
        w_init[:, :, 0] = shift_x * 0.8
        w_init[:, :, 1] = shift_y * 0.8

        diso = DisoOF()
        flow = diso(fixed, moving, w=w_init)

        assert flow.shape == (H, W, 2)
        assert flow.dtype == np.float32

        # With warm start, should converge closer to true displacement
        center_mask = fixed > 0.5
        if center_mask.any():
            mean_u = flow[center_mask, 0].mean()
            mean_v = flow[center_mask, 1].mean()

            assert abs(mean_u - shift_x) < 2.0
            assert abs(mean_v - shift_y) < 2.0

    def test_diso_multichannel_input(self):
        """Test flow computation with multi-channel images."""
        H, W = 64, 64

        # Create 3-channel test images
        fixed = np.random.rand(H, W, 3).astype(np.float32) * 0.8 + 0.1
        moving = np.roll(fixed, shift=(2, 3), axis=(0, 1))

        diso = DisoOF()
        flow = diso(fixed, moving)

        assert flow.shape == (H, W, 2)
        assert flow.dtype == np.float32

    def test_diso_with_weights(self):
        """Test flow computation with channel weights."""
        H, W = 64, 64

        # Create 3-channel test images
        fixed = np.random.rand(H, W, 3).astype(np.float32) * 0.8 + 0.1
        moving = np.roll(fixed, shift=(2, 3), axis=(0, 1))

        # Use custom weights
        weights = np.array([0.5, 0.3, 0.2])

        diso = DisoOF()
        flow = diso(fixed, moving, weight=weights)

        assert flow.shape == (H, W, 2)
        assert flow.dtype == np.float32

    def test_diso_uint8_input(self):
        """Test that DisoOF handles uint8 input correctly."""
        H, W = 64, 64

        # Create uint8 images
        fixed = (np.random.rand(H, W) * 255).astype(np.uint8)
        moving = np.roll(fixed, shift=(2, 3), axis=(0, 1))

        diso = DisoOF()
        flow = diso(fixed, moving)

        assert flow.shape == (H, W, 2)
        assert flow.dtype == np.float32

    def test_diso_zero_flow(self):
        """Test flow computation when images are identical."""
        H, W = 64, 64
        image = np.random.rand(H, W).astype(np.float32)

        diso = DisoOF()
        flow = diso(image, image)

        assert flow.shape == (H, W, 2)
        assert flow.dtype == np.float32

        # Flow should be near zero
        assert np.abs(flow).mean() < 0.5


class TestDisoOFPresets:
    """Test different DIS presets."""

    @pytest.fixture
    def test_images(self):
        """Create test images."""
        H, W = 128, 128
        fixed = np.random.rand(H, W).astype(np.float32) * 0.8 + 0.1
        moving = np.roll(fixed, shift=(3, 5), axis=(0, 1))
        return fixed, moving

    def test_diso_ultrafast_preset(self, test_images):
        """Test ultrafast preset."""
        fixed, moving = test_images

        diso = DisoOF(preset=cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        flow = diso(fixed, moving)

        assert flow.shape == (fixed.shape[0], fixed.shape[1], 2)
        assert flow.dtype == np.float32

    def test_diso_fast_preset(self, test_images):
        """Test fast preset."""
        fixed, moving = test_images

        diso = DisoOF(preset=cv2.DISOPTICAL_FLOW_PRESET_FAST)
        flow = diso(fixed, moving)

        assert flow.shape == (fixed.shape[0], fixed.shape[1], 2)
        assert flow.dtype == np.float32

    def test_diso_medium_preset(self, test_images):
        """Test medium preset."""
        fixed, moving = test_images

        diso = DisoOF(preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = diso(fixed, moving)

        assert flow.shape == (fixed.shape[0], fixed.shape[1], 2)
        assert flow.dtype == np.float32


class TestDisoOFPickle:
    """Test pickling support for multiprocessing."""

    def test_diso_pickle_roundtrip(self):
        """Test that DisoOF can be pickled and unpickled."""
        diso = DisoOF(
            preset=cv2.DISOPTICAL_FLOW_PRESET_FAST, finest_scale=3, patch_size=16
        )

        # Pickle and unpickle
        pickled = pickle.dumps(diso)
        diso_restored = pickle.loads(pickled)

        # Check that parameters are preserved
        params = diso_restored.get_params()
        assert params["preset"] == cv2.DISOPTICAL_FLOW_PRESET_FAST
        assert params["finest_scale"] == 3
        assert params["patch_size"] == 16

    def test_diso_pickle_with_computation(self):
        """Test that pickled DisoOF still works after unpickling."""
        diso = DisoOF()

        # Create test images
        H, W = 64, 64
        fixed = np.random.rand(H, W).astype(np.float32)
        moving = np.roll(fixed, shift=(2, 3), axis=(0, 1))

        # Compute flow before pickling
        flow_before = diso(fixed, moving)

        # Pickle and unpickle
        pickled = pickle.dumps(diso)
        diso_restored = pickle.loads(pickled)

        # Compute flow after unpickling
        flow_after = diso_restored(fixed, moving)

        assert flow_after.shape == flow_before.shape
        assert flow_after.dtype == flow_before.dtype


class TestDisoFactory:
    """Test the diso_factory function."""

    def test_diso_factory_basic(self):
        """Test basic factory functionality."""
        # Create factory function
        diso_fn = _diso_factory(preset=cv2.DISOPTICAL_FLOW_PRESET_FAST)

        # Create test images
        H, W = 64, 64
        fixed = np.random.rand(H, W).astype(np.float32)
        moving = np.roll(fixed, shift=(2, 3), axis=(0, 1))

        # Call factory function
        flow = diso_fn(fixed, moving)

        assert flow.shape == (H, W, 2)
        assert flow.dtype == np.float32

    def test_diso_factory_with_params(self):
        """Test factory with custom parameters."""
        diso_fn = _diso_factory(
            preset=cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, finest_scale=4, patch_size=16
        )

        H, W = 64, 64
        fixed = np.random.rand(H, W).astype(np.float32)
        moving = np.roll(fixed, shift=(1, 2), axis=(0, 1))

        flow = diso_fn(fixed, moving)

        assert flow.shape == (H, W, 2)
        assert flow.dtype == np.float32

    def test_diso_factory_with_initial_flow(self):
        """Test factory with initial flow."""
        diso_fn = _diso_factory()

        H, W = 64, 64
        fixed = np.random.rand(H, W).astype(np.float32)
        moving = np.roll(fixed, shift=(2, 3), axis=(0, 1))

        # Create initial flow
        w_init = np.ones((H, W, 2), dtype=np.float32)

        flow = diso_fn(fixed, moving, uv=w_init)

        assert flow.shape == (H, W, 2)
        assert flow.dtype == np.float32

    def test_diso_factory_with_weights(self):
        """Test factory with channel weights."""
        diso_fn = _diso_factory()

        H, W = 64, 64
        fixed = np.random.rand(H, W, 3).astype(np.float32)
        moving = np.roll(fixed, shift=(1, 2), axis=(0, 1))

        weights = np.array([0.5, 0.3, 0.2])

        flow = diso_fn(fixed, moving, weight=weights)

        assert flow.shape == (H, W, 2)
        assert flow.dtype == np.float32
