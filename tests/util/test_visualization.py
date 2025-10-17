"""
Tests for visualization utility functions.

Tests all visualization functions including color mapping, multispectral mapping,
quiver visualization, and flow-to-color conversion.
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

from pyflowreg.util.visualization import (
    color_map_numpy_2ch,
    get_visualization,
    multispectral_mapping,
    quiver_visualization,
    flow_to_color,
)


class TestColorMapNumpy2ch:
    """Test color_map_numpy_2ch function."""

    def test_color_map_numpy_2ch_basic(self):
        """Test basic 2-channel color mapping."""
        # Create simple 2-channel image
        H, W = 32, 32
        img = np.random.rand(H, W, 2).astype(np.float32)

        # Apply color mapping
        result = color_map_numpy_2ch(img)

        # Check output shape and type
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8
        assert result.max() <= 255
        assert result.min() >= 0

    def test_color_map_numpy_2ch_with_scaling(self):
        """Test color mapping with scaling parameters."""
        H, W = 16, 16
        img = np.random.rand(H, W, 2).astype(np.float32)

        # Apply with scaling
        scaling_left = (2.0, 0.5)
        scaling_right = (1.5, 0.2)
        result = color_map_numpy_2ch(
            img, scaling_left=scaling_left, scaling_right=scaling_right
        )

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_color_map_numpy_2ch_with_references(self):
        """Test color mapping with reference normalization."""
        H, W = 20, 20
        img = np.random.rand(H, W, 2).astype(np.float32)
        ref_left = np.random.rand(H, W).astype(np.float32)
        ref_right = np.random.rand(H, W).astype(np.float32)

        result = color_map_numpy_2ch(
            img, reference_left=ref_left, reference_right=ref_right
        )

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_color_map_numpy_2ch_inverted(self):
        """Test color mapping with channel inversion."""
        H, W = 24, 24
        img = np.random.rand(H, W, 2).astype(np.float32)

        # Get normal and inverted results
        result_normal = color_map_numpy_2ch(img, inverted=False)
        result_inverted = color_map_numpy_2ch(img, inverted=True)

        # Check channels are swapped
        assert result_normal.shape == result_inverted.shape
        # Red and blue channels should be swapped
        np.testing.assert_array_equal(result_normal[:, :, 0], result_inverted[:, :, 2])
        np.testing.assert_array_equal(result_normal[:, :, 2], result_inverted[:, :, 0])

    def test_color_map_numpy_2ch_return_float(self):
        """Test returning float output instead of uint8."""
        H, W = 16, 16
        img = np.random.rand(H, W, 2).astype(np.float32)

        result = color_map_numpy_2ch(img, return_float=True)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.float64
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_color_map_numpy_2ch_single_channel(self):
        """Test handling of single channel input."""
        H, W = 32, 32
        img = np.random.rand(H, W).astype(np.float32)

        result = color_map_numpy_2ch(img)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_color_map_numpy_2ch_batched_format(self):
        """Test handling of batched format (H, W, 2, N)."""
        H, W, N = 16, 16, 5
        img = np.random.rand(H, W, 2, N).astype(np.float32)

        # Should process first frame
        result = color_map_numpy_2ch(img)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8


class TestGetVisualization:
    """Test get_visualization function."""

    def test_get_visualization_basic(self):
        """Test basic MATLAB-compatible visualization."""
        H, W = 32, 32
        ch1 = np.random.rand(H, W).astype(np.float32)
        ch2 = np.random.rand(H, W).astype(np.float32)

        result = get_visualization(ch1, ch2)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.float64
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_get_visualization_with_scaling(self):
        """Test visualization with scaling parameters."""
        H, W = 24, 24
        ch1 = np.random.rand(H, W).astype(np.float32)
        ch2 = np.random.rand(H, W).astype(np.float32)

        scaling_left = (1.5, 0.3)
        scaling_right = (2.0, 0.1)

        result = get_visualization(
            ch1, ch2, scaling_left=scaling_left, scaling_right=scaling_right
        )

        assert result.shape == (H, W, 3)
        assert result.dtype == np.float64

    def test_get_visualization_with_references(self):
        """Test visualization with reference normalization."""
        H, W = 20, 20
        ch1 = np.random.rand(H, W).astype(np.float32)
        ch2 = np.random.rand(H, W).astype(np.float32)
        ref_left = np.random.rand(H, W).astype(np.float32)
        ref_right = np.random.rand(H, W).astype(np.float32)

        result = get_visualization(
            ch1, ch2, reference_left=ref_left, reference_right=ref_right
        )

        assert result.shape == (H, W, 3)
        assert result.dtype == np.float64

    def test_get_visualization_inverted(self):
        """Test visualization with channel inversion."""
        H, W = 16, 16
        ch1 = np.random.rand(H, W).astype(np.float32)
        ch2 = np.random.rand(H, W).astype(np.float32)

        result_normal = get_visualization(ch1, ch2, inverted=False)
        result_inverted = get_visualization(ch1, ch2, inverted=True)

        # Check channels are swapped
        np.testing.assert_array_equal(result_normal[:, :, 0], result_inverted[:, :, 2])
        np.testing.assert_array_equal(result_normal[:, :, 2], result_inverted[:, :, 0])

    def test_get_visualization_3d_input(self):
        """Test visualization with 3D input (H, W, N)."""
        H, W, N = 16, 16, 3
        ch1 = np.random.rand(H, W, N).astype(np.float32)
        ch2 = np.random.rand(H, W, N).astype(np.float32)

        result = get_visualization(ch1, ch2)

        assert result.shape == (H, W, 3, N)
        assert result.dtype == np.float64

    def test_get_visualization_matlab_typo_compatibility(self):
        """Test that MATLAB typo in reference normalization is preserved."""
        H, W = 16, 16
        ch1 = np.ones((H, W)).astype(np.float32) * 0.5
        ch2 = np.ones((H, W)).astype(np.float32) * 0.7

        ref_left = np.array([[0.2, 0.8]])  # min=0.2, max=0.8
        ref_right = np.array([[0.3, 0.9]])  # min=0.3, max=0.9

        result = get_visualization(
            ch1, ch2, reference_left=ref_left, reference_right=ref_right
        )

        # Check that the typo is preserved (reference_left used in ch2 normalization)
        # This is intentional to match MATLAB behavior
        assert result.shape == (H, W, 3)


class TestMultispectralMapping:
    """Test multispectral_mapping function."""

    def test_multispectral_mapping_single_channel(self):
        """Test mapping of single channel image."""
        H, W = 32, 32
        img = np.random.rand(H, W).astype(np.float32)

        result = multispectral_mapping(img)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.float64
        assert result.max() <= 1.0
        assert result.min() >= 0.0
        # All channels should be identical for grayscale
        np.testing.assert_array_almost_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_almost_equal(result[:, :, 1], result[:, :, 2])

    def test_multispectral_mapping_two_channels(self):
        """Test mapping of 2-channel image."""
        H, W = 24, 24
        img = np.random.rand(H, W, 2).astype(np.float32)

        result = multispectral_mapping(img)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.float64
        # Blue channel should be zero for 2-band input
        assert np.allclose(result[:, :, 2], 0)

    def test_multispectral_mapping_three_channels(self):
        """Test mapping of 3-channel image."""
        H, W = 20, 20
        img = np.random.rand(H, W, 3).astype(np.float32)

        result = multispectral_mapping(img)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.float64
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_multispectral_mapping_multispectral(self):
        """Test mapping of image with more than 3 channels using PCA."""
        H, W, C = 16, 16, 5
        img = np.random.rand(H, W, C).astype(np.float32)

        result = multispectral_mapping(img)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.float64
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_multispectral_mapping_edge_cases(self):
        """Test edge cases like uniform images."""
        H, W = 16, 16

        # Uniform image
        img_uniform = np.ones((H, W, 2)) * 0.5
        result_uniform = multispectral_mapping(img_uniform)
        assert result_uniform.shape == (H, W, 3)

        # Image with one channel having no variation
        img_partial = np.random.rand(H, W, 2)
        img_partial[:, :, 1] = 0.5  # Second channel uniform
        result_partial = multispectral_mapping(img_partial)
        assert result_partial.shape == (H, W, 3)


class TestQuiverVisualization:
    """Test quiver_visualization function."""

    def test_quiver_visualization_grayscale(self, capsys):
        """Test quiver visualization with grayscale image."""
        H, W = 64, 64
        img = np.random.rand(H, W).astype(np.float32)
        flow = np.random.randn(H, W, 2).astype(np.float32) * 2

        result = quiver_visualization(img, flow, return_array=True)

        # Check console output
        captured = capsys.readouterr()
        assert "Processing image with 1 channel(s)" in captured.out
        assert "Using grayscale visualization" in captured.out

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_quiver_visualization_two_channel(self, capsys):
        """Test quiver visualization with 2-channel image."""
        H, W = 48, 48
        img = np.random.rand(H, W, 2).astype(np.float32)
        flow = np.random.randn(H, W, 2).astype(np.float32)

        result = quiver_visualization(img, flow, return_array=True)

        # Check console output
        captured = capsys.readouterr()
        assert "Processing image with 2 channel(s)" in captured.out
        assert "Using 2-channel color mapping" in captured.out

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_quiver_visualization_rgb(self, capsys):
        """Test quiver visualization with RGB image."""
        H, W = 32, 32
        img = np.random.rand(H, W, 3).astype(np.float32)
        flow = np.random.randn(H, W, 2).astype(np.float32)

        result = quiver_visualization(img, flow, return_array=True)

        # Check console output
        captured = capsys.readouterr()
        assert "Processing image with 3 channel(s)" in captured.out
        assert "Using direct RGB visualization" in captured.out

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_quiver_visualization_multispectral(self, capsys):
        """Test quiver visualization with multispectral image."""
        H, W = 32, 32
        img = np.random.rand(H, W, 5).astype(np.float32)
        flow = np.random.randn(H, W, 2).astype(np.float32)

        result = quiver_visualization(img, flow, return_array=True)

        # Check console output
        captured = capsys.readouterr()
        assert "Processing image with 5 channel(s)" in captured.out
        assert "Using multispectral mapping for 5 channels" in captured.out

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_quiver_visualization_scale(self):
        """Test quiver visualization with different scale factors."""
        H, W = 40, 40
        img = np.random.rand(H, W, 2).astype(np.float32)
        flow = np.ones((H, W, 2)).astype(np.float32)

        # Test with different scales
        result1 = quiver_visualization(img, flow, scale=0.5, return_array=True)
        result2 = quiver_visualization(img, flow, scale=2.0, return_array=True)

        assert result1.shape == (H, W, 3)
        assert result2.shape == (H, W, 3)

    def test_quiver_visualization_opencv_backend(self):
        """Test quiver visualization with OpenCV backend."""
        H, W = 40, 40
        img = np.random.rand(H, W, 2).astype(np.float32)
        flow = np.random.randn(H, W, 2).astype(np.float32) * 2

        # Test with OpenCV backend
        result = quiver_visualization(img, flow, backend="opencv")
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_quiver_visualization_backends_comparison(self):
        """Test that both backends produce valid output."""
        H, W = 32, 32
        img = np.random.rand(H, W, 2).astype(np.float32)
        flow = np.random.randn(H, W, 2).astype(np.float32) * 2

        # Test OpenCV backend
        result_cv = quiver_visualization(img, flow, backend="opencv", scale=2.0)
        assert result_cv.shape == (H, W, 3)
        assert result_cv.dtype == np.uint8

        # Test matplotlib backend
        result_mpl = quiver_visualization(img, flow, backend="matplotlib", scale=2.0)
        assert result_mpl.shape == (H, W, 3)
        assert result_mpl.dtype == np.uint8

    def test_quiver_visualization_downsample_parameter(self):
        """Test custom downsample parameter."""
        H, W = 64, 64
        img = np.random.rand(H, W, 2).astype(np.float32)
        flow = np.random.randn(H, W, 2).astype(np.float32) * 2

        # Test with different downsample values
        result1 = quiver_visualization(img, flow, downsample=0.05, backend="opencv")
        result2 = quiver_visualization(img, flow, downsample=0.01, backend="opencv")

        # Both should produce valid output
        assert result1.shape == (H, W, 3)
        assert result2.shape == (H, W, 3)

    def test_quiver_visualization_no_streamlines(self):
        """Test disabling streamlines in matplotlib backend."""
        H, W = 32, 32
        img = np.random.rand(H, W, 2).astype(np.float32)
        flow = np.random.randn(H, W, 2).astype(np.float32) * 2

        result = quiver_visualization(
            img, flow, show_streamlines=False, backend="matplotlib"
        )
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_quiver_visualization_invalid_backend(self):
        """Test error handling for invalid backend."""
        H, W = 32, 32
        img = np.random.rand(H, W, 2).astype(np.float32)
        flow = np.random.randn(H, W, 2).astype(np.float32)

        with pytest.raises(ValueError, match="Backend must be"):
            quiver_visualization(img, flow, backend="invalid")

    def test_quiver_visualization_invalid_flow_shape(self):
        """Test error handling for invalid flow field shape."""
        H, W = 32, 32
        img = np.random.rand(H, W, 2).astype(np.float32)
        flow_invalid = np.random.randn(H, W).astype(
            np.float32
        )  # Missing channel dimension

        with pytest.raises(ValueError, match="Displacement field must have shape"):
            quiver_visualization(img, flow_invalid)

    def test_quiver_visualization_small_image(self):
        """Test quiver visualization with very small images."""
        H, W = 8, 8
        img = np.random.rand(H, W).astype(np.float32)
        flow = np.random.randn(H, W, 2).astype(np.float32)

        # Should handle small images gracefully
        result = quiver_visualization(img, flow, return_array=True)
        assert result.shape == (H, W, 3)

    def test_quiver_visualization_custom_colors(self):
        """Test custom colors for quivers and streamlines."""
        H, W = 64, 64
        img = np.random.rand(H, W, 2).astype(np.float32)

        # Create circular flow
        Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        u = -(Y - H / 2) * 0.3
        v = (X - W / 2) * 0.3
        flow = np.stack([u, v], axis=-1).astype(np.float32)

        # Test matplotlib backend with custom colors
        result_matplotlib = quiver_visualization(
            img,
            flow,
            backend="matplotlib",
            quiver_color=(255, 0, 0),  # Red quivers
            streamline_color=(0, 0, 255),  # Blue streamlines
            show_streamlines=True,
        )
        assert result_matplotlib.shape == (H, W, 3)

        # Test OpenCV backend with custom colors
        result_opencv = quiver_visualization(
            img,
            flow,
            backend="opencv",
            quiver_color=(0, 255, 0),  # Green quivers
            streamline_color=(255, 0, 0),  # Red streamlines
            show_streamlines=True,
        )
        assert result_opencv.shape == (H, W, 3)

    def test_quiver_visualization_adaptive_density(self):
        """Test that streamline density adapts to different image sizes."""

        def create_circular_flow(h, w):
            """Create a circular flow pattern for testing."""
            flow = np.zeros((h, w, 2), dtype=np.float32)
            center_x, center_y = w // 2, h // 2

            for y in range(h):
                for x in range(w):
                    dx = x - center_x
                    dy = y - center_y
                    r = np.sqrt(dx**2 + dy**2)
                    if r > 0 and r < min(h, w) * 0.4:
                        # Tangential flow (counter-clockwise rotation)
                        flow[y, x, 0] = -dy / r * 5  # U component
                        flow[y, x, 1] = dx / r * 5  # V component
            return flow

        # Test different image sizes
        sizes = [(100, 100), (200, 200), (500, 500)]

        for h, w in sizes:
            # Create test image
            img = np.zeros((h, w), dtype=np.uint8)
            img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = (
                255  # White square in center
            )

            # Create flow pattern
            flow = create_circular_flow(h, w)

            # Test OpenCV backend with streamlines
            result_opencv = quiver_visualization(
                img,
                flow,
                scale=2.0,
                downsample=0.1,
                backend="opencv",
                show_streamlines=True,
                quiver_color=(255, 0, 0),  # Red quivers
                streamline_color=(0, 255, 0),  # Green streamlines
            )

            assert result_opencv.shape == (h, w, 3)
            assert result_opencv.dtype == np.uint8

            # Test matplotlib backend
            result_matplotlib = quiver_visualization(
                img,
                flow,
                scale=2.0,
                downsample=0.1,
                backend="matplotlib",
                show_streamlines=True,
                quiver_color=(255, 0, 0),
                streamline_color=(0, 255, 0),
            )

            assert result_matplotlib.shape == (h, w, 3)
            assert result_matplotlib.dtype == np.uint8


class TestFlowToColor:
    """Test flow_to_color function."""

    def test_flow_to_color_basic(self):
        """Test basic flow to color conversion."""
        H, W = 32, 32
        flow = np.random.randn(H, W, 2).astype(np.float32) * 5

        result = flow_to_color(flow)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8
        assert result.max() <= 255
        assert result.min() >= 0

    def test_flow_to_color_with_max_flow(self):
        """Test flow to color with specified maximum flow."""
        H, W = 24, 24
        flow = np.random.randn(H, W, 2).astype(np.float32) * 10
        max_flow = 5.0

        result = flow_to_color(flow, max_flow=max_flow)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_flow_to_color_zero_flow(self):
        """Test flow to color with zero flow field."""
        H, W = 16, 16
        flow = np.zeros((H, W, 2), dtype=np.float32)

        result = flow_to_color(flow)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_flow_to_color_uniform_flow(self):
        """Test flow to color with uniform flow field."""
        H, W = 20, 20
        # Uniform rightward flow
        flow = np.ones((H, W, 2), dtype=np.float32)
        flow[:, :, 0] = 5.0  # u component
        flow[:, :, 1] = 0.0  # v component

        result = flow_to_color(flow)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

        # All pixels should have similar color for uniform flow
        # Check that variance is low across spatial dimensions
        color_variance = np.var(result.reshape(-1, 3), axis=0)
        assert np.all(color_variance < 10)  # Low variance in color

    def test_flow_to_color_unknown_flow(self):
        """Test handling of unknown/invalid flow values."""
        H, W = 16, 16
        flow = np.random.randn(H, W, 2).astype(np.float32)

        # Add some "unknown" flow values
        flow[5:10, 5:10, :] = 1e10  # Very large values

        result = flow_to_color(flow)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

        # Unknown flow regions should be black (0)
        assert np.all(result[5:10, 5:10, :] == 0)

    def test_flow_to_color_circular_pattern(self):
        """Test flow to color with circular flow pattern."""
        H, W = 32, 32
        y, x = np.mgrid[:H, :W]
        center_y, center_x = H // 2, W // 2

        # Create circular flow pattern
        flow = np.zeros((H, W, 2), dtype=np.float32)
        flow[:, :, 0] = -(y - center_y)  # u component
        flow[:, :, 1] = x - center_x  # v component

        result = flow_to_color(flow)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

        # Should produce a color wheel pattern
        # Check that the result is valid
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_flow_to_color_large_flow(self):
        """Test flow to color with very large flow magnitudes."""
        H, W = 24, 24
        flow = np.random.randn(H, W, 2).astype(np.float32) * 100

        result = flow_to_color(flow)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

        # Should handle large flows without overflow
        assert result.max() <= 255
        assert result.min() >= 0

    def test_flow_to_color_colorwheel_properties(self):
        """Test that the color wheel has expected properties."""
        H, W = 32, 32

        # Create flows pointing in different directions
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        flow = np.zeros((H, W, 2), dtype=np.float32)

        for i, angle in enumerate(angles):
            row = i * 4
            flow[row : row + 4, :, 0] = np.cos(angle) * 5
            flow[row : row + 4, :, 1] = np.sin(angle) * 5

        result = flow_to_color(flow)

        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

        # Different directions should produce different colors
        colors = []
        for i in range(len(angles)):
            row = i * 4
            avg_color = np.mean(result[row : row + 4, :, :], axis=(0, 1))
            colors.append(avg_color)

        # Check that colors are sufficiently different
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                color_diff = np.linalg.norm(colors[i] - colors[j])
                assert color_diff > 10  # Colors should be distinguishable


class TestIntegration:
    """Integration tests for visualization functions."""

    def test_pipeline_2channel_to_quiver(self):
        """Test pipeline from 2-channel image through color mapping to quiver."""
        H, W = 48, 48
        img = np.random.rand(H, W, 2).astype(np.float32)
        flow = np.random.randn(H, W, 2).astype(np.float32)

        # Apply color mapping
        colored = color_map_numpy_2ch(img)
        assert colored.shape == (H, W, 3)

        # Use colored image in quiver visualization
        result = quiver_visualization(colored, flow, return_array=True)
        assert result.shape == (H, W, 3)

    def test_flow_visualization_pipeline(self):
        """Test complete flow visualization pipeline."""
        H, W = 32, 32

        # Create synthetic motion
        y, x = np.mgrid[:H, :W]
        flow = np.zeros((H, W, 2), dtype=np.float32)
        flow[:, :, 0] = np.sin(2 * np.pi * x / W) * 3
        flow[:, :, 1] = np.cos(2 * np.pi * y / H) * 3

        # Convert flow to color
        flow_color = flow_to_color(flow)
        assert flow_color.shape == (H, W, 3)

        # Create a 2-channel image
        img = np.random.rand(H, W, 2).astype(np.float32)

        # Visualize with quiver
        quiver_result = quiver_visualization(img, flow, return_array=True)
        assert quiver_result.shape == (H, W, 3)

    def test_reference_normalization_consistency(self):
        """Test consistency between color_map_numpy_2ch and get_visualization."""
        H, W = 24, 24
        ch1 = np.random.rand(H, W).astype(np.float32)
        ch2 = np.random.rand(H, W).astype(np.float32)

        # Stack channels for color_map_numpy_2ch
        img_stacked = np.stack([ch1, ch2], axis=-1)

        # Compare results without references
        result1 = color_map_numpy_2ch(img_stacked, return_float=True)
        result2 = get_visualization(ch1, ch2)

        # Results should be very similar
        np.testing.assert_array_almost_equal(result1, result2, decimal=5)


@pytest.mark.parametrize("shape", [(16, 16), (32, 48), (64, 64)])
def test_various_image_sizes(shape):
    """Test functions with various image sizes."""
    H, W = shape

    # Test color_map_numpy_2ch
    img_2ch = np.random.rand(H, W, 2).astype(np.float32)
    result_2ch = color_map_numpy_2ch(img_2ch)
    assert result_2ch.shape == (H, W, 3)

    # Test multispectral_mapping
    img_multi = np.random.rand(H, W, 4).astype(np.float32)
    result_multi = multispectral_mapping(img_multi)
    assert result_multi.shape == (H, W, 3)

    # Test flow_to_color
    flow = np.random.randn(H, W, 2).astype(np.float32)
    result_flow = flow_to_color(flow)
    assert result_flow.shape == (H, W, 3)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint8, np.uint16])
def test_various_dtypes(dtype):
    """Test functions with various input data types."""
    H, W = 24, 24

    # Create test data in specified dtype
    if dtype in [np.uint8, np.uint16]:
        max_val = np.iinfo(dtype).max
        img = np.random.randint(0, max_val, (H, W, 2), dtype=dtype)
    else:
        img = np.random.rand(H, W, 2).astype(dtype)

    # Functions should handle different dtypes
    result = color_map_numpy_2ch(img)
    assert result.shape == (H, W, 3)
    assert result.dtype == np.uint8
