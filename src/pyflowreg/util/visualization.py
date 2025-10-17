from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_SUPPORTED = True
except ImportError:
    MATPLOTLIB_SUPPORTED = False

try:
    from sklearn.decomposition import PCA

    SKLEARN_SUPPORTED = True
except ImportError:
    SKLEARN_SUPPORTED = False


def color_map_numpy_2ch(
    img_in: np.ndarray,
    scaling_left: Optional[Tuple[float, float]] = None,
    scaling_right: Optional[Tuple[float, float]] = None,
    reference_left: Optional[np.ndarray] = None,
    reference_right: Optional[np.ndarray] = None,
    inverted: bool = False,
    return_float: bool = False,
) -> np.ndarray:
    """
    Convert a 2-channel image to a 3-channel color visualization.

    Args:
        img_in: Input 2-channel image (H, W, 2) or stacked 2-channel images
        scaling_left: Tuple of (scale, offset) for left channel. Default (1, 0)
        scaling_right: Tuple of (scale, offset) for right channel. Default (1, 0)
        reference_left: Reference array for left channel normalization
        reference_right: Reference array for right channel normalization
        inverted: If True, swap red and blue channels
        return_float: If True, return float array in [0,1], else uint8 in [0,255]

    Returns:
        3-channel color image
    """
    # Handle different input shapes
    if img_in.ndim == 2:
        # Single channel input - treat as grayscale
        ch1 = ch2 = img_in.astype(np.float64)
    elif img_in.ndim == 3 and img_in.shape[-1] == 2:
        # Standard 2-channel input (H, W, 2)
        ch1 = img_in[:, :, 0].astype(np.float64)
        ch2 = img_in[:, :, 1].astype(np.float64)
    elif img_in.ndim == 4 and img_in.shape[2] == 2:
        # Batched format (H, W, 2, N) - process first frame
        ch1 = img_in[:, :, 0, 0].astype(np.float64)
        ch2 = img_in[:, :, 1, 0].astype(np.float64)
    else:
        # Try to extract first two channels
        ch1 = img_in[..., 0].astype(np.float64)
        ch2 = img_in[..., 1].astype(np.float64)

    # Normalization
    if reference_left is None and reference_right is None:
        # Normalize each channel independently
        ch1_min = ch1.min()
        ch1_max = ch1.max()
        if ch1_max > ch1_min:
            ch1 = (ch1 - ch1_min) / (ch1_max - ch1_min)
        else:
            ch1 = np.zeros_like(ch1)

        ch2_min = ch2.min()
        ch2_max = ch2.max()
        if ch2_max > ch2_min:
            ch2 = (ch2 - ch2_min) / (ch2_max - ch2_min)
        else:
            ch2 = np.zeros_like(ch2)
    else:
        # Use reference arrays for normalization
        if reference_left is not None:
            ref_left_min = reference_left.min()
            ref_left_max = reference_left.max()
            if ref_left_max > ref_left_min:
                ch1 = (ch1 - ref_left_min) / (ref_left_max - ref_left_min)
            else:
                ch1 = np.zeros_like(ch1)

        if reference_right is not None:
            ref_right_min = reference_right.min()
            ref_right_max = reference_right.max()
            # Note: MATLAB code has a typo here using reference_left min for denominator
            # We'll reproduce the exact behavior
            if ref_right_max > ref_right_min:
                ch2 = (ch2 - ref_right_min) / (ref_right_max - ref_right_min)
            else:
                ch2 = np.zeros_like(ch2)

    # Apply scaling
    if scaling_left is None:
        scaling_left = (1.0, 0.0)
    if scaling_right is None:
        scaling_right = (1.0, 0.0)

    ch1 = scaling_left[0] * ch1 - scaling_left[1]
    ch2 = scaling_right[0] * ch2 - scaling_right[1]

    # Create RGB image
    img_shape = ch1.shape + (3,)
    img = np.zeros(img_shape, dtype=np.float64)

    img[..., 0] = ch1
    img[..., 1] = (ch1 + ch2) * 0.5
    img[..., 2] = ch2

    # Clip to [0, 1]
    img = np.clip(img, 0, 1)

    # Swap channels if inverted
    if inverted:
        img[..., [0, 2]] = img[..., [2, 0]]

    # Return as float or uint8
    if return_float:
        return img
    else:
        return (img * 255).astype(np.uint8)


def get_visualization(
    ch1: np.ndarray,
    ch2: np.ndarray,
    scaling_left: Optional[Tuple[float, float]] = None,
    scaling_right: Optional[Tuple[float, float]] = None,
    reference_left: Optional[np.ndarray] = None,
    reference_right: Optional[np.ndarray] = None,
    inverted: bool = False,
) -> np.ndarray:
    """
    MATLAB-compatible visualization function for 2-channel images.

    This function reproduces the exact behavior of the MATLAB get_visualization function,
    including the typo in the original where reference_left min is used in ch2 normalization.

    Args:
        ch1: First channel data
        ch2: Second channel data
        scaling_left: Tuple of (scale, offset) for left channel. Default (1, 0)
        scaling_right: Tuple of (scale, offset) for right channel. Default (1, 0)
        reference_left: Reference array for left channel normalization
        reference_right: Reference array for right channel normalization
        inverted: If True, swap red and blue channels

    Returns:
        3-channel float image in range [0, 1]
    """
    ch1 = ch1.astype(np.float64)
    ch2 = ch2.astype(np.float64)

    # Normalization
    if reference_left is None and reference_right is None:
        # Normalize each channel independently
        ch1 = ch1 - ch1.min()
        ch1_max = ch1.max()
        if ch1_max > 0:
            ch1 = ch1 / ch1_max

        ch2 = ch2 - ch2.min()
        ch2_max = ch2.max()
        if ch2_max > 0:
            ch2 = ch2 / ch2_max
    else:
        # Use reference arrays for normalization
        if reference_left is not None:
            ch1 = ch1 - reference_left.min()
            denom = reference_left.max() - reference_left.min()
            if denom > 0:
                ch1 = ch1 / denom

        if reference_right is not None:
            ch2 = ch2 - reference_right.min()
            # Note: MATLAB code has typo using reference_left here
            denom = reference_right.max() - reference_left.min()
            if denom > 0:
                ch2 = ch2 / denom

    # Apply scaling
    if scaling_left is None:
        scaling_left = (1.0, 0.0)
    if scaling_right is None:
        scaling_right = (1.0, 0.0)

    ch1 = scaling_left[0] * ch1 - scaling_left[1]
    ch2 = scaling_right[0] * ch2 - scaling_right[1]

    # Create RGB image - handle different dimensionalities
    if ch1.ndim == 2:
        # 2D case (H, W)
        img = np.zeros(ch1.shape + (3,), dtype=np.float64)
        img[:, :, 0] = ch1
        img[:, :, 1] = (ch1 + ch2) * 0.5
        img[:, :, 2] = ch2
    elif ch1.ndim == 3:
        # 3D case (H, W, N) - add channel dimension
        img = np.zeros(ch1.shape[:2] + (3,) + (ch1.shape[2],), dtype=np.float64)
        img[:, :, 0, :] = ch1
        img[:, :, 1, :] = (ch1 + ch2) * 0.5
        img[:, :, 2, :] = ch2
    else:
        # General case
        img = np.zeros(ch1.shape + (3,), dtype=np.float64)
        img[..., 0] = ch1
        img[..., 1] = (ch1 + ch2) * 0.5
        img[..., 2] = ch2

    # Clip to [0, 1]
    img = np.clip(img, 0, 1)

    # Swap channels if inverted
    if inverted:
        if ch1.ndim == 2:
            img[:, :, [0, 2]] = img[:, :, [2, 0]]
        elif ch1.ndim == 3:
            img[:, :, [0, 2], :] = img[:, :, [2, 0], :]
        else:
            img[..., [0, 2]] = img[..., [2, 0]]

    return img


def multispectral_mapping(img: np.ndarray) -> np.ndarray:
    """
    Map multispectral image to RGB visualization.

    Args:
        img: Input image with shape (H, W, C) where C is number of channels/bands

    Returns:
        RGB image with shape (H, W, 3) normalized to [0, 1]
    """
    img = np.asarray(img, dtype=np.float64)

    # Handle different input dimensions
    if img.ndim == 2:
        # Single channel - expand to 3D
        img = img[:, :, np.newaxis]

    m, n, n_bands = img.shape
    rgb = np.zeros((m, n, 3), dtype=np.float64)

    if n_bands == 1:
        # Grayscale - replicate to all channels
        normalized = (img[:, :, 0] - img[:, :, 0].min()) / (
            img[:, :, 0].max() - img[:, :, 0].min() + 1e-10
        )
        rgb[:, :, 0] = normalized
        rgb[:, :, 1] = normalized
        rgb[:, :, 2] = normalized

    elif n_bands == 2:
        # Two bands - map to red and green channels (swapped as in MATLAB)
        ch1 = img[:, :, 0]
        ch2 = img[:, :, 1]
        rgb[:, :, 0] = (ch2 - ch2.min()) / (ch2.max() - ch2.min() + 1e-10)
        rgb[:, :, 1] = (ch1 - ch1.min()) / (ch1.max() - ch1.min() + 1e-10)
        rgb[:, :, 2] = 0

    elif n_bands == 3:
        # Three bands - normalize each channel
        for i in range(3):
            ch = img[:, :, i]
            rgb[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-10)

    else:
        # More than 3 bands - use PCA to reduce to 3 components
        if not SKLEARN_SUPPORTED:
            raise ImportError(
                "Multispectral mapping with >3 channels requires 'scikit-learn' library"
            )

        # Reshape for PCA
        img_reshaped = img.reshape(m * n, n_bands)

        # Apply PCA
        pca = PCA(n_components=3)
        components = pca.fit_transform(img_reshaped)

        # Reshape back and normalize each component
        for i in range(3):
            comp = components[:, i].reshape(m, n)
            rgb[:, :, i] = (comp - comp.min()) / (comp.max() - comp.min() + 1e-10)

    return np.clip(rgb, 0, 1)


def _quiver_visualization_opencv(
    img: np.ndarray,
    flow: np.ndarray,
    scale: float = 1.0,
    downsample: float = 0.03,
    show_streamlines: bool = True,
    quiver_color: Tuple[int, int, int] = (255, 255, 255),
    streamline_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Create quiver visualization using OpenCV backend (no matplotlib required).

    Args:
        img: Input image (H, W) or (H, W, C)
        flow: Displacement field with shape (H, W, 2)
        scale: Scale factor for quiver arrows
        downsample: Downsampling factor for quiver display
        show_streamlines: Whether to show streamlines
        quiver_color: RGB color for quiver arrows (default white)
        streamline_color: RGB color for streamlines (default black)

    Returns:
        Visualization image as numpy array with shape (H, W, 3)
    """
    # Ensure correct shapes
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    h, w, n_channels = img.shape

    # Prepare background image based on number of channels
    if n_channels == 1:
        # Grayscale to RGB
        img_rgb = np.stack([img[:, :, 0]] * 3, axis=-1)
    elif n_channels == 2:
        # Use get_visualization for 2-channel
        img_rgb = get_visualization(img[:, :, 0], img[:, :, 1])
    elif n_channels == 3:
        img_rgb = img.copy()
    else:
        # Use first 3 channels or multispectral mapping
        if n_channels > 3:
            # For >3 channels, could use multispectral_mapping but that requires sklearn
            # So just use first 3 channels as fallback
            img_rgb = img[:, :, :3].copy()
        else:
            img_rgb = multispectral_mapping(img)

    # Normalize to 0-255 range
    if img_rgb.max() <= 1.0:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    else:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    # Create a copy for drawing
    result = img_rgb.copy()

    # Downsample for quiver
    new_h = max(2, int(h * downsample))
    new_w = max(2, int(w * downsample))

    # Create sampling grid
    y_indices = np.linspace(0, h - 1, new_h, dtype=int)
    x_indices = np.linspace(0, w - 1, new_w, dtype=int)

    # Draw arrows
    # Note: scale parameter increases arrow length (unlike matplotlib where it decreases)
    arrow_scale = scale

    # Remove edge points (similar to matplotlib version)
    if len(y_indices) > 2 and len(x_indices) > 2:
        y_indices = y_indices[1:-1]
        x_indices = x_indices[1:-1]

    for y in y_indices:
        for x in x_indices:
            u = flow[y, x, 0] * arrow_scale
            v = flow[y, x, 1] * arrow_scale

            # Skip very small displacements
            if abs(u) < 0.5 and abs(v) < 0.5:
                continue

            # Draw arrow
            start_point = (int(x), int(y))
            end_point = (int(x + u), int(y + v))

            # Draw black outline for visibility
            cv2.arrowedLine(
                result,
                start_point,
                end_point,
                color=(0, 0, 0),  # Black outline
                thickness=2,
                tipLength=0.2,
                line_type=cv2.LINE_AA,
            )

            # Draw colored arrow on top
            cv2.arrowedLine(
                result,
                start_point,
                end_point,
                color=quiver_color,
                thickness=1,
                tipLength=0.2,
                line_type=cv2.LINE_AA,
            )

    # Add streamlines if requested
    if show_streamlines:
        # Create streamlines using line integral convolution approach
        # We'll trace particles through the flow field
        h, w = flow.shape[:2]

        # Apply light Gaussian smoothing to flow field for smoother streamlines
        # Very small kernel to maintain accuracy while reducing noise
        flow_smooth = cv2.GaussianBlur(flow, (3, 3), 0.5)

        # Create density grid to track visited cells (similar to matplotlib)
        # This prevents overlapping streamlines
        # Match matplotlib's approach: 30x30 grid at density=1
        # Scale density based on image size for better visual results
        base_size = 500  # Reference image size
        size_factor = np.sqrt((h * w) / (base_size * base_size))
        density = max(1.5, min(5.0, size_factor))  # Clamp between 0.5 and 2.0
        grid_nx = max(2, int(30 * density))
        grid_ny = max(2, int(30 * density))
        cell_size_x = max(1, int(np.ceil(w / grid_nx)))
        cell_size_y = max(1, int(np.ceil(h / grid_ny)))
        visited_grid = np.zeros((grid_ny, grid_nx), dtype=bool)

        # Create seed points in a grid (similar to matplotlib)
        # Use finer spacing for seed points to get more streamlines
        # Seed spacing should be finer than the density grid
        seed_spacing_x = max(2, cell_size_x // 2)
        seed_spacing_y = max(2, cell_size_y // 2)
        seed_points = []
        for y in range(seed_spacing_y // 2, h, seed_spacing_y):
            for x in range(seed_spacing_x // 2, w, seed_spacing_x):
                seed_points.append([x, y])

        # Trace streamlines from each seed point
        for seed in seed_points:
            for direction in (1.0, -1.0):
                streamline = []
                x, y = float(seed[0]), float(seed[1])

                cell_x = min(grid_nx - 1, int(x // cell_size_x))
                cell_y = min(grid_ny - 1, int(y // cell_size_y))
                if visited_grid[cell_y, cell_x]:
                    if direction == 1.0:
                        continue
                    else:
                        break
                visited_grid[cell_y, cell_x] = True
                cur_cx, cur_cy = cell_x, cell_y

                for _ in range(200):
                    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
                        break

                    ix, iy = int(x), int(y)
                    fx, fy = x - ix, y - iy

                    if ix < w - 1 and iy < h - 1:
                        u00 = flow_smooth[iy, ix, 0]
                        u10 = flow_smooth[iy, ix + 1, 0]
                        u01 = flow_smooth[iy + 1, ix, 0]
                        u11 = flow_smooth[iy + 1, ix + 1, 0]
                        u = (
                            (1 - fx) * (1 - fy) * u00
                            + fx * (1 - fy) * u10
                            + (1 - fx) * fy * u01
                            + fx * fy * u11
                        )

                        v00 = flow_smooth[iy, ix, 1]
                        v10 = flow_smooth[iy, ix + 1, 1]
                        v01 = flow_smooth[iy + 1, ix, 1]
                        v11 = flow_smooth[iy + 1, ix + 1, 1]
                        v = (
                            (1 - fx) * (1 - fy) * v00
                            + fx * (1 - fy) * v10
                            + (1 - fx) * fy * v01
                            + fx * fy * v11
                        )
                    else:
                        u = flow_smooth[iy, ix, 0]
                        v = flow_smooth[iy, ix, 1]

                    step_size = 0.35 * min(cell_size_x, cell_size_y)
                    nx, ny = (
                        x + direction * u * step_size,
                        y + direction * v * step_size,
                    )
                    if nx < 0 or nx >= w - 1 or ny < 0 or ny >= h - 1:
                        break

                    next_cx = min(grid_nx - 1, int(nx // cell_size_x))
                    next_cy = min(grid_ny - 1, int(ny // cell_size_y))
                    if (next_cx, next_cy) != (cur_cx, cur_cy):
                        if visited_grid[next_cy, next_cx]:
                            break
                        visited_grid[next_cy, next_cx] = True
                        cur_cx, cur_cy = next_cx, next_cy

                    streamline.append((int(x), int(y)))
                    x, y = nx, ny

                    if abs(u) < 0.02 and abs(v) < 0.02:
                        break

                if len(streamline) > 3:
                    pts = np.asarray(streamline, dtype=np.int32)
                    cv2.polylines(
                        result, [pts], False, streamline_color, 1, cv2.LINE_AA
                    )

    return result


def quiver_visualization(
    img: np.ndarray,
    w: np.ndarray,
    scale: float = 1.0,
    downsample: float = 0.03,
    show_streamlines: bool = True,
    backend: str = "matplotlib",
    return_array: bool = True,
    quiver_color: Tuple[int, int, int] = (255, 255, 255),
    streamline_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Create quiver visualization of displacement field overlaid on image.
    Automatically detects number of channels and applies appropriate mapping.

    Args:
        img: Input image (H, W) or (H, W, C)
        w: Displacement field with shape (H, W, 2)
        scale: Scale factor for quiver arrows
        downsample: Downsampling factor for quiver display (default 0.03)
        show_streamlines: Whether to show streamlines
        backend: Visualization backend - "matplotlib" or "opencv"
        return_array: If True, return numpy array; if False, display plot
        quiver_color: RGB color for quiver arrows (default white)
        streamline_color: RGB color for streamlines (default black)

    Returns:
        Visualization image as numpy array if return_array=True

    Raises:
        ImportError: If matplotlib is not available when using matplotlib backend
        ValueError: If invalid backend is specified
    """
    if backend not in ["matplotlib", "opencv"]:
        raise ValueError(f"Backend must be 'matplotlib' or 'opencv', got '{backend}'")

    if backend == "matplotlib" and not MATPLOTLIB_SUPPORTED:
        raise ImportError("Matplotlib backend requires 'matplotlib' library")
    # Ensure displacement field has correct shape
    if w.ndim != 3 or w.shape[2] != 2:
        raise ValueError(f"Displacement field must have shape (H, W, 2), got {w.shape}")

    # Use OpenCV backend if specified
    if backend == "opencv":
        return _quiver_visualization_opencv(
            img, w, scale, downsample, show_streamlines, quiver_color, streamline_color
        )

    # Otherwise use matplotlib backend
    # Ensure image is 3D
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    h, w_width, n_channels = img.shape

    # Determine visualization based on number of channels
    print(f"Processing image with {n_channels} channel(s)")

    if n_channels == 1:
        print("Using grayscale visualization")
        # Convert to RGB for display
        img_rgb = np.stack([img[:, :, 0]] * 3, axis=-1)
        if img_rgb.max() > 1:
            img_rgb = img_rgb / img_rgb.max()

    elif n_channels == 2:
        print("Using 2-channel color mapping")
        # Use the 2-channel color mapping
        img_rgb = get_visualization(img[:, :, 0], img[:, :, 1])

    elif n_channels == 3:
        print("Using direct RGB visualization")
        img_rgb = img.copy()
        if img_rgb.max() > 1:
            img_rgb = img_rgb / img_rgb.max()

    else:
        print(f"Using multispectral mapping for {n_channels} channels")
        img_rgb = multispectral_mapping(img)

    # Downsample displacement field for visualization
    new_h = max(2, int(h * downsample))
    new_w = max(2, int(w_width * downsample))
    w_small = cv2.resize(w, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create figure with aspect ratio matching the image
    # Calculate figure size based on image aspect ratio
    aspect_ratio = w_width / h
    if aspect_ratio >= 1:
        # Wide image: fix height, adjust width
        fig_width = 10 * aspect_ratio
        fig_height = 10
    else:
        # Tall image: fix width, adjust height
        fig_width = 10
        fig_height = 10 / aspect_ratio

    # Limit maximum figure size to avoid memory issues
    max_size = 20
    if fig_width > max_size:
        fig_width = max_size
        fig_height = max_size / aspect_ratio
    if fig_height > max_size:
        fig_height = max_size
        fig_width = max_size * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Display image
    ax.imshow(img_rgb, extent=[0, w_width, h, 0])

    # Create grid for quiver plot
    h_small, w_small_width = w_small.shape[:2]
    x = np.linspace(0, w_width, w_small_width)
    y = np.linspace(0, h, h_small)

    # Adjust grid for streamlines (add half spacing)
    if show_streamlines and len(x) > 1 and len(y) > 1:
        dx = (x[1] - x[0]) * 0.5
        dy = (y[1] - y[0]) * 0.5
        # Create seed points at half-grid offsets (matching MATLAB behavior)
        X_seeds, Y_seeds = np.meshgrid(x[:-1] + dx, y[:-1] + dy)
        seed_points = np.column_stack([X_seeds.ravel(), Y_seeds.ravel()])

        # Add streamlines with explicit seed points
        # Convert RGB color tuple to matplotlib color (0-1 range)
        stream_color = tuple(c / 255.0 for c in streamline_color)
        try:
            # Note: Negate V component because image coordinates have y increasing downward
            # while matplotlib display has y increasing upward (due to extent and ylim settings)
            ax.streamplot(
                x,
                y,
                w_small[:, :, 0],
                w_small[:, :, 1],
                start_points=seed_points,
                color=stream_color,
                density=1.0,
                linewidth=1,
                arrowsize=1.5,
            )
        except Exception as e:
            print(f"Warning: Streamlines failed to render: {e}")

    # Remove edge points for quiver
    if len(x) > 2 and len(y) > 2:
        x_quiv = x[1:-1]
        y_quiv = y[1:-1]
        w_quiv = w_small[1:-1, 1:-1, :]

        # Create meshgrid for quiver
        X_quiv, Y_quiv = np.meshgrid(x_quiv, y_quiv)

        # Add quiver plot
        # Convert RGB color tuple to matplotlib color (0-1 range)
        quiv_color = tuple(c / 255.0 for c in quiver_color)
        # Note: Negate V component to match image coordinate system
        ax.quiver(
            X_quiv,
            Y_quiv,
            w_quiv[:, :, 0],
            -w_quiv[:, :, 1],
            scale_units="xy",
            scale=1.0 / scale,
            width=0.003,
            color=quiv_color,
            alpha=0.9,
        )

    ax.set_xlim(0, w_width)
    ax.set_ylim(h, 0)
    ax.axis("off")

    # Remove padding/margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.margins(0, 0)

    if return_array:
        # Convert to array
        fig.canvas.draw()
        # Use buffer_rgba for newer matplotlib versions
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Remove alpha channel
        buf = buf[:, :, :3]

        # Resize to original dimensions
        result = cv2.resize(buf, (w_width, h), interpolation=cv2.INTER_LINEAR)

        plt.close(fig)
        return result
    else:
        plt.show()
        return None


def flow_to_color(flow, max_flow=None):
    UNKNOWN_FLOW_THRESH = 1e9

    u = np.array(flow[..., 0], dtype=np.float64).copy()
    v = np.array(flow[..., 1], dtype=np.float64).copy()

    idx_unknown = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH)
    u[idx_unknown] = 0
    v[idx_unknown] = 0

    rad = np.sqrt(u * u + v * v)
    if max_flow is not None and max_flow > 0:
        radmax = float(max_flow)
    else:
        radmax = float(np.max(rad))
    radmax = max(radmax, np.finfo(np.float64).eps)

    u /= radmax
    v /= radmax
    rad = np.sqrt(u * u + v * v)

    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.float64)
    col = 0
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col += YG
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col += CB
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col += BM
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255

    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int64)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    col0 = colorwheel[k0] / 255.0
    col1 = colorwheel[k1] / 255.0
    col = (1 - f[..., None]) * col0 + f[..., None] * col1

    mask = rad <= 1
    col[mask] = 1 - rad[mask, None] * (1 - col[mask])
    col[~mask] *= 0.75

    img = (col * 255).astype(np.uint8)
    img[idx_unknown] = 0
    return img
