"""
Image processing utilities for motion correction.
Provides normalization and filtering functions extracted from CompensateRecording.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional, Literal
from collections import deque


def normalize(
    arr: np.ndarray,
    ref: Optional[np.ndarray] = None,
    channel_normalization: Literal["together", "separate"] = "together",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize array to [0,1] range.

    Args:
        arr: Array to normalize, shape (H,W,C) or (T,H,W,C)
        ref: Optional reference for normalization ranges (MATLAB compatibility)
        channel_normalization: 'separate' for per-channel, 'together' for global
        eps: Small value to avoid division by zero

    Returns:
        Normalized array in [0,1] range
    """
    if channel_normalization == "separate":
        # Per-channel normalization
        result = np.zeros_like(arr, dtype=np.float64)

        if arr.ndim == 3:  # (H,W,C)
            for c in range(arr.shape[2]):
                if ref is not None and ref.ndim >= 3:
                    # Use reference's min/max for this channel
                    ref_channel = ref[..., c]
                    min_val = ref_channel.min()
                    max_val = ref_channel.max()
                else:
                    # Use array's own min/max
                    channel = arr[..., c]
                    min_val = channel.min()
                    max_val = channel.max()
                result[..., c] = (arr[..., c] - min_val) / (max_val - min_val + eps)

        elif arr.ndim == 4:  # (T,H,W,C)
            for c in range(arr.shape[3]):
                if ref is not None and ref.ndim >= 3:
                    ref_channel = ref[..., c]
                    min_val = ref_channel.min()
                    max_val = ref_channel.max()
                else:
                    channel = arr[..., c]
                    min_val = channel.min()
                    max_val = channel.max()
                result[..., c] = (arr[..., c] - min_val) / (max_val - min_val + eps)
        else:
            # 2D or unsupported, use global normalization
            if ref is not None:
                min_val = ref.min()
                max_val = ref.max()
            else:
                min_val = arr.min()
                max_val = arr.max()
            return (arr - min_val) / (max_val - min_val + eps)

        return result
    else:
        # Global normalization
        if ref is not None:
            min_val = ref.min()
            max_val = ref.max()
        else:
            min_val = arr.min()
            max_val = arr.max()
        return (arr - min_val) / (max_val - min_val + eps)


def apply_gaussian_filter(
    arr: np.ndarray, sigma: np.ndarray, mode: str = "reflect", truncate: float = 4.0
) -> np.ndarray:
    """
    Apply Gaussian filtering matching MATLAB's imgaussfilt3 for multichannel data.

    Args:
        arr: Input array, shape (H,W,C) or (T,H,W,C)
        sigma: Standard deviation for Gaussian kernel
               - array shape (3,): [sy, sx, st] for all channels
               - array shape (n_channels, 3): per-channel sigmas
        mode: Boundary handling mode
        truncate: Truncate filter at this many standard deviations

    Returns:
        Filtered array
    """
    sigma = np.asarray(sigma)

    if arr.ndim == 3:  # (H,W,C) - spatial only
        result = np.zeros_like(arr, dtype=np.float64)
        for c in range(arr.shape[2]):
            if sigma.ndim == 2:  # Per-channel sigmas
                s = sigma[min(c, len(sigma) - 1), :2]  # Use only spatial components
            else:
                s = sigma[:2]  # Use first two components
            result[..., c] = gaussian_filter(
                arr[..., c], sigma=s, mode=mode, truncate=truncate
            )
        return result

    elif arr.ndim == 4:  # (T,H,W,C) - spatiotemporal
        result = np.zeros_like(arr, dtype=np.float64)
        for c in range(arr.shape[3]):  # C is last dimension
            if sigma.ndim == 2:  # Per-channel sigmas
                s = sigma[min(c, len(sigma) - 1)]
                # Reorder from (sy, sx, st) to (st, sy, sx) for scipy
                s_3d = (s[2], s[0], s[1])
            else:
                s_3d = (sigma[2], sigma[0], sigma[1])

            # Apply 3D Gaussian filter
            result[..., c] = gaussian_filter(
                arr[..., c], sigma=s_3d, mode=mode, truncate=truncate
            )
        return result

    else:
        # 2D or unsupported dimensionality
        return arr


def gaussian_filter_1d_half_kernel(
    buffer: deque, sigma_t: float, mode: str = "reflect", truncate: float = 4.0
) -> np.ndarray:
    """
    Fast 1D Gaussian filter using half kernel for temporal dimension.
    Optimized for real-time filtering with circular buffer.

    Args:
        buffer: deque of 2D filtered frames [oldest...newest]
        sigma_t: Temporal standard deviation
        mode: Boundary handling mode (unused, kept for API consistency)
        truncate: Truncate filter at this many standard deviations

    Returns:
        Temporally filtered current frame (last in buffer)
    """
    if not buffer or len(buffer) == 0:
        return None

    if len(buffer) == 1:
        return buffer[-1].copy()

    # No temporal filtering if sigma is 0
    if sigma_t <= 0:
        return buffer[-1].copy()

    # Create half Gaussian kernel
    kernel_radius = int(truncate * sigma_t + 0.5)
    kernel_size = min(kernel_radius + 1, len(buffer))  # Half kernel including center

    # Generate half Gaussian kernel weights (only past frames + current)
    x = np.arange(kernel_size, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma_t) ** 2)
    kernel = kernel / kernel.sum()

    # Apply weighted average using half kernel
    # buffer[-1] is current frame, buffer[-2] is previous, etc.
    result = np.zeros_like(buffer[-1], dtype=np.float64)

    for i in range(kernel_size):
        # Index from the end of buffer
        frame_idx = -(i + 1)
        result += kernel[i] * buffer[frame_idx]

    return result.astype(buffer[-1].dtype)
