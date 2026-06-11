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
    Normalize an array to the [0, 1] range.

    Parameters
    ----------
    arr : np.ndarray
        Array to normalize, shape (H, W, C) or (T, H, W, C).
    ref : np.ndarray, optional
        Reference array whose min/max values define the normalization
        range instead of ``arr``'s own values (MATLAB compatibility).
        Values in ``arr`` outside the reference range map outside [0, 1].
    channel_normalization : {"together", "separate"}, optional
        ``"separate"`` normalizes each channel independently using
        per-channel min/max; ``"together"`` uses a single global range.
        Default is ``"together"``.
    eps : float, optional
        Small value added to the denominator to avoid division by zero.
        Default is 1e-8.

    Returns
    -------
    np.ndarray
        Normalized array scaled to the [0, 1] range. Per-channel
        normalization returns a float64 array.

    Notes
    -----
    With ``channel_normalization="separate"``, per-channel reference
    ranges are only used when ``ref`` has at least 3 dimensions;
    otherwise the array's own per-channel min/max is used. Inputs that
    are not 3D or 4D fall back to global normalization.
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

    Each channel is filtered independently with
    ``scipy.ndimage.gaussian_filter``. 3D inputs (H, W, C) are filtered
    spatially only; 4D inputs (T, H, W, C) are filtered spatiotemporally.

    Parameters
    ----------
    arr : np.ndarray
        Input array, shape (H, W, C) or (T, H, W, C).
    sigma : np.ndarray
        Standard deviations of the Gaussian kernel ordered as
        ``[sx, sy, st]`` — ``sx`` smooths along the width/columns (W),
        ``sy`` along the height/rows (H), ``st`` along frames (T) —
        matching MATLAB's ``imgaussfilt``/``imgaussfilt3`` convention and
        the ``OFOptions.sigma`` field. Shape (3,) applies the same sigmas
        to all channels; shape (n_channels, 3) gives per-channel sigmas.
        For (H, W, C) input only the spatial components ``[sx, sy]`` are
        used.
    mode : str, optional
        Boundary handling mode passed to ``scipy.ndimage.gaussian_filter``.
        Default is "reflect".
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    np.ndarray
        Filtered float64 array with the same shape as ``arr``. Inputs
        that are not 3D or 4D are returned unchanged.
    """
    sigma = np.asarray(sigma)

    if arr.ndim == 3:  # (H,W,C) - spatial only
        result = np.zeros_like(arr, dtype=np.float64)
        for c in range(arr.shape[2]):
            if sigma.ndim == 2:  # Per-channel sigmas
                s = sigma[min(c, len(sigma) - 1), :2]  # Use only spatial components
            else:
                s = sigma[:2]  # Use first two components
            # Reorder from (sx, sy) to scipy's axis order (sy, sx) for (H, W)
            result[..., c] = gaussian_filter(
                arr[..., c], sigma=(s[1], s[0]), mode=mode, truncate=truncate
            )
        return result

    elif arr.ndim == 4:  # (T,H,W,C) - spatiotemporal
        result = np.zeros_like(arr, dtype=np.float64)
        for c in range(arr.shape[3]):  # C is last dimension
            if sigma.ndim == 2:  # Per-channel sigmas
                s = sigma[min(c, len(sigma) - 1)]
            else:
                s = sigma
            # Reorder from (sx, sy, st) to scipy's axis order (st, sy, sx)
            # for the (T, H, W) per-channel volume
            s_3d = (s[2], s[1], s[0])

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
    Filter the newest buffered frame with a causal half-kernel 1D Gaussian.

    Optimized for real-time temporal filtering with a circular buffer:
    only the current frame and past frames contribute, weighted by the
    causal half of a normalized Gaussian kernel.

    Parameters
    ----------
    buffer : collections.deque
        Buffer of spatially filtered frames ordered [oldest, ..., newest].
    sigma_t : float
        Temporal standard deviation. If <= 0, the newest frame is
        returned unfiltered.
    mode : str, optional
        Boundary handling mode (unused, kept for API consistency).
        Default is "reflect".
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    np.ndarray or None
        Temporally filtered newest frame, cast to the dtype of the
        newest buffered frame. Returns ``None`` if the buffer is empty;
        returns a copy of the newest frame if the buffer holds a single
        frame or ``sigma_t <= 0``.
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
