"""
Utility functions for image warping and valid mask computation.

Provides helpers for computing backward warping validity masks and
binary mask warping with nearest-neighbor interpolation.
"""

import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Optional


def backward_valid_mask(u, v):
    """
    Compute the valid region mask after backward warping.

    Given displacement fields u and v, computes which pixels remain
    in-bounds after applying the backward warp. This is essential for
    tracking valid data regions in motion-corrected sequences.

    Parameters
    ----------
    u : ndarray, shape (H, W)
        Horizontal displacement field (x-direction)
    v : ndarray, shape (H, W)
        Vertical displacement field (y-direction)

    Returns
    -------
    valid_mask : ndarray, shape (H, W), dtype=bool
        Boolean mask where True indicates pixel remains in-bounds

    Notes
    -----
    Mirrors MATLAB idx_warp computation from imregister_wrapper_w.
    The backward warp maps from output coordinates (y, x) to input
    coordinates (y+v, x+u). A pixel is valid if the mapped location
    falls within [0, H) x [0, W).

    Examples
    --------
    >>> u = np.ones((10, 10)) * 2.5  # shift right by 2.5 pixels
    >>> v = np.zeros((10, 10))
    >>> mask = backward_valid_mask(u, v)
    >>> # Right edge pixels will be False (out of bounds)
    """
    H, W = u.shape

    # Generate coordinate grid
    gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Compute mapped coordinates
    mx = gx + u
    my = gy + v

    # Check bounds
    valid = (mx >= 0) & (mx < W) & (my >= 0) & (my < H)

    return valid.astype(bool)


def imregister_binary(mask, u, v):
    """
    Warp a binary mask using nearest-neighbor interpolation.

    Applies backward warping to a binary mask and composes with the
    in-bounds mask to prevent artifacts from extrapolation.

    Parameters
    ----------
    mask : ndarray, shape (H, W)
        Binary mask to warp (will be converted to bool)
    u : ndarray, shape (H, W)
        Horizontal displacement field (x-direction)
    v : ndarray, shape (H, W)
        Vertical displacement field (y-direction)

    Returns
    -------
    warped_mask : ndarray, shape (H, W), dtype=bool
        Warped mask with out-of-bounds pixels set to False

    Notes
    -----
    Mirrors MATLAB mask warping in get_session_valid_index_v3:
        [reg_m, idx_warp] = imregister_wrapper_w(
            double(m), w, zeros(size(m)), 'nearest')
        aligned_valid_masks{i} = (reg_m > 0.5) & idx_warp

    Uses cv2.INTER_NEAREST for binary interpolation to avoid
    intermediate values.

    Examples
    --------
    >>> mask = np.ones((10, 10), dtype=bool)
    >>> u = np.ones((10, 10)) * 2.0  # shift right
    >>> v = np.zeros((10, 10))
    >>> warped = imregister_binary(mask, u, v)
    >>> # Right 2 columns will be False
    """
    H, W = mask.shape

    # Generate coordinate grid
    gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Compute map coordinates (clipped for cv2.remap safety)
    mx = np.clip((gx + u).astype(np.float32), 0, W - 1)
    my = np.clip((gy + v).astype(np.float32), 0, H - 1)

    # Warp with nearest-neighbor
    warped = cv2.remap(
        mask.astype(np.float32),
        mx,
        my,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Compute in-bounds mask
    in_bounds = (gx + u >= 0) & (gx + u < W) & (gy + v >= 0) & (gy + v < H)

    # Combine: warped mask AND in-bounds
    return (warped > 0.5) & in_bounds


def compute_batch_valid_masks(w):
    """
    Compute valid masks for a batch of displacement fields.

    Parameters
    ----------
    w : ndarray, shape (T, H, W, 2)
        Batch of displacement fields where w[..., 0] = u, w[..., 1] = v

    Returns
    -------
    valid_masks : ndarray, shape (T, H, W), dtype=uint8
        Valid masks for each frame (0 or 255 for compatibility)

    Notes
    -----
    Used in BatchMotionCorrector to persist per-frame validity.
    Returns uint8 for efficient HDF5 storage.
    """
    T = w.shape[0]
    H, W = w.shape[1:3]

    valid_batch = np.empty((T, H, W), dtype=np.uint8)

    for t in range(T):
        u = w[t, ..., 0]
        v = w[t, ..., 1]
        valid_batch[t] = backward_valid_mask(u, v).astype(np.uint8) * 255

    return valid_batch


def imregister_wrapper(f2_level, u, v, f1_level, interpolation_method="cubic"):
    """
    Backward warp of moving image using displacement field.

    Performs backward registration by warping f2_level toward f1_level using
    displacement field (u, v) with bicubic or bilinear interpolation via cv2.remap.
    Out-of-bounds pixels are replaced with corresponding pixels from f1_level.

    Parameters
    ----------
    f2_level : np.ndarray
        Moving image to warp, shape (H, W) or (H, W, C)
    u : np.ndarray
        Horizontal displacement field, shape (H, W)
    v : np.ndarray
        Vertical displacement field, shape (H, W)
    f1_level : np.ndarray
        Fixed (reference) image, shape (H, W) or (H, W, C)
    interpolation_method : str, default='cubic'
        Interpolation method: 'cubic' (bicubic) or 'linear' (bilinear).
        Defaults to bicubic following Sun et al. best practices.

    Returns
    -------
    warped : np.ndarray
        Backward-warped image, shape (H, W) or (H, W, C)

    Notes
    -----
    The displacement convention is: warped_pos = original_pos + (u, v)
    Out-of-bounds regions use values from f1_level to maintain continuity.

    Bicubic interpolation is more accurate than bilinear for optical flow
    estimation and is the recommended default.

    References
    ----------
    .. [1] Sun, D., Roth, S., and Black, M. J. "Secrets of Optical Flow
       Estimation and Their Principles", CVPR 2010.
    """
    if f2_level.ndim == 2:
        f2_level = f2_level[:, :, None]
        f1_level = f1_level[:, :, None]
    # f2_level = f2_level[1:-1, 1:-1]
    # f1_level = f1_level[1:-1, 1:-1]
    # u = u[1:-1, 1:-1]
    # v = v[1:-1, 1:-1]
    H, W, C = f2_level.shape
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    map_x = (grid_x + u).astype(np.float32)
    map_y = (grid_y + v).astype(np.float32)
    out_of_bounds = (map_x < 0) | (map_x >= W) | (map_y < 0) | (map_y >= H)
    map_x_clipped = np.clip(map_x, 0, W - 1).astype(np.float32)
    map_y_clipped = np.clip(map_y, 0, H - 1).astype(np.float32)
    if interpolation_method.lower() == "cubic":
        interp = cv2.INTER_CUBIC
    elif interpolation_method.lower() == "linear":
        interp = cv2.INTER_LINEAR
    else:
        raise ValueError("Unsupported interpolation method. Use 'linear' or 'cubic'.")
    warped = np.empty_like(f2_level, dtype=np.float32)
    for c in range(C):
        warped[:, :, c] = cv2.remap(
            f2_level[:, :, c],
            map_x_clipped,
            map_y_clipped,
            interpolation=interp,
            borderMode=cv2.BORDER_REPLICATE,
        )

    for c in range(C):
        warped[:, :, c][out_of_bounds] = f1_level[:, :, c][out_of_bounds]
    if warped.shape[2] == 1:
        warped = warped[:, :, 0]
    return warped


def warpingDepth(eta, levels, m, n):
    """
    Calculate maximum pyramid depth for given dimension and warping factor.

    Determines how many pyramid levels can be computed given the downsampling
    factor eta before the dimension becomes too small (< 10 pixels) for
    reliable optical flow estimation. At pyramid level i, the dimension size
    is dim * eta^i, where dim = min(m, n).

    Parameters
    ----------
    eta : float
        Pyramid downsampling factor per level (0 < eta <= 1)
    levels : int
        Maximum number of levels to attempt
    m : int
        First dimension
    n : int
        Second dimension

    Returns
    -------
    warpingdepth : int
        Maximum pyramid depth satisfying: round(dim * eta^i) >= 10,
        where dim = min(m, n). Approximately floor(log(10/dim) / log(eta)).

    Notes
    -----
    When called from get_displacement with (m, m) for height and (n, n) for
    width, this enables independent pyramid depth computation per dimension,
    allowing narrow ROIs to achieve large displacements along their longer
    dimension without being limited by the shorter dimension.
    """
    min_dim = min(m, n)
    warpingdepth = 0
    for _ in range(levels):
        warpingdepth += 1
        min_dim *= eta
        if round(min_dim) < 10:
            break
    return warpingdepth


def align_sequence(
    batch: np.ndarray,
    displacement: np.ndarray,
    reference: np.ndarray,
    interpolation_method: str = "cubic",
    n_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Apply displacement field to align a batch of frames to a reference.

    This function warps each frame in a batch using the provided displacement
    field to align them to a common reference frame. Uses multiprocessing
    for efficient batch processing.

    Parameters
    ----------
    batch : ndarray, shape (T, H, W, C) or (T, H, W)
        Batch of frames to align
    displacement : ndarray, shape (H, W, 2)
        Displacement field where displacement[..., 0] = u (horizontal),
        displacement[..., 1] = v (vertical)
    reference : ndarray, shape (H, W, C) or (H, W)
        Reference image used to fill out-of-bounds regions
    interpolation_method : str, default='cubic'
        Interpolation method: 'cubic' or 'linear'
    n_workers : int, optional
        Number of parallel workers. If None, uses cpu_count()

    Returns
    -------
    aligned_batch : ndarray
        Aligned frames with same shape and dtype as input batch

    Examples
    --------
    >>> batch = np.random.rand(100, 512, 512, 1)
    >>> displacement = np.zeros((512, 512, 2))  # No displacement
    >>> reference = np.mean(batch, axis=0)
    >>> aligned = align_sequence(batch, displacement, reference)
    """
    # Track if we need to add/remove channel dimension
    added_channel_dim = False
    if batch.ndim == 3:
        batch = batch[..., np.newaxis]
        added_channel_dim = True
    if reference.ndim == 2:
        reference = reference[..., np.newaxis]

    T, H, W, C = batch.shape
    u = displacement[..., 0]
    v = displacement[..., 1]

    # Prepare reference as float64
    reference_f64 = reference.astype(np.float64, copy=False)

    # Set up multiprocessing
    if n_workers is None:
        n_workers = min(cpu_count(), T)

    # Process frames
    if n_workers > 1:
        # Multiprocessing requires a picklable function with single argument
        # So we use functools.partial to bind the fixed parameters
        from functools import partial

        # Create partial function with fixed displacement and reference
        warp_func = partial(
            imregister_wrapper,
            u=u,
            v=v,
            f1_level=reference_f64,
            interpolation_method=interpolation_method,
        )

        # Convert frames to float64 for processing
        frames_f64 = [batch[t].astype(np.float64, copy=False) for t in range(T)]

        with Pool(processes=n_workers) as pool:
            warped_frames = pool.map(warp_func, frames_f64)
    else:
        # Sequential processing for small batches
        warped_frames = []
        for t in range(T):
            warped = imregister_wrapper(
                batch[t].astype(np.float64, copy=False),
                u,
                v,
                reference_f64,
                interpolation_method=interpolation_method,
            )
            warped_frames.append(warped)

    # Stack results and preserve dtype
    aligned_batch = np.empty_like(batch)
    for t, warped in enumerate(warped_frames):
        if warped.ndim == 2:
            warped = warped[..., np.newaxis]
        aligned_batch[t] = warped.astype(batch.dtype, copy=False)

    # Remove singleton channel dimension only if we added it
    if added_channel_dim and aligned_batch.shape[-1] == 1:
        aligned_batch = aligned_batch[..., 0]

    return aligned_batch
