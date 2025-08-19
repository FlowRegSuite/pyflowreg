import numpy as np
import pyflowreg as pfr
from scipy.ndimage import gaussian_filter


def preprocess_pyflowreg(frame1, frame2, apply_gaussian=True, sigma=1):
    """
    Preprocess frames for PyFlowReg matching synth_evaluation.py exactly.
    
    Args:
        frame1: Reference frame (H, W, C) already in float32
        frame2: Moving frame (H, W, C) already in float32
        apply_gaussian: If True, apply Gaussian filtering
        sigma: Gaussian filter sigma for spatial dimensions
    
    Returns:
        Preprocessed frame1, frame2
    """
    # Apply Gaussian filtering if requested (should be done BEFORE normalization)
    if apply_gaussian:
        # Apply per-channel Gaussian filtering
        for c in range(frame1.shape[-1]):
            frame1[..., c] = gaussian_filter(frame1[..., c], sigma, mode='constant')
            frame2[..., c] = gaussian_filter(frame2[..., c], sigma, mode='constant')
    
    # Normalize using frame1's statistics for BOTH frames (exactly like synth_evaluation.py)
    mins = frame1.min(axis=(0, 1), keepdims=True)  # shape (1,1,C)
    maxs = frame1.max(axis=(0, 1), keepdims=True)  # shape (1,1,C)
    ranges = maxs - mins
    ranges = np.where(ranges < 1e-6, 1.0, ranges)  # Avoid division by zero
    
    frame1 = (frame1 - mins) / ranges
    frame2 = (frame2 - mins) / ranges
    
    return frame1.astype(np.float32), frame2.astype(np.float32)


def estimate_flow(fixed, moving, preprocess=True, **kw):
    """
    Estimate optical flow using PyFlowReg.
    
    Args:
        fixed: Reference frame (H, W) or (H, W, C)
        moving: Moving frame (H, W) or (H, W, C)
        preprocess: If True, apply PyFlowReg-specific preprocessing
        **kw: Additional parameters for pfr.get_displacement
    """
    # Handle 2D inputs
    if fixed.ndim == 2:
        fixed = fixed[..., np.newaxis]
    if moving.ndim == 2:
        moving = moving[..., np.newaxis]
    
    # Apply preprocessing if requested
    if preprocess:
        # Note: For benchmarks, Gaussian filtering is already applied in data loading
        # So we only apply normalization here
        fixed, moving = preprocess_pyflowreg(fixed, moving, apply_gaussian=True)
    
    # Adapt weights to number of channels
    C = fixed.shape[-1]
    
    base_params = dict(
        alpha=(8, 8),
        iterations=100,
        a_data=0.45,
        a_smooth=1.0,
        weight=np.full(C, 1.0/C, np.float32),  # Equal weights for all channels
        levels=50,
        eta=0.8,
        update_lag=5,
        min_level=0
    )
    base_params.update(kw)
    
    w = pfr.get_displacement(fixed, moving, **base_params)
    
    # pfr.get_displacement already returns (H, W, 2) format
    return w.astype(np.float32)