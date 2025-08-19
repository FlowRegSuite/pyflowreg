import numpy as np
import cv2
from jnormcorre import normcorre


def estimate_flow(fixed, moving, grid_size=(64, 64), max_shift=10, stride=32):
    if fixed.ndim == 3:
        fixed = fixed[..., 0]
    if moving.ndim == 3:
        moving = moving[..., 0]
    
    Y = np.stack([fixed, moving], 0)[None]
    
    res = normcorre.normcorre_batch(
        Y,
        grid_size=grid_size,
        max_shifts=max_shift,
        strides=(stride, stride)
    )
    
    # Get shifts for second frame
    sh = res["shifts_2d"][0, 1]  # batch 0, frame 1
    
    H, W = fixed.shape
    v = np.zeros((2, H, W), np.float32)
    
    # Check if it's rigid (1D) or nonrigid (grid)
    if sh.ndim == 1:
        # Rigid case - single shift values
        dx, dy = sh[0], sh[1]
        v[0] += dy
        v[1] += dx
        return v
    
    # Nonrigid case - grid of shifts
    dx = sh[..., 0]
    dy = sh[..., 1]
    # Expand the grid to full resolution
    v[0] = cv2.resize(dy, (W, H), interpolation=cv2.INTER_NEAREST)
    v[1] = cv2.resize(dx, (W, H), interpolation=cv2.INTER_NEAREST)
    return v