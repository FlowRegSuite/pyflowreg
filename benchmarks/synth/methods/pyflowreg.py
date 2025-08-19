import numpy as np
import pyflowreg as pfr


def estimate_flow(fixed, moving, **kw):
    # Handle 2D inputs
    if fixed.ndim == 2:
        fixed = fixed[..., np.newaxis]
    if moving.ndim == 2:
        moving = moving[..., np.newaxis]
    
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