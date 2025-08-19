import numpy as np
import ants


def estimate_flow(fixed, moving, transform="SyN", metric="CC", use_multichannel=True):
    if fixed.ndim == 2:
        fixed = fixed[..., np.newaxis]
    if moving.ndim == 2:
        moving = moving[..., np.newaxis]
    
    # Use first channel as primary
    f_primary = ants.from_numpy(fixed[..., 0].astype(np.float32))
    m_primary = ants.from_numpy(moving[..., 0].astype(np.float32))
    
    # Build multivariate extras for all additional channels
    multivariate_extras = []
    if use_multichannel and fixed.shape[-1] > 1:
        for c in range(1, fixed.shape[-1]):
            f_secondary = ants.from_numpy(fixed[..., c].astype(np.float32))
            m_secondary = ants.from_numpy(moving[..., c].astype(np.float32))
            multivariate_extras.append([metric, f_secondary, m_secondary, 1, 4])
    
    tx = ants.registration(
        fixed=f_primary,
        moving=m_primary,
        type_of_transform=transform,
        flow_sigma=3,
        total_sigma=0,
        syn_metric=metric,
        multivariate_extras=multivariate_extras if multivariate_extras else None
    )
    
    # Get displacement field
    H, W = fixed.shape[:2]
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    
    X_warped = ants.apply_transforms(
        fixed=ants.from_numpy(X),
        moving=ants.from_numpy(X),
        transformlist=tx['fwdtransforms']
    ).numpy()
    
    Y_warped = ants.apply_transforms(
        fixed=ants.from_numpy(Y),
        moving=ants.from_numpy(Y),
        transformlist=tx['fwdtransforms']
    ).numpy()
    
    # Return in (H, W, 2) format
    v = np.stack([Y_warped - Y, X_warped - X], axis=-1).astype(np.float32)
    
    return v