import numpy as np
from suite2p.registration import rigid, nonrigid


def estimate_flow(fixed, moving, use_nonrigid=False):
    """
    Estimate optical flow using Suite2p's registration functions directly
    
    Args:
        fixed: Reference frame (H, W) or (H, W, C)
        moving: Frame to align (H, W) or (H, W, C)  
        use_nonrigid: If True, use nonrigid registration
    """
    # Handle multi-channel inputs
    if fixed.ndim == 3:
        fixed = fixed[..., 0]
    if moving.ndim == 3:
        moving = moving[..., 0]
    
    H, W = fixed.shape
    
    # Convert to float32 and normalize
    fixed = fixed.astype(np.float32)
    moving = moving.astype(np.float32)
    
    # Normalize to 0-1 range
    fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min() + 1e-8)
    moving = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)
    
    if use_nonrigid:
        # Nonrigid registration parameters
        # Match NoRMCorre parameters: grid_size=[16,16] for 512x512 
        # This translates to block_size ~ 512/16 = 32 pixels
        block_size = [max(32, H//16), max(32, W//16)]
        
        # Compute nonrigid shifts
        yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(
            Ly=H, Lx=W, block_size=block_size
        )
        
        # Get mask and reference image
        maskMul, maskOffset = rigid.compute_masks(
            refImg=fixed, maskSlope=0
        )
        
        # Compute reference FFT
        cfRefImg = np.fft.fft2(fixed * maskMul + maskOffset)
        
        # Compute shifts using phase correlation for each block
        # We need to extract shifts for each block region
        ymax1 = np.zeros(len(yblock), dtype=np.float32)
        xmax1 = np.zeros(len(xblock), dtype=np.float32)
        
        for idx, (yb, xb) in enumerate(zip(yblock, xblock)):
            y0, y1 = yb
            x0, x1 = xb
            
            # Extract block from moving image
            block_moving = moving[y0:y1, x0:x1]
            block_fixed = fixed[y0:y1, x0:x1]
            
            # Compute FFT of block reference (use block, not full image)
            cfBlockRef = np.fft.fft2(block_fixed)
            
            # Compute phase correlation for this block
            # Note: phasecorr expects (frames, H, W) shape
            block_data = block_moving[np.newaxis, :, :]
            
            dy, dx, _ = rigid.phasecorr(
                data=block_data,
                cfRefImg=cfBlockRef,
                maxregshift=0.2,  # Increased from 0.1 to allow larger shifts
                smooth_sigma_time=0
            )
            
            ymax1[idx] = dy[0]
            xmax1[idx] = dx[0]
        
        # Convert block shifts to dense flow field
        v = np.zeros((H, W, 2), dtype=np.float32)
        
        # Create coordinate grids for each block
        for idx, (yb, xb) in enumerate(zip(yblock, xblock)):
            dy = ymax1[idx]  # y shift for this block
            dx = xmax1[idx]  # x shift for this block
            
            # Get block boundaries
            y0, y1 = yb
            x0, x1 = xb
            
            # Apply shift to this block region
            v[y0:y1, x0:x1, 0] = dy
            v[y0:y1, x0:x1, 1] = dx
            
    else:
        # Rigid registration
        # Get mask for registration
        maskMul, maskOffset = rigid.compute_masks(
            refImg=fixed, maskSlope=0
        )
        
        # Compute reference FFT
        cfRefImg = np.fft.fft2(fixed * maskMul + maskOffset)
        
        # Compute shift using phase correlation
        # Note: phasecorr expects (frames, H, W) shape
        frames = moving[np.newaxis, :, :]
        
        # Compute rigid shift
        ymax, xmax, cmax = rigid.phasecorr(
            data=frames,
            cfRefImg=cfRefImg,
            maxregshift=0.2,  # Increased from 0.1 to allow larger shifts
            smooth_sigma_time=0
        )
        
        # Create uniform displacement field in (H, W, 2) format
        v = np.zeros((H, W, 2), dtype=np.float32)
        v[:, :, 0] = ymax[0]  # y shift
        v[:, :, 1] = xmax[0]  # x shift
    
    return v