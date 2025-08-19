import numpy as np
import tempfile
import os
import cv2
from scipy.ndimage import gaussian_filter
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params


def mat2gray(x):
    """Convert to grayscale matching reference implementation"""
    x = x.astype(np.float64)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx > mn:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x)


def preprocess_frames(frames, sigma=(0.5, 0.5, 0.001)):
    """Apply gaussian filtering matching reference implementation"""
    if frames.ndim == 3:  # (H, W, C)
        H, W, C = frames.shape
        # Create a 4D array with single time dimension for filtering
        frames_4d = frames[:, :, :, np.newaxis]  # (H, W, C, 1)
        out = np.empty_like(frames_4d, dtype=np.float64)
        for c in range(C):
            out[:, :, c, 0] = gaussian_filter(frames_4d[:, :, c, 0], 
                                             sigma=sigma[:2], mode="nearest")
        return out[:, :, :, 0]  # Return (H, W, C)
    elif frames.ndim == 2:  # (H, W)
        return gaussian_filter(frames, sigma=sigma[:2], mode="nearest")
    return frames


def estimate_flow(fixed, moving):
    """
    Estimate optical flow using CaImAn's NoRMCorre implementation
    Parameters matched to reference implementation.
    
    Args:
        fixed: Reference frame (H, W) or (H, W, C)
        moving: Frame to align (H, W) or (H, W, C)
    """
    # Preprocess frames matching reference
    sigma = (0.5, 0.5, 0.001)
    fixed_proc = preprocess_frames(fixed, sigma)
    moving_proc = preprocess_frames(moving, sigma)
    
    # Use first channel if multi-channel
    if fixed_proc.ndim == 3:
        fixed_proc = fixed_proc[..., 0]
    if moving_proc.ndim == 3:
        moving_proc = moving_proc[..., 0]
    
    # Apply mat2gray normalization
    fixed_norm = mat2gray(fixed_proc)
    moving_norm = mat2gray(moving_proc)
    
    # Create temporary file for the movie
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        fname = tmp.name
    
    try:
        # Save as tiff movie - normalized frames
        movie = np.stack([fixed_norm, moving_norm], axis=0).astype(np.float32)
        cm.movie(movie).save(fname)
        
        # Parameters matching reference implementation
        H, W = fixed_norm.shape
        
        # Match reference: grid_size=(16, 16), max_shift=25
        # CaImAn uses strides+overlaps to define patch size
        # Reference grid_size of 16x16 means patches every 16 pixels
        # We'll use small overlaps to get smooth interpolation
        strides = (16, 16)  # Match reference grid_size
        overlaps = (8, 8)   # Half stride for smooth interpolation
        
        opts_dict = {
            'max_shifts': (25, 25),  # Match reference max_shift=25
            'niter_rig': 1,
            'pw_rigid': True,  # Always use piecewise-rigid like reference
            'strides': strides,
            'overlaps': overlaps,
            'max_deviation_rigid': 15,  # Match reference max_dev
            'border_nan': 'copy',
            'upsample_factor_grid': 4,  # For smoother interpolation
            'use_cuda': False,
            'nonneg_movie': True
        }
        
        opts = params.CNMFParams(params_dict=opts_dict)
        
        # Run motion correction with fixed template
        mc = MotionCorrect(fname, dview=None, **opts.get_group('motion'))
        mc.motion_correct(save_movie=False, template=fixed_norm)
        
        # Extract shifts for the second frame (index 1)
        # CaImAn stores piecewise-rigid shifts differently
        H, W = fixed_norm.shape
        
        if hasattr(mc, 'x_shifts_els') and len(mc.x_shifts_els) > 0:
            # Piecewise-rigid shifts are available
            # Get shifts for frame 1 (second frame, 0-indexed)
            frame_idx = 1
            
            # x_shifts_els and y_shifts_els are lists of arrays
            # Each element corresponds to a frame
            if frame_idx < len(mc.x_shifts_els):
                x_shifts = mc.x_shifts_els[frame_idx]  # x shifts for all patches
                y_shifts = mc.y_shifts_els[frame_idx]  # y shifts for all patches
                
                # Get the coordinate grid for the patches
                if hasattr(mc, 'coord_shifts_els') and len(mc.coord_shifts_els) > 0:
                    coords = mc.coord_shifts_els[0]  # Coordinates are same for all frames
                    
                    # coords contains the center positions of each patch
                    # Create full resolution flow field by interpolation
                    v = np.zeros((2, H, W), dtype=np.float32)
                    
                    # Build a grid of shift values at patch centers
                    # Then interpolate to full resolution
                    if len(x_shifts) > 1:
                        # Multiple patches - need to interpolate
                        # Extract unique x and y coordinates
                        y_coords = np.unique([c[0] for c in coords])
                        x_coords = np.unique([c[1] for c in coords])
                        
                        n_y = len(y_coords)
                        n_x = len(x_coords)
                        
                        # Reshape shifts into grid
                        if len(x_shifts) == n_y * n_x:
                            x_grid = x_shifts.reshape(n_y, n_x)
                            y_grid = y_shifts.reshape(n_y, n_x)
                            
                            # Interpolate to full resolution
                            # Note: cv2.resize expects (width, height) not (height, width)
                            v[1] = cv2.resize(x_grid.astype(np.float32), (W, H), 
                                            interpolation=cv2.INTER_LINEAR)
                            v[0] = cv2.resize(y_grid.astype(np.float32), (W, H), 
                                            interpolation=cv2.INTER_LINEAR)
                        else:
                            # Irregular grid - use average shift
                            v[1] += np.mean(x_shifts)
                            v[0] += np.mean(y_shifts)
                    else:
                        # Single patch - uniform shift
                        v[1] += float(x_shifts[0]) if len(x_shifts) > 0 else 0
                        v[0] += float(y_shifts[0]) if len(y_shifts) > 0 else 0
                else:
                    # No coordinate info - fall back to average shifts
                    v = np.zeros((2, H, W), dtype=np.float32)
                    v[1] += np.mean(x_shifts) if len(x_shifts) > 0 else 0
                    v[0] += np.mean(y_shifts) if len(y_shifts) > 0 else 0
            else:
                # Frame index out of range
                v = np.zeros((2, H, W), dtype=np.float32)
        elif hasattr(mc, 'shifts_rig') and len(mc.shifts_rig) > 1:
            # Fall back to rigid shifts
            shifts = mc.shifts_rig[1]  # Get shifts for second frame
            v = np.zeros((2, H, W), dtype=np.float32)
            v[0] += shifts[0]  # y shift  
            v[1] += shifts[1]  # x shift
        else:
            # No shifts found
            v = np.zeros((2, H, W), dtype=np.float32)
            
    finally:
        # Clean up temporary file
        if os.path.exists(fname):
            try:
                os.remove(fname)
            except:
                pass  # Windows sometimes has file locking issues
    
    # Return flow field in (H, W, 2) format to match other methods
    # Transpose from (2, H, W) to (H, W, 2)
    return np.transpose(v, (1, 2, 0))