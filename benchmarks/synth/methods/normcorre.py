import numpy as np
import tempfile
import os
import cv2
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params


def estimate_flow(fixed, moving, pw_rigid=True, max_shifts=(10, 10), strides=(64, 64), overlaps=(32, 32)):
    """
    Estimate optical flow using CaImAn's NoRMCorre implementation
    
    Args:
        fixed: Reference frame
        moving: Frame to align
        pw_rigid: If True, use piecewise-rigid registration
        max_shifts: Maximum allowed shifts
        strides: Stride for patches in pw-rigid mode
        overlaps: Overlap between patches
    """
    if fixed.ndim == 3:
        fixed = fixed[..., 0]
    if moving.ndim == 3:
        moving = moving[..., 0]
    
    # Create temporary file for the movie
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        fname = tmp.name
    
    try:
        # Save as tiff movie
        movie = np.stack([fixed, moving], axis=0).astype(np.float32)
        cm.movie(movie).save(fname)
        
        # Set up parameters
        opts_dict = {
            'max_shifts': max_shifts,
            'niter_rig': 1,
            'pw_rigid': pw_rigid,
            'strides': strides,
            'overlaps': overlaps,
            'max_deviation_rigid': 3,
            'border_nan': 'copy'
        }
        
        opts = params.CNMFParams(params_dict=opts_dict)
        
        # Run motion correction
        mc = MotionCorrect(fname, dview=None, **opts.get_group('motion'))
        mc.motion_correct(save_movie=False, template=fixed)
        
        # Extract shifts for the second frame
        if pw_rigid:
            # Get nonrigid shifts
            shifts = mc.x_shifts_els[1], mc.y_shifts_els[1]
            
            # The shifts are on a grid - need to expand to full resolution
            H, W = fixed.shape
            
            # Get the grid dimensions from mc object
            if hasattr(mc, 'total_template_els'):
                # Expand the shifts to full resolution
                y_shifts = shifts[1]  # y shifts
                x_shifts = shifts[0]  # x shifts
                
                # Reshape and resize if needed
                if isinstance(y_shifts, np.ndarray) and y_shifts.ndim > 0:
                    # Calculate grid dimensions
                    n_patches_y = len(np.arange(0, H - overlaps[0], strides[0])) + 1
                    n_patches_x = len(np.arange(0, W - overlaps[1], strides[1])) + 1
                    
                    if y_shifts.size == n_patches_y * n_patches_x:
                        y_grid = y_shifts.reshape(n_patches_y, n_patches_x)
                        x_grid = x_shifts.reshape(n_patches_y, n_patches_x)
                        
                        # Expand to full resolution
                        v = np.zeros((2, H, W), dtype=np.float32)
                        v[0] = cv2.resize(y_grid.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                        v[1] = cv2.resize(x_grid.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                    else:
                        # Fallback to uniform shift
                        v = np.zeros((2, H, W), dtype=np.float32)
                        v[0] += np.mean(y_shifts)
                        v[1] += np.mean(x_shifts)
                else:
                    # Single shift value
                    v = np.zeros((2, H, W), dtype=np.float32)
                    v[0] += float(y_shifts)
                    v[1] += float(x_shifts)
            else:
                # Fallback to rigid shifts
                v = np.zeros((2, H, W), dtype=np.float32)
                v[0] += mc.shifts_rig[1][0]  # y shift
                v[1] += mc.shifts_rig[1][1]  # x shift
        else:
            # Rigid registration - single shift for whole frame
            shifts = mc.shifts_rig[1]  # Get shifts for second frame
            H, W = fixed.shape
            v = np.zeros((2, H, W), dtype=np.float32)
            v[0] += shifts[0]  # y shift
            v[1] += shifts[1]  # x shift
            
    finally:
        # Clean up temporary file
        if os.path.exists(fname):
            os.remove(fname)
    
    return v