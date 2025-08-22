"""
Synthetic example using DIS-Fast backend for optical flow.
Performs the same tests as synth_example.py but with DISO initialization.
"""

import numpy as np
import h5py
import cv2
from os.path import join, dirname
import os
import sys
from time import time

# Add src directory to path for imports
src_path = join(dirname(dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from pyflowreg.core.diso import DISFast
from pyflowreg.core.optical_flow import imregister_wrapper


def main():
    # Load synthetic test data
    input_folder = join(dirname(dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    with h5py.File(join(input_folder, "synth_frames.h5"), "r") as f:
        # Use same data as synth_example.py
        clean = f["clean"][:]
        noisy35db = f["noisy35db"][:]
        clean = f["noisy30db"][:]  # Override with noisy30db as in original
        w_ground_truth = f["w"][:]
    
    # Prepare frames - same preprocessing as synth_example.py
    frame1 = np.permute_dims(clean[0], (1, 2, 0)).astype(float)
    frame2 = np.permute_dims(clean[1], (1, 2, 0)).astype(float)
    frame1 = cv2.GaussianBlur(frame1, None, 1.5)
    frame2 = cv2.GaussianBlur(frame2, None, 1.5)
    
    # Normalize frames (same as synth_example.py)
    eps = 1e-6
    mins = frame1.min(axis=(0, 1))[None, None, :]  # shape (1,1,C)
    maxs = frame1.max(axis=(0, 1))[None, None, :]  # shape (1,1,C)
    
    ranges = maxs - mins
    ranges[ranges < eps] = 1.0
    
    frame1 = (frame1 - mins) / ranges
    frame2 = (frame2 - mins) / ranges
    
    # Ground truth flow
    w_gt = np.permute_dims(w_ground_truth, (1, 2, 0))
    
    print(f"Frame shape: {frame1.shape}")
    print(f"Frame1 range: [{frame1.min():.3f}, {frame1.max():.3f}]")
    print(f"Frame2 range: [{frame2.min():.3f}, {frame2.max():.3f}]")
    
    # Initialize DIS-Fast solver
    dis_fast = DISFast(eta=0.75, use_ic=True, input_color="BGR")
    
    # Test 1: Fast settings (equivalent to alpha=(2,2) in variational)
    print("\n=== Test 1: Fast DIS-Fast settings ===")
    start = time()
    
    # For multi-channel input, we need to combine channels
    # DIS-Fast expects grayscale, so we'll use weighted average
    weights = np.array([0.6, 0.4])  # Same as synth_example.py
    if frame1.shape[2] == 2:
        frame1_gray = frame1[:, :, 0] * weights[0] + frame1[:, :, 1] * weights[1]
        frame2_gray = frame2[:, :, 0] * weights[0] + frame2[:, :, 1] * weights[1]
    else:
        frame1_gray = cv2.cvtColor(frame1.astype(np.float32), cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2.astype(np.float32), cv2.COLOR_BGR2GRAY)
    
    w_fast = dis_fast.multilevel(
        frame1_gray, frame2_gray,
        patch_radius=5,      # Small patch for speed
        stride=4,            # Standard stride
        iters=3,             # Few iterations (IC+propagation is efficient)
        lam=1e-4,           # Low damping
        eps=1e-3,           # Standard epsilon
        sigma=2.0,          # Gaussian weight sigma
        min_level=20,       # Don't go too coarse
        data_term="ssd",    # Standard SSD
        use_edge_aware=False,
        med_size=0          # No median filter for speed
    )
    
    elapsed = time() - start
    print(f"Elapsed time: {elapsed:.3f}s")
    
    # Test 2: High quality settings (equivalent to alpha=(8,8) in variational)
    print("\n=== Test 2: High quality DIS-Fast settings ===")
    times = []
    
    for i in range(2):  # Run twice as in original
        start = time()
        
        w_quality = dis_fast.multilevel(
            frame1_gray, frame2_gray,
            patch_radius=7,      # Larger patch for quality
            stride=3,            # Smaller stride for denser sampling
            iters=5,             # More iterations
            lam=1e-5,           # Lower damping for accuracy
            eps=1e-4,           # Smaller epsilon
            sigma=2.5,          # Larger sigma
            min_level=24,       # Allow finer pyramid
            data_term="ssd",    # Could use "ncc" for robustness
            use_edge_aware=True, # Enable edge-aware smoothing
            med_size=3          # Small median filter
        )
        
        elapsed = time() - start
        if i > 0:
            times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")
    
    if times:
        print(f"Average elapsed time: {sum(times) / len(times):.3f}s")
    
    # Use the quality result for visualization
    w = w_quality
    print(f"\nFlow shape: {w.shape}")
    
    # Compute flow statistics
    print(f"Flow X range: [{w[:,:,0].min():.2f}, {w[:,:,0].max():.2f}]")
    print(f"Flow Y range: [{w[:,:,1].min():.2f}, {w[:,:,1].max():.2f}]")
    
    # Compare with ground truth if available
    if w_gt.shape == w.shape:
        error = np.sqrt((w - w_gt)[:,:,0]**2 + (w - w_gt)[:,:,1]**2)
        print(f"Mean EPE vs ground truth: {error.mean():.3f} pixels")
        print(f"Max EPE vs ground truth: {error.max():.3f} pixels")
    
    # Visualize flow components
    img1 = w[..., 0]
    img2 = w[..., 1]
    img1_vis = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2_vis = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    cv2.imshow("DIS-Fast Flow X", img1_vis)
    cv2.imshow("DIS-Fast Flow Y", img2_vis)
    
    # Warp frame2 using the computed flow
    print("\nWarping frame2 using DIS-Fast flow...")
    
    # For warping multi-channel data, warp each channel
    if len(frame2.shape) == 3 and frame2.shape[2] > 1:
        warped_frame2 = np.zeros_like(frame2)
        for c in range(frame2.shape[2]):
            warped_frame2[:,:,c] = imregister_wrapper(
                frame2[:,:,c], w[..., 0], w[..., 1], 
                frame1[:,:,c], interpolation_method='cubic'
            )
    else:
        warped_frame2 = imregister_wrapper(
            frame2_gray, w[..., 0], w[..., 1], 
            frame1_gray, interpolation_method='cubic'
        )
    
    # Display results (using channel 1 if multi-channel, as in original)
    if len(frame2.shape) == 3 and frame2.shape[2] > 1:
        warped_display = cv2.normalize(warped_frame2[..., 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame1_display = cv2.normalize(frame1[..., 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame2_display = cv2.normalize(frame2[..., 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        warped_display = cv2.normalize(warped_frame2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame1_display = cv2.normalize(frame1_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame2_display = cv2.normalize(frame2_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    cv2.imshow("Frame1 (Reference) - DISO", frame1_display)
    cv2.imshow("Frame2 (Original) - DISO", frame2_display)
    cv2.imshow("Warped Frame2 - DISO", warped_display)
    
    # Compute and display difference image
    diff = np.abs(frame1_display.astype(float) - warped_display.astype(float))
    diff_display = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("Registration Error - DISO", diff_display)
    
    print("\nRegistration quality metrics:")
    mse = np.mean((frame1_display.astype(float) - warped_display.astype(float))**2)
    print(f"MSE: {mse:.2f}")
    print(f"PSNR: {10 * np.log10(255**2 / mse):.2f} dB")
    
    # Test 3: NCC mode for brightness robustness
    print("\n=== Test 3: DIS-Fast with NCC (brightness robust) ===")
    start = time()
    
    w_ncc = dis_fast.multilevel(
        frame1_gray, frame2_gray,
        patch_radius=5,
        stride=4,
        iters=4,
        lam=1e-4,
        eps=1e-3,
        sigma=2.0,
        min_level=20,
        data_term="ncc",    # Use NCC for brightness robustness
        use_edge_aware=False,
        med_size=0
    )
    
    elapsed = time() - start
    print(f"NCC mode time: {elapsed:.3f}s")
    
    if w_gt.shape == w_ncc.shape:
        error_ncc = np.sqrt((w_ncc - w_gt)[:,:,0]**2 + (w_ncc - w_gt)[:,:,1]**2)
        print(f"NCC Mean EPE: {error_ncc.mean():.3f} pixels")
    
    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()