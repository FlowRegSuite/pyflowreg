"""
Fast Direct Inverse Search (DIS-Fast) implementation.
Core optical flow initialization using inverse compositional updates with propagation.
"""

import numpy as np
from numba import njit, prange
import cv2


# ============================================================================
# Core Numba-JIT Kernels
# ============================================================================

@njit(fastmath=True, cache=True)
def _gauss_kernel(radius, sigma):
    """Create Gaussian kernel for patch weighting."""
    k = np.empty((2 * radius + 1, 2 * radius + 1), np.float32)
    s2 = 2.0 * sigma * sigma
    idx = 0
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            k.flat[idx] = np.exp(-(x * x + y * y) / s2)
            idx += 1
    s = 1.0 / k.sum()
    for i in range(k.size):
        k.flat[i] *= s
    return k


@njit(fastmath=True, cache=True)
def _grad_xy(img):
    """Compute image gradients."""
    H, W, C = img.shape
    gx = np.zeros((H, W, C), np.float32)
    gy = np.zeros((H, W, C), np.float32)
    for c in range(C):
        for y in range(H):
            for x in range(1, W - 1):
                gx[y, x, c] = 0.5 * (img[y, x + 1, c] - img[y, x - 1, c])
        for y in range(1, H - 1):
            for x in range(W):
                gy[y, x, c] = 0.5 * (img[y + 1, x, c] - img[y - 1, x, c])
        for y in range(H):
            gx[y, 0, c] = img[y, 1, c] - img[y, 0, c]
            gx[y, W - 1, c] = img[y, W - 1, c] - img[y, W - 2, c]
        for x in range(W):
            gy[0, x, c] = img[1, x, c] - img[0, x, c]
            gy[H - 1, x, c] = img[H - 1, x, c] - img[H - 2, x, c]
    return gx, gy


@njit(fastmath=True, cache=True)
def _bilinear_at(img, y, x, c):
    """Bilinear interpolation for 3D images."""
    H, W = img.shape[0], img.shape[1]
    if y < 0.0:
        y = 0.0
    if x < 0.0:
        x = 0.0
    if y > H - 1.0:
        y = H - 1.0
    if x > W - 1.0:
        x = W - 1.0
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = y0 + 1
    x1 = x0 + 1
    if y1 >= H:
        y1 = H - 1
    if x1 >= W:
        x1 = W - 1
    fy = y - y0
    fx = x - x0
    v00 = img[y0, x0, c]
    v01 = img[y0, x1, c]
    v10 = img[y1, x0, c]
    v11 = img[y1, x1, c]
    v0 = v00 + fx * (v01 - v00)
    v1 = v10 + fx * (v11 - v10)
    return v0 + fy * (v1 - v0)


@njit(fastmath=True, cache=True)
def _bilinear_at_2d(img, y, x):
    """Bilinear interpolation for 2D images."""
    H, W = img.shape
    if y < 0.0:
        y = 0.0
    if x < 0.0:
        x = 0.0
    if y > H - 1.0:
        y = H - 1.0
    if x > W - 1.0:
        x = W - 1.0
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = y0 + 1
    x1 = x0 + 1
    if y1 >= H:
        y1 = H - 1
    if x1 >= W:
        x1 = W - 1
    fy = y - y0
    fx = x - x0
    v00 = img[y0, x0]
    v01 = img[y0, x1]
    v10 = img[y1, x0]
    v11 = img[y1, x1]
    v0 = v00 + fx * (v01 - v00)
    v1 = v10 + fx * (v11 - v10)
    return v0 + fy * (v1 - v0)


# ============================================================================
# Patch Matching and Propagation
# ============================================================================

@njit(fastmath=True, cache=True)
def _patch_ssd(I1, I2, yc, xc, uxy, pr, wpatch):
    """Compute patch SSD at given displacement."""
    yy = uxy[1]
    xx = uxy[0]
    H, W = I1.shape
    s = 0.0
    idx = 0
    for dy in range(-pr, pr+1):
        y = yc + dy
        if y < 0 or y >= H: 
            idx += 2*pr+1
            continue
        for dx in range(-pr, pr+1):
            x = xc + dx
            w = wpatch.flat[idx]
            idx += 1
            if x < 0 or x >= W: 
                continue
            r = _bilinear_at_2d(I2, y+yy, x+xx) - I1[y, x]
            s += w * r * r
    return s


@njit(fastmath=True, cache=True)
def _propagate_once(I1, I2, centers, ny, nx, pr, wpatch, u, forward):
    """Scanline propagation with actual SSD evaluation."""
    n = centers.shape[0]
    start, stop, step = (0, n, 1) if forward else (n-1, -1, -1)

    for k in range(start, stop, step):
        yc, xc = centers[k, 0], centers[k, 1]
        best = u[k]
        best_e = _patch_ssd(I1, I2, yc, xc, best, pr, wpatch)

        iy = k // nx
        ix = k % nx

        if forward:
            cand_ids = []
            if iy > 0:       cand_ids.append((iy - 1) * nx + ix)     # top
            if ix > 0:       cand_ids.append(iy * nx + (ix - 1))     # left
        else:
            cand_ids = []
            if iy < ny - 1:  cand_ids.append((iy + 1) * nx + ix)     # bottom
            if ix < nx - 1:  cand_ids.append(iy * nx + (ix + 1))     # right

        for nk in cand_ids:
            cand = u[nk]
            e = _patch_ssd(I1, I2, yc, xc, cand, pr, wpatch)
            if e < best_e:
                best = cand
                best_e = e

        u[k, 0] = best[0]
        u[k, 1] = best[1]
    return u


@njit(fastmath=True, cache=True)
def _rand_seed(I1, I2, centers, pr, wpatch, radius, tries):
    """Random search with actual SSD evaluation."""
    u = np.zeros((centers.shape[0], 2), np.float32)
    for k in range(centers.shape[0]):
        yc, xc = centers[k, 0], centers[k, 1]
        best = u[k]
        best_e = _patch_ssd(I1, I2, yc, xc, best, pr, wpatch)
        rad = radius
        for _ in range(tries):
            rx = (np.random.rand() - 0.5) * 2.0 * rad
            ry = (np.random.rand() - 0.5) * 2.0 * rad
            cand = np.array((rx, ry), np.float32)
            e = _patch_ssd(I1, I2, yc, xc, cand, pr, wpatch)
            if e < best_e:
                best = cand
                best_e = e
            rad *= 0.5
        u[k] = best
    return u


# ============================================================================
# Inverse Compositional Updates
# ============================================================================

@njit(fastmath=True, cache=True, parallel=True)
def _precompute_ic(I1, centers, pr, wpatch):
    """Precompute Hessian and gradient for inverse compositional."""
    H, W = I1.shape[:2]
    num_patches = centers.shape[0]
    
    # Preallocate arrays
    H00 = np.zeros(num_patches, np.float32)
    H01 = np.zeros(num_patches, np.float32)
    H11 = np.zeros(num_patches, np.float32)
    
    # Compute gradients (always work with 3D internally)
    I1_3d = np.expand_dims(I1, axis=2)
    gx, gy = _grad_xy(I1_3d)
    
    # For each patch, compute Hessian
    for k in prange(num_patches):
        yc = centers[k, 0]
        xc = centers[k, 1]
        
        h00 = 0.0
        h01 = 0.0
        h11 = 0.0
        idx = 0
        
        for dy in range(-pr, pr + 1):
            y = yc + dy
            for dx in range(-pr, pr + 1):
                x = xc + dx
                w = wpatch.flat[idx]
                idx += 1
                
                if y < 0 or y >= H or x < 0 or x >= W:
                    continue
                
                # Use gradient at template position
                jx = gx[y, x, 0]
                jy = gy[y, x, 0]
                
                h00 += w * jx * jx
                h01 += w * jx * jy
                h11 += w * jy * jy
        
        H00[k] = h00
        H01[k] = h01
        H11[k] = h11
    
    return H00, H01, H11, gx, gy


@njit(fastmath=True, cache=True, parallel=True)
def _ic_step(I1, I2, gx, gy, centers, pr, wpatch, H00, H01, H11, u, lam, eps):
    """Inverse compositional update step."""
    num_patches = centers.shape[0]
    
    # Ensure I2 is 3D for _bilinear_at
    if I2.ndim == 2:
        I2_3d = np.expand_dims(I2, axis=2)
    else:
        I2_3d = I2
    
    for k in prange(num_patches):
        yc = centers[k, 0]
        xc = centers[k, 1]
        
        # Get precomputed Hessian
        h00 = H00[k] + lam
        h01 = H01[k]
        h11 = H11[k] + lam
        
        # Compute residual and Jacobian^T * residual
        b0 = 0.0
        b1 = 0.0
        idx = 0
        
        for dy in range(-pr, pr + 1):
            y = yc + dy
            for dx in range(-pr, pr + 1):
                x = xc + dx
                w = wpatch.flat[idx]
                idx += 1
                
                if y < 0 or y >= I1.shape[0] or x < 0 or x >= I1.shape[1]:
                    continue
                
                # Sample I2 at displaced position
                yy = y + u[k, 1]
                xx = x + u[k, 0]
                
                # Compute residual
                r = _bilinear_at(I2_3d, yy, xx, 0) - I1[y, x]
                
                # Use template gradients (IC)
                jx = gx[y, x, 0]
                jy = gy[y, x, 0]
                
                # Accumulate J^T * r
                b0 += w * jx * r
                b1 += w * jy * r
        
        # Solve 2x2 system
        det = h00 * h11 - h01 * h01
        if det > 1e-12:
            dx = (h11 * b0 - h01 * b1) / det
            dy = (-h01 * b0 + h00 * b1) / det
            u[k, 0] += dx
            u[k, 1] += dy
    
    return u


@njit(fastmath=True, cache=True, parallel=True)
def _dis_inverse_search_ic_impl(I1, I2, patch_radius, stride, iters, eps, lam, sigma):
    """Fast DIS with inverse compositional and propagation (njit implementation)."""
    H, W = I1.shape[:2]
    
    # Gaussian weights
    wpatch = _gauss_kernel(patch_radius, sigma)
    
    # Patch grid
    ny = (H - 2 * patch_radius + stride - 1) // stride
    nx = (W - 2 * patch_radius + stride - 1) // stride
    
    # Build patch centers (row-major layout)
    centers = np.empty((ny * nx, 2), np.int32)
    k = 0
    for iy in range(ny):
        yc = patch_radius + iy * stride
        for ix in range(nx):
            xc = patch_radius + ix * stride
            centers[k, 0] = yc
            centers[k, 1] = xc
            k += 1
    
    # Precompute IC Hessian and gradients
    H00, H01, H11, gx, gy = _precompute_ic(I1, centers, patch_radius, wpatch)
    
    # Initialize with random search
    u = _rand_seed(I1, I2, centers, patch_radius, wpatch, radius=3.0, tries=3)
    
    # Initial propagation passes
    u = _propagate_once(I1, I2, centers, ny, nx, patch_radius, wpatch, u, True)
    u = _propagate_once(I1, I2, centers, ny, nx, patch_radius, wpatch, u, False)
    
    # IC iterations with propagation
    for _ in range(iters):
        u = _ic_step(I1, I2, gx, gy, centers, 
                     patch_radius, wpatch, H00, H01, H11, u, lam, eps)
        u = _propagate_once(I1, I2, centers, ny, nx, patch_radius, wpatch, u, True)
        u = _propagate_once(I1, I2, centers, ny, nx, patch_radius, wpatch, u, False)
    
    # Reshape to grid
    disp = u.reshape(ny, nx, 2)
    return disp


def dis_inverse_search_ic(I1, I2, patch_radius, stride, iters, eps, lam, sigma):
    """Fast DIS with inverse compositional and propagation."""
    # Enforce grayscale float32 input
    if I1.ndim != 2 or I2.ndim != 2:
        raise ValueError("dis_inverse_search_ic expects grayscale float32 images")
    
    # Ensure float32 and contiguous
    I1 = np.ascontiguousarray(I1, dtype=np.float32)
    I2 = np.ascontiguousarray(I2, dtype=np.float32)
    
    # Call the actual njit implementation
    return _dis_inverse_search_ic_impl(I1, I2, patch_radius, stride, iters, eps, lam, sigma)


# ============================================================================
# Local Contrast Normalization for NCC
# ============================================================================

def _lcn_gray32(img, win=9, eps=1e-6):
    """Local contrast normalization using OpenCV."""
    mean = cv2.boxFilter(img, ddepth=-1, ksize=(win, win), 
                        normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_sq = cv2.boxFilter(img*img, ddepth=-1, ksize=(win, win), 
                           normalize=True, borderType=cv2.BORDER_REFLECT101)
    var = np.maximum(mean_sq - mean*mean, 0)
    std = np.sqrt(var + eps)
    return (img - mean) / std


# ============================================================================
# Pyramid and Flow Utilities
# ============================================================================

def _build_pyramid(img, eta=0.75, min_side=20, max_levels=10):
    """Build image pyramid with consistent float32 dtype."""
    pyr = [img.astype(np.float32)]
    for _ in range(max_levels - 1):
        h, w = pyr[-1].shape[:2]
        if min(h, w) * eta < min_side:
            break
        nh = max(1, int(round(h * eta)))
        nw = max(1, int(round(w * eta)))
        # Use INTER_AREA for downsampling (anti-aliasing)
        resized = cv2.resize(pyr[-1], (nw, nh), interpolation=cv2.INTER_AREA)
        if img.ndim == 3 and resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        pyr.append(resized)
    return pyr


def _rescale_flow(flow, new_hw):
    """Rescale flow to new resolution with proper scaling."""
    h, w = flow.shape[:2]
    nh, nw = new_hw
    sx = nw / float(w)
    sy = nh / float(h)
    # Resize flow field
    f = cv2.resize(flow, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # Scale flow vectors
    f[:, :, 0] *= sx  # Scale x-displacement
    f[:, :, 1] *= sy  # Scale y-displacement
    return f.astype(np.float32)


def _warp(I, u, v):
    """Warp image with flow field using bilinear interpolation."""
    h, w = I.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x + u
    map_y = grid_y + v
    
    if I.ndim == 3:
        warped = np.empty_like(I, dtype=np.float32)
        for c in range(I.shape[2]):
            warped[:, :, c] = cv2.remap(I[:, :, c], map_x, map_y,
                                        cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT101)
    else:
        warped = cv2.remap(I, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)
    return warped


# ============================================================================
# Fast Densification
# ============================================================================

def densify_fast(disp, H, W, patch_radius, stride, sigma):
    """Fast densification using separable Gaussian filtering."""
    ny, nx = disp.shape[:2]
    
    # Create sparse accumulation arrays
    acc = np.zeros((H, W, 2), np.float32)
    weight = np.zeros((H, W), np.float32)
    
    # Place sparse values
    for iy in range(ny):
        y = patch_radius + iy * stride
        if y < H:
            for ix in range(nx):
                x = patch_radius + ix * stride
                if x < W:
                    acc[y, x, 0] = disp[iy, ix, 0]
                    acc[y, x, 1] = disp[iy, ix, 1]
                    weight[y, x] = 1.0
    
    # Gaussian kernel size
    ksize = int(2 * round(3 * sigma) + 1)
    if ksize < 3:
        ksize = 3
    
    # Apply separable Gaussian blur
    acc[:, :, 0] = cv2.GaussianBlur(acc[:, :, 0], (ksize, ksize), sigma,
                                     borderType=cv2.BORDER_REFLECT101)
    acc[:, :, 1] = cv2.GaussianBlur(acc[:, :, 1], (ksize, ksize), sigma,
                                     borderType=cv2.BORDER_REFLECT101)
    weight = cv2.GaussianBlur(weight, (ksize, ksize), sigma,
                              borderType=cv2.BORDER_REFLECT101)
    
    # Normalize with inverse weight
    invw = 1.0 / np.maximum(weight, 1e-6)
    acc[:, :, 0] *= invw
    acc[:, :, 1] *= invw
    # Zero where weight is tiny
    mask = weight <= 1e-6
    acc[mask, 0] = 0
    acc[mask, 1] = 0
    
    return acc


# ============================================================================
# Optional: Edge-Aware Smoothing
# ============================================================================

def edge_aware_smooth(flow, guide, r=8, eps=1e-3):
    """Edge-aware smoothing using guided filter or joint bilateral filter."""
    g = guide.astype(np.float32)
    if guide.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    
    # Try guided filter if available
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
        f0 = cv2.ximgproc.guidedFilter(g, flow[:, :, 0].astype(np.float32), r, eps)
        f1 = cv2.ximgproc.guidedFilter(g, flow[:, :, 1].astype(np.float32), r, eps)
    # Try joint bilateral filter if available
    elif hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'jointBilateralFilter'):
        g8 = (np.clip(g, 0, 1) * 255).astype(np.uint8) if g.max() <= 1.0 else g.astype(np.uint8)
        f0 = cv2.ximgproc.jointBilateralFilter(g8, flow[:, :, 0].astype(np.float32), 9, 25, 7)
        f1 = cv2.ximgproc.jointBilateralFilter(g8, flow[:, :, 1].astype(np.float32), 9, 25, 7)
    else:
        # Plain bilateral as last resort
        f0 = cv2.bilateralFilter(flow[:, :, 0].astype(np.float32), 9, 25, 7)
        f1 = cv2.bilateralFilter(flow[:, :, 1].astype(np.float32), 9, 25, 7)
    return np.stack([f0, f1], -1)


# ============================================================================
# Main DIS-Fast Class
# ============================================================================

class DISFast:
    """Fast Direct Inverse Search with IC updates and propagation."""
    
    def __init__(self, eta=0.75, use_ic=True, input_color="BGR"):
        """
        Initialize fast DIS solver.
        
        Args:
            eta: Pyramid downsampling factor
            use_ic: Use inverse compositional (True) 
            input_color: Color order for RGB images ("BGR" for OpenCV, "RGB" for PIL/matplotlib)
        """
        self.eta = eta
        self.use_ic = use_ic
        self.input_color = input_color
    
    def multilevel(self, I1, I2, patch_radius=5, stride=4, iters=3, 
                   lam=1e-4, eps=1e-3, sigma=2.0, min_level=24,
                   data_term="ssd", use_edge_aware=False, med_size=0):
        """
        Multi-level DIS with fast propagation.
        
        Args:
            I1, I2: Input images (grayscale or RGB)
            patch_radius: Patch radius in pixels
            stride: Stride between patches
            iters: Iterations per level (can be low with IC+propagation)
            lam: Levenberg-Marquardt damping
            eps: Robust function epsilon
            sigma: Gaussian weight sigma
            min_level: Minimum image dimension at coarsest level
            data_term: "ssd" (standard) or "ncc" (local contrast normalized)
            use_edge_aware: Apply edge-aware smoothing
            med_size: Median filter size (0 to disable)
        
        Returns:
            Dense flow field (H, W, 2)
        """
        # Convert to grayscale with dtype-aware scaling
        def _to_gray32(x):
            if x.ndim == 3:
                # Choose color conversion based on input_color
                if self.input_color == "BGR":
                    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:  # RGB
                    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
            else:
                g = x.astype(np.float32)
            return g / 255.0 if x.dtype == np.uint8 else g
        
        I1_gray = _to_gray32(I1)
        I2_gray = _to_gray32(I2)
        
        # Build pyramids
        I1_pyr = _build_pyramid(I1_gray, self.eta, min_level, max_levels=10)
        I2_pyr = _build_pyramid(I2_gray, self.eta, min_level, max_levels=10)
        
        # Initialize flow at coarsest level
        u = np.zeros(I1_pyr[-1].shape[:2] + (2,), np.float32)
        
        # Process each pyramid level (coarse to fine)
        for lvl in range(len(I1_pyr) - 1, -1, -1):
            I1_lvl = I1_pyr[lvl]
            
            # Warp I2 with current flow estimate (except at coarsest)
            if lvl < len(I1_pyr) - 1:
                I2_lvl = _warp(I2_pyr[lvl], u[:, :, 0], u[:, :, 1])
            else:
                I2_lvl = I2_pyr[lvl]
            
            # Apply local contrast normalization if NCC mode
            if data_term == "ncc":
                I1_proc = _lcn_gray32(I1_lvl, win=9, eps=1e-6)
                I2_proc = _lcn_gray32(I2_lvl, win=9, eps=1e-6)
            else:
                I1_proc = I1_lvl
                I2_proc = I2_lvl
            
            # Adaptive parameters
            level_stride = max(2, int(stride * (self.eta ** (lvl * 0.5))))
            level_iters = max(3, iters - lvl)  # More iters at finer levels
            
            # Run DIS at this level
            disp = dis_inverse_search_ic(I1_proc, I2_proc, patch_radius, level_stride, 
                                        level_iters, eps, lam, sigma)
            
            # Densify
            du = densify_fast(disp, I1_lvl.shape[0], I1_lvl.shape[1],
                             patch_radius, level_stride, sigma)
            
            # Accumulate flow
            u = u + du
            
            # Apply median filter if requested
            if med_size >= 3 and lvl > 0:
                u[:, :, 0] = cv2.medianBlur(u[:, :, 0], med_size)
                u[:, :, 1] = cv2.medianBlur(u[:, :, 1], med_size)
            
            # Upscale for next level
            if lvl > 0:
                u = _rescale_flow(u, I1_pyr[lvl - 1].shape[:2])
        
        # Optional: Edge-aware smoothing (using original image as guide)
        if use_edge_aware:
            u = edge_aware_smooth(u, I1_gray, r=8, eps=1e-3)
        
        return u


# ============================================================================
# Numba Warm-up for <200ms Budget
# ============================================================================

def _warmup_numba():
    """Warm up Numba JIT compilation."""
    try:
        # Small test to trigger JIT compilation
        I1 = np.zeros((16, 16), np.float32)
        I2 = np.zeros((16, 16), np.float32)
        _ = dis_inverse_search_ic(I1, I2, patch_radius=3, stride=4, 
                                  iters=2, eps=1e-3, lam=1e-4, sigma=1.0)
    except:
        pass  # Silently ignore warm-up errors

# Trigger warm-up at import
_warmup_numba()