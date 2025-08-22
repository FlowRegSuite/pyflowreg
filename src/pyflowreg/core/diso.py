import numpy as np
from numba import njit, prange
import cv2
from scipy.ndimage import median_filter

# Import helper functions from optical_flow
from pyflowreg.core.optical_flow import (
    warpingDepth, 
    add_boundary, 
    imregister_wrapper,
    resize
)


# ============================================================================
# Fast DIS Implementation with Inverse Compositional Updates
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
def _accumulate_system(I1, I2, gx2, gy2, yc, xc, u0, radius, wpatch, eps):
    H00 = 0.0
    H01 = 0.0
    H11 = 0.0
    b0 = 0.0
    b1 = 0.0
    H, W, C = I1.shape
    idx = 0
    uy = u0[1]
    ux = u0[0]
    for dy in range(-radius, radius + 1):
        y = yc + dy
        for dx in range(-radius, radius + 1):
            x = xc + dx
            w = wpatch.flat[idx]
            idx += 1
            if y < 0 or y >= H or x < 0 or x >= W:
                continue
            yy = y + uy
            xx = x + ux
            rr = 0.0
            gx = 0.0
            gy = 0.0
            for c in range(C):
                r = _bilinear_at(I2, yy, xx, c) - I1[y, x, c]
                rr += r * r
            ww = 1.0 / np.sqrt(rr + eps * eps)
            for c in range(C):
                r = _bilinear_at(I2, yy, xx, c) - I1[y, x, c]
                jx = _bilinear_at(gx2, yy, xx, c)
                jy = _bilinear_at(gy2, yy, xx, c)
                jx *= ww
                jy *= ww
                r *= ww
                H00 += w * jx * jx
                H01 += w * jx * jy
                H11 += w * jy * jy
                b0 += w * jx * r
                b1 += w * jy * r
    return H00, H01, H11, b0, b1


@njit(fastmath=True, cache=True)
def accumulate_system_ncc(I1, I2, gx2, gy2, yc, xc, u0, radius, wpatch, eps):
    H00 = 0.0
    H01 = 0.0
    H11 = 0.0
    b0 = 0.0
    b1 = 0.0
    H, W, C = I1.shape
    uy = u0[1]
    ux = u0[0]
    mean1 = np.zeros(C, np.float32)
    mean2 = np.zeros(C, np.float32)
    invs1 = np.ones(C, np.float32)
    invs2 = np.ones(C, np.float32)
    for c in range(C):
        s1 = 0.0
        s2 = 0.0
        q1 = 0.0
        q2 = 0.0
        for dy in range(-radius, radius + 1):
            y = yc + dy
            if y < 0 or y >= H:
                continue
            for dx in range(-radius, radius + 1):
                x = xc + dx
                if x < 0 or x >= W:
                    continue
                v1 = I1[y, x, c]
                v2 = _bilinear_at(I2, y + uy, x + ux, c)
                s1 += v1
                s2 += v2
                q1 += v1 * v1
                q2 += v2 * v2
        n = float((2 * radius + 1) * (2 * radius + 1))
        m1 = s1 / max(1.0, n)
        m2 = s2 / max(1.0, n)
        v1 = q1 / max(1.0, n) - m1 * m1
        v2 = q2 / max(1.0, n) - m2 * m2
        mean1[c] = m1
        mean2[c] = m2
        invs1[c] = 1.0 / np.sqrt(max(v1, 1e-12))
        invs2[c] = 1.0 / np.sqrt(max(v2, 1e-12))
    idx = 0
    for dy in range(-radius, radius + 1):
        y = yc + dy
        for dx in range(-radius, radius + 1):
            x = xc + dx
            w = wpatch.flat[idx]
            idx += 1
            if y < 0 or y >= H or x < 0 or x >= W:
                continue
            yy = y + uy
            xx = x + ux
            rr = 0.0
            for c in range(C):
                r = (_bilinear_at(I2, yy, xx, c) - mean2[c]) * invs2[c] - (I1[y, x, c] - mean1[c]) * invs1[c]
                rr += r * r
            ww = 1.0 / np.sqrt(rr + eps * eps)
            for c in range(C):
                r = (_bilinear_at(I2, yy, xx, c) - mean2[c]) * invs2[c] - (I1[y, x, c] - mean1[c]) * invs1[c]
                jx = _bilinear_at(gx2, yy, xx, c) * invs2[c]
                jy = _bilinear_at(gy2, yy, xx, c) * invs2[c]
                r *= ww
                jx *= ww
                jy *= ww
                H00 += w * jx * jx
                H01 += w * jx * jy
                H11 += w * jy * jy
                b0 += w * jx * r
                b1 += w * jy * r
    return H00, H01, H11, b0, b1


@njit(fastmath=True, cache=True)
def _solve2x2(H00, H01, H11, b0, b1, lam):
    H00 += lam
    H11 += lam
    det = H00 * H11 - H01 * H01
    if det <= 1e-12:
        return 0.0, 0.0
    inv00 = H11 / det
    inv01 = -H01 / det
    inv11 = H00 / det
    dx = inv00 * b0 + inv01 * b1
    dy = inv01 * b0 + inv11 * b1
    return dx, dy


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
            # For 2D images (grayscale)
            r = _bilinear_at_2d(I2, y+yy, x+xx) - I1[y, x]
            s += w * r * r
    return s


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


@njit(fastmath=True, cache=True)
def _propagate_once(I1, I2, centers, pr, wpatch, u, forward):
    """Scanline propagation with actual SSD evaluation."""
    n = centers.shape[0]
    if forward:
        start, stop, step = 0, n, 1
    else:
        start, stop, step = n-1, -1, -1

    H, W = I1.shape
    nx = int((W - 2*pr + 1) // 4)  # Stride is typically 4
    
    for k in range(start, stop, step):
        yc, xc = centers[k, 0], centers[k, 1]
        best = u[k]
        best_e = _patch_ssd(I1, I2, yc, xc, best, pr, wpatch)
        
        # Get neighbor indices based on row-major layout
        iy = k // nx
        ix = k % nx
        
        if forward:
            # Check top and left neighbors
            neighbors = []
            if iy > 0:  # Top neighbor
                neighbors.append((iy - 1) * nx + ix)
            if ix > 0:  # Left neighbor
                neighbors.append(iy * nx + (ix - 1))
        else:
            # Check bottom and right neighbors
            neighbors = []
            if iy < (n // nx) - 1:  # Bottom neighbor
                neighbors.append((iy + 1) * nx + ix)
            if ix < nx - 1:  # Right neighbor
                neighbors.append(iy * nx + (ix + 1))
        
        for nk in neighbors:
            if nk >= 0 and nk < n:
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
                
                # Use gradient at template position (always 3D)
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
                
                # Use template gradients (IC) - always 3D
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


def dis_inverse_search_ic(I1, I2, patch_radius, stride, iters, eps, lam, sigma):
    """Fast DIS with inverse compositional and propagation."""
    # Convert to grayscale if needed (outside of njit for simplicity)
    if I1.ndim == 3:
        I1 = I1[:, :, 0].copy()
    if I2.ndim == 3:
        I2 = I2[:, :, 0].copy()
    
    # Call the actual njit implementation
    return _dis_inverse_search_ic_impl(I1, I2, patch_radius, stride, iters, eps, lam, sigma)


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
    u = _propagate_once(I1, I2, centers, patch_radius, wpatch, u, True)
    u = _propagate_once(I1, I2, centers, patch_radius, wpatch, u, False)
    
    # IC iterations with propagation
    for _ in range(iters):
        u = _ic_step(I1, I2, gx, gy, centers, 
                     patch_radius, wpatch, H00, H01, H11, u, lam, eps)
        u = _propagate_once(I1, I2, centers, patch_radius, wpatch, u, True)
        u = _propagate_once(I1, I2, centers, patch_radius, wpatch, u, False)
    
    # Reshape to grid
    disp = u.reshape(ny, nx, 2)
    return disp


@njit(fastmath=True, cache=True, parallel=True)
def dis_inverse_search_multi(I1, I2, patch_radius, stride, iters, eps, lam, sigma):
    H, W, C = I1.shape
    gx2, gy2 = _grad_xy(I2)
    wpatch = _gauss_kernel(patch_radius, sigma)
    ny = (H - 2 * patch_radius + stride - 1) // stride
    nx = (W - 2 * patch_radius + stride - 1) // stride
    disp = np.zeros((ny, nx, 2), np.float32)
    for iy in range(ny):
        yc = patch_radius + iy * stride
        for ix in range(nx):
            xc = patch_radius + ix * stride
            u0x = 0.0
            u0y = 0.0
            for _ in range(iters):
                H00, H01, H11, b0, b1 = _accumulate_system(I1, I2, gx2, gy2, yc, xc, (u0x, u0y), patch_radius, wpatch,
                                                           eps)
                dx, dy = _solve2x2(H00, H01, H11, b0, b1, lam)
                u0x += dx
                u0y += dy
                if dx * dx + dy * dy < 1e-8:
                    break
            disp[iy, ix, 0] = u0x
            disp[iy, ix, 1] = u0y
    return disp


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
    
    # Normalize
    mask = weight > 1e-6
    acc[mask, 0] /= weight[mask]
    acc[mask, 1] /= weight[mask]
    
    return acc


def fb_consistency_check(wf, wb, threshold=1.0):
    """Forward-backward consistency check."""
    # Warp backward flow to forward frame
    wb_warped = _warp(wb, wf[:, :, 0], wf[:, :, 1])
    
    # Check consistency
    err = np.sqrt((wf[:, :, 0] + wb_warped[:, :, 0])**2 + 
                  (wf[:, :, 1] + wb_warped[:, :, 1])**2)
    
    # Create mask
    return err < threshold


def edge_aware_smooth(flow, guide, r=8, eps=1e-3):
    """Edge-aware smoothing using guided filter or bilateral filter."""
    # Convert guide to grayscale if needed
    if guide.ndim == 3:
        g = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        g = guide.astype(np.float32)
    
    # Try to use guided filter if available
    try:
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
            f0 = cv2.ximgproc.guidedFilter(g, flow[:, :, 0].astype(np.float32), r, eps)
            f1 = cv2.ximgproc.guidedFilter(g, flow[:, :, 1].astype(np.float32), r, eps)
        else:
            # Fall back to bilateral filter with guide
            # Convert to uint8 for bilateral filter
            g_uint8 = (g * 255).astype(np.uint8) if g.max() <= 1.0 else g.astype(np.uint8)
            f0 = cv2.bilateralFilter(flow[:, :, 0].astype(np.float32), 9, 25, 7)
            f1 = cv2.bilateralFilter(flow[:, :, 1].astype(np.float32), 9, 25, 7)
    except:
        # Last resort: just bilateral filter
        f0 = cv2.bilateralFilter(flow[:, :, 0].astype(np.float32), 9, 25, 7)
        f1 = cv2.bilateralFilter(flow[:, :, 1].astype(np.float32), 9, 25, 7)
    
    return np.stack([f0, f1], -1)


# Variational refinement removed for speed - DIS-fast focuses on propagation


@njit(fastmath=True, cache=True)
def densify(disp, H, W, patch_radius, stride, sigma):
    wpatch = _gauss_kernel(patch_radius, sigma)
    flow = np.zeros((H, W, 2), np.float32)
    wsum = np.zeros((H, W), np.float32)
    ny, nx = disp.shape[0], disp.shape[1]
    for iy in range(ny):
        yc = patch_radius + iy * stride
        for ix in range(nx):
            xc = patch_radius + ix * stride
            u0x = disp[iy, ix, 0]
            u0y = disp[iy, ix, 1]
            idx = 0
            for dy in range(-patch_radius, patch_radius + 1):
                y = yc + dy
                if y < 0 or y >= H:
                    idx += 2 * patch_radius + 1
                    continue
                for dx in range(-patch_radius, patch_radius + 1):
                    x = xc + dx
                    w = wpatch.flat[idx]
                    idx += 1
                    if x < 0 or x >= W:
                        continue
                    wsum[y, x] += w
                    flow[y, x, 0] += w * u0x
                    flow[y, x, 1] += w * u0y
    for y in range(H):
        for x in range(W):
            s = wsum[y, x]
            if s > 0.0:
                s = 1.0 / s
                flow[y, x, 0] *= s
                flow[y, x, 1] *= s
    return flow


def build_diso_inverse_search(accum):
    @njit(fastmath=True, cache=True)
    def _dis(I1, I2, patch_radius, stride, iters, eps, lam, sigma):
        H, W, C = I1.shape
        gx2, gy2 = _grad_xy(I2)
        wpatch = _gauss_kernel(patch_radius, sigma)
        ny = (H - 2*patch_radius + stride - 1) // stride
        nx = (W - 2*patch_radius + stride - 1) // stride
        disp = np.zeros((ny, nx, 2), np.float32)
        for iy in range(ny):
            yc = patch_radius + iy*stride
            for ix in range(nx):
                xc = patch_radius + ix*stride
                u0x = 0.0
                u0y = 0.0
                for _ in range(iters):
                    H00, H01, H11, b0, b1 = accum(I1, I2, gx2, gy2, yc, xc, (u0x, u0y), patch_radius, wpatch, eps)
                    dx, dy = _solve2x2(H00, H01, H11, b0, b1, lam)
                    u0x += dx
                    u0y += dy
                    if dx*dx + dy*dy < 1e-8:
                        break
                disp[iy, ix, 0] = u0x
                disp[iy, ix, 1] = u0y
        return disp
    return _dis


diso_map = {
    "ncc": accumulate_system_ncc,
    "cv2": _accumulate_system
}


class DISFast:
    """Fast Direct Inverse Search with IC updates and propagation."""
    
    def __init__(self, mode="cv2", eta=0.75, use_ic=True):
        """
        Initialize fast DIS solver.
        
        Args:
            mode: "cv2" (standard) or "ncc" (normalized cross-correlation)
            eta: Pyramid downsampling factor
            use_ic: Use inverse compositional (True) or standard additive (False)
        """
        self.mode = mode
        self.eta = eta
        self.use_ic = use_ic
        
        if use_ic:
            self.inv = dis_inverse_search_ic
        else:
            self.inv = build_diso_inverse_search(diso_map[mode])
    
    def multilevel(self, I1, I2, patch_radius=7, stride=4, iters=5, 
                   lam=1e-4, eps=1e-3, sigma=2.0, min_level=20,
                   use_fb_check=False, use_edge_aware=True):
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
            use_fb_check: Apply forward-backward consistency check
            use_edge_aware: Apply edge-aware smoothing
        
        Returns:
            Dense flow field (H, W, 2)
        """
        # Convert to grayscale if needed (for speed)
        if I1.ndim == 3:
            I1_gray = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            I1_gray = I1.astype(np.float32)
        
        if I2.ndim == 3:
            I2_gray = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            I2_gray = I2.astype(np.float32)
        
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
            
            # Adaptive parameters
            level_stride = max(2, int(stride * (self.eta ** (lvl * 0.5))))
            level_iters = max(3, iters - lvl)  # More iters at finer levels
            
            # Run DIS at this level
            disp = self.inv(I1_lvl, I2_lvl, patch_radius, level_stride, 
                           level_iters, eps, lam, sigma)
            
            # Densify
            du = densify_fast(disp, I1_lvl.shape[0], I1_lvl.shape[1],
                             patch_radius, level_stride, sigma)
            
            # Accumulate flow
            u = u + du
            
            # Apply median filter (adaptive size) using scipy for float32 safety
            if lvl > 0:
                med_size = min(5, max(3, int(5 * (self.eta ** lvl))))
                if med_size >= 3:
                    from scipy.ndimage import median_filter
                    u[:, :, 0] = median_filter(u[:, :, 0], size=med_size, mode='mirror')
                    u[:, :, 1] = median_filter(u[:, :, 1], size=med_size, mode='mirror')
            
            # Upscale for next level
            if lvl > 0:
                u = _rescale_flow(u, I1_pyr[lvl - 1].shape[:2])
        
        # Optional: Forward-backward consistency check
        if use_fb_check:
            # Compute backward flow
            u_back = self.multilevel(I2, I1, patch_radius, stride, iters,
                                    lam, eps, sigma, min_level,
                                    use_fb_check=False, use_edge_aware=False)
            # Apply consistency mask
            mask = fb_consistency_check(u, u_back)
            # Could inpaint inconsistent regions here
        
        # Optional: Edge-aware smoothing (using original image as guide)
        if use_edge_aware:
            u = edge_aware_smooth(u, I1_gray, r=8, eps=1e-3)
        
        return u


class DISOInverseSearch:
    """Legacy DISO class for backward compatibility."""
    
    def __init__(self, mode="ncc"):
        """
        Initialize DISO (Direct Inverse Search Optimization) solver.
        
        Args:
            mode: Accumulation mode - "ncc" (normalized cross-correlation) or "cv2" (standard)
        """
        self.mode = mode
        self.inv = build_diso_inverse_search(diso_map[mode])
        self.dis_fast = DISFast(mode="cv2", use_ic=True)

    def __call__(self, I1, I2, patch_radius=3, stride=5, iters=4, eps=1e-3, lam=1e-4, sigma=1.5):
        """
        Single-level DISO solver (backward compatibility).
        
        Args:
            I1: Reference image
            I2: Target image
            patch_radius: Radius of patches for matching
            stride: Stride between patch centers
            iters: Number of iterations per patch
            eps: Regularization epsilon
            lam: Levenberg-Marquardt damping
            sigma: Gaussian weight sigma
            
        Returns:
            Dense flow field (H, W, 2)
        """
        # Ensure 3D arrays for legacy solver
        if I1.ndim == 2:
            I1 = I1[:, :, np.newaxis]
        if I2.ndim == 2:
            I2 = I2[:, :, np.newaxis]
        
        inv = self.inv
        disp = inv(I1, I2, patch_radius, stride, iters, eps, lam, sigma)
        H, W = I1.shape[0], I1.shape[1]
        flow = densify(disp, H, W, patch_radius, stride, sigma)
        return flow
    
    def get_displacement_fast(self, fixed, moving, **kwargs):
        """Fast multi-level solver using DISFast."""
        return self.dis_fast.multilevel(fixed, moving, **kwargs)
    
    def get_displacement_multilevel(self, fixed, moving, patch_radius=7, stride=4, 
                                   iters=10, eps=1e-3, lam=0.01, sigma=2.0,
                                   min_level=0, levels=50, eta=0.8, 
                                   uv=None, weight=None, use_median=True):
        """
        [DEPRECATED] Legacy multi-level DISO solver with pyramid approach.
        Use get_displacement_fast() or DISFast.multilevel() for better performance.
        
        Args:
            fixed: Reference image (H, W) or (H, W, C)
            moving: Target image (H, W) or (H, W, C)
            patch_radius: Radius of patches for matching
            stride: Stride between patch centers
            iters: Number of iterations per patch
            eps: Regularization epsilon for robust weighting
            lam: Levenberg-Marquardt damping parameter
            sigma: Gaussian weight sigma for patch weighting
            min_level: Minimum pyramid level (0 = full resolution)
            levels: Maximum number of pyramid levels to attempt
            eta: Pyramid downsampling factor (0 < eta < 1)
            uv: Initial displacement field (H, W, 2) or None
            weight: Channel weights (not used in DISO, kept for API compatibility)
            use_median: Whether to apply median filtering between levels
            
        Returns:
            Displacement field (H, W, 2) where [:,:,0] is x-displacement, [:,:,1] is y-displacement
        """
        # Convert to float64 for precision
        fixed = fixed.astype(np.float64)
        moving = moving.astype(np.float64)
        
        # Handle dimensions
        if fixed.ndim == 3:
            m, n, n_channels = fixed.shape
        else:
            m, n = fixed.shape
            n_channels = 1
            fixed = fixed[:, :, np.newaxis]
            moving = moving[:, :, np.newaxis]
        
        # Initialize displacement
        if uv is not None:
            u_init = uv[:, :, 0]
            v_init = uv[:, :, 1]
        else:
            u_init = np.zeros((m, n), dtype=np.float64)
            v_init = np.zeros((m, n), dtype=np.float64)
        
        # Keep original images for pyramid
        f1_low = fixed
        f2_low = moving
        
        # Compute pyramid depth
        max_level_y = warpingDepth(eta, levels, m, n)
        max_level_x = warpingDepth(eta, levels, m, n)
        max_level = min(max_level_x, max_level_y) * 4
        max_level_y = min(max_level_y, max_level)
        max_level_x = min(max_level_x, max_level)
        
        # Adjust min_level if necessary
        if max(max_level_x, max_level_y) <= min_level:
            min_level = max(max_level_x, max_level_y) - 1
        if min_level < 0:
            min_level = 0
        
        # Initialize flow
        u = None
        v = None
        
        # Pyramid iteration (coarse to fine)
        for i in range(max(max_level_x, max_level_y), min_level - 1, -1):
            # Compute level size
            level_size = (
                int(round(m * eta ** min(i, max_level_y))),
                int(round(n * eta ** min(i, max_level_x)))
            )
            
            # Resize images to current level
            f1_level = resize(f1_low, level_size)
            f2_level = resize(f2_low, level_size)
            
            # Ensure 3D arrays
            if f1_level.ndim == 2:
                f1_level = f1_level[:, :, np.newaxis]
                f2_level = f2_level[:, :, np.newaxis]
            
            # Compute scaling factors for displacement
            current_hx = float(m) / f1_level.shape[0]
            current_hy = float(n) / f1_level.shape[1]
            
            # Initialize or propagate flow
            if i == max(max_level_x, max_level_y):
                # Coarsest level - use initialization
                u = add_boundary(resize(u_init, level_size))
                v = add_boundary(resize(v_init, level_size))
            else:
                # Propagate from coarser level
                u = add_boundary(resize(u[1:-1, 1:-1], level_size))
                v = add_boundary(resize(v[1:-1, 1:-1], level_size))
                
                # Warp moving image with current flow estimate
                f2_level = imregister_wrapper(
                    f2_level, 
                    u[1:-1, 1:-1] / current_hy, 
                    v[1:-1, 1:-1] / current_hx, 
                    f1_level
                )
            
            # Ensure 3D
            if f2_level.ndim == 2:
                f2_level = f2_level[:, :, np.newaxis]
            
            # Adaptive parameters based on level
            if i == min_level:
                # Full resolution - use provided parameters
                level_stride = stride
                level_patch_radius = patch_radius
                level_iters = iters
                level_lam = lam
            else:
                # Coarser levels - adapt parameters
                scale_factor = eta ** (-0.5 * i)
                level_stride = max(2, int(stride * eta ** i))
                level_patch_radius = max(3, int(patch_radius * (0.7 + 0.3 * (1 - eta ** i))))
                level_iters = max(5, int(iters * 0.7))
                level_lam = lam * scale_factor
            
            # Run DISO at current level
            # Note: We compute incremental flow (between warped f2 and f1)
            # Check if we need to handle boundaries differently
            if i == max(max_level_x, max_level_y):
                # First iteration - no warping yet, use original images
                f1_for_diso = f1_level
                f2_for_diso = f2_level
            else:
                # Subsequent iterations - f2_level is already warped
                f1_for_diso = f1_level
                f2_for_diso = f2_level
            
            # Ensure we have valid dimensions
            if f1_for_diso.shape[0] < 2 * level_patch_radius + 1 or f1_for_diso.shape[1] < 2 * level_patch_radius + 1:
                # Image too small for this patch radius, skip DISO and just interpolate
                du = np.zeros_like(u)
                dv = np.zeros_like(v)
            else:
                # Run DISO
                disp_sparse = self.inv(
                    f1_for_diso,
                    f2_for_diso,
                    level_patch_radius,
                    level_stride,
                    level_iters,
                    eps,
                    level_lam,
                    sigma
                )
                
                # Densify sparse displacement
                du_level = densify(
                    disp_sparse,
                    f1_for_diso.shape[0],
                    f1_for_diso.shape[1],
                    level_patch_radius,
                    level_stride,
                    sigma
                )
                
                # Add boundaries to match u, v size
                du = np.zeros_like(u)
                dv = np.zeros_like(v)
                du[1:-1, 1:-1] = du_level[:, :, 0] * current_hx  # Scale by pixel spacing
                dv[1:-1, 1:-1] = du_level[:, :, 1] * current_hy
            
            # Apply median filter if enabled and image is large enough
            if use_median and min(level_size) > 5:
                du[1:-1, 1:-1] = median_filter(du[1:-1, 1:-1], size=(5, 5), mode='mirror')
                dv[1:-1, 1:-1] = median_filter(dv[1:-1, 1:-1], size=(5, 5), mode='mirror')
            
            # Update flow
            u = u + du
            v = v + dv
        
        # Prepare output
        w = np.zeros((u.shape[0] - 2, u.shape[1] - 2, 2), dtype=np.float64)
        w[:, :, 0] = u[1:-1, 1:-1]
        w[:, :, 1] = v[1:-1, 1:-1]
        
        # Resize to original resolution if needed
        if min_level > 0:
            w = cv2.resize(w, (n, m), interpolation=cv2.INTER_CUBIC)
        
        return w


if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path to enable imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    try:
        from pyflowreg.util.visualization import quiver_visualization, flow_to_color
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        print("Warning: Visualization module not available, using simple flow_to_color")
        VISUALIZATION_AVAILABLE = False
        
        def flow_to_color(flow):
            """Simple flow to color conversion for testing."""
            import numpy as np
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
            hsv[:, :, 0] = ang * 180 / np.pi / 2
            hsv[:, :, 1] = 255
            hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return rgb.astype(np.float32) / 255.0
        
        def quiver_visualization(img, flow, **kwargs):
            """Simple quiver visualization for testing."""
            return img  # Just return the image for now
    
    import matplotlib.pyplot as plt
    
    def create_gaussian_2d(height, width, center_y, center_x, sigma_y=10.0, sigma_x=10.0):
        """Create a 2D Gaussian centered at (center_y, center_x)"""
        y, x = np.ogrid[:height, :width]
        gaussian = np.exp(-((y - center_y)**2 / (2.0 * sigma_y**2) + 
                           (x - center_x)**2 / (2.0 * sigma_x**2)))
        return gaussian.astype(np.float32)
    
    def create_moving_gaussian_pair(height=256, width=256, displacement=(5.0, 10.0), 
                                   sigma=15.0, num_gaussians=3):
        """
        Create a pair of images with multiple moving Gaussians.
        
        Args:
            height, width: Image dimensions
            displacement: (dy, dx) displacement for all Gaussians
            sigma: Standard deviation of Gaussians
            num_gaussians: Number of Gaussians to place in the image
        
        Returns:
            I1: First image with Gaussians
            I2: Second image with displaced Gaussians
            true_flow: Ground truth displacement field
        """
        # Create empty images (3 channels for RGB)
        I1 = np.zeros((height, width, 3), dtype=np.float32)
        I2 = np.zeros((height, width, 3), dtype=np.float32)
        
        # Random seed for reproducibility
        np.random.seed(42)
        
        # Create multiple Gaussians with different colors
        for i in range(num_gaussians):
            # Random position for first image (with margin)
            margin = sigma * 3
            cy1 = np.random.uniform(margin, height - margin)
            cx1 = np.random.uniform(margin, width - margin)
            
            # Position in second image (with displacement)
            cy2 = cy1 + displacement[0]
            cx2 = cx1 + displacement[1]
            
            # Random color weights for each channel
            color = np.random.uniform(0.3, 1.0, 3)
            
            # Add Gaussian to each channel
            for c in range(3):
                I1[:, :, c] += color[c] * create_gaussian_2d(height, width, cy1, cx1, sigma, sigma)
                I2[:, :, c] += color[c] * create_gaussian_2d(height, width, cy2, cx2, sigma, sigma)
        
        # Add small amount of noise
        noise_level = 0.01
        I1 += np.random.randn(height, width, 3).astype(np.float32) * noise_level
        I2 += np.random.randn(height, width, 3).astype(np.float32) * noise_level
        
        # Normalize to [0, 1]
        I1 = np.clip(I1, 0, 1)
        I2 = np.clip(I2, 0, 1)
        
        # Create ground truth flow field (constant displacement)
        true_flow = np.zeros((height, width, 2), dtype=np.float32)
        true_flow[:, :, 0] = displacement[1]  # x-displacement
        true_flow[:, :, 1] = displacement[0]  # y-displacement
        
        return I1, I2, true_flow
    
    def create_rotation_test(height=256, width=256, angle_deg=5.0, sigma=20.0):
        """
        Create a test case with rotational motion around image center.
        """
        I1 = np.zeros((height, width, 3), dtype=np.float32)
        I2 = np.zeros((height, width, 3), dtype=np.float32)
        
        # Center of rotation
        cy, cx = height / 2, width / 2
        
        # Create a pattern of Gaussians in a circle
        num_gaussians = 8
        radius = min(height, width) / 3
        
        true_flow = np.zeros((height, width, 2), dtype=np.float32)
        
        for i in range(num_gaussians):
            angle1 = 2 * np.pi * i / num_gaussians
            angle2 = angle1 + np.radians(angle_deg)
            
            # Position in first image
            y1 = cy + radius * np.sin(angle1)
            x1 = cx + radius * np.cos(angle1)
            
            # Position in second image (after rotation)
            y2 = cy + radius * np.sin(angle2)
            x2 = cx + radius * np.cos(angle2)
            
            # Color based on position
            color = np.array([
                0.5 + 0.5 * np.cos(angle1),
                0.5 + 0.5 * np.sin(angle1),
                0.7
            ])
            
            for c in range(3):
                I1[:, :, c] += color[c] * create_gaussian_2d(height, width, y1, x1, sigma, sigma)
                I2[:, :, c] += color[c] * create_gaussian_2d(height, width, y2, x2, sigma, sigma)
        
        # Compute approximate flow field (for visualization comparison)
        # This is approximate since we only compute flow at Gaussian centers
        y_grid, x_grid = np.mgrid[:height, :width]
        for i in range(num_gaussians):
            angle1 = 2 * np.pi * i / num_gaussians
            angle2 = angle1 + np.radians(angle_deg)
            
            y1 = cy + radius * np.sin(angle1)
            x1 = cx + radius * np.cos(angle1)
            y2 = cy + radius * np.sin(angle2)
            x2 = cx + radius * np.cos(angle2)
            
            # Weight based on distance to Gaussian center
            dist_sq = (y_grid - y1)**2 + (x_grid - x1)**2
            weight = np.exp(-dist_sq / (2 * sigma**2))
            
            true_flow[:, :, 0] += weight * (x2 - x1)
            true_flow[:, :, 1] += weight * (y2 - y1)
        
        # Normalize flow weights
        weight_sum = np.zeros((height, width), dtype=np.float32)
        for i in range(num_gaussians):
            angle1 = 2 * np.pi * i / num_gaussians
            y1 = cy + radius * np.sin(angle1)
            x1 = cx + radius * np.cos(angle1)
            dist_sq = (y_grid - y1)**2 + (x_grid - x1)**2
            weight_sum += np.exp(-dist_sq / (2 * sigma**2))
        
        mask = weight_sum > 0.1
        true_flow[mask, 0] /= weight_sum[mask]
        true_flow[mask, 1] /= weight_sum[mask]
        
        I1 = np.clip(I1, 0, 1)
        I2 = np.clip(I2, 0, 1)
        
        return I1, I2, true_flow
    
    def evaluate_diso(I1, I2, true_flow, patch_radius=7, stride=4, iters=20, 
                     eps=1e-3, lam=0.01, sigma=2.0):
        """
        Run DISO and evaluate against ground truth.
        """
        print(f"\nDISO Parameters:")
        print(f"  Patch radius: {patch_radius}")
        print(f"  Stride: {stride}")
        print(f"  Iterations: {iters}")
        print(f"  Epsilon: {eps}")
        print(f"  Lambda: {lam}")
        print(f"  Sigma: {sigma}")
        
        # Run DISO
        import time
        start_time = time.time()
        disp_sparse = dis_inverse_search_multi(I1, I2, patch_radius, stride, iters, eps, lam, sigma)
        elapsed = time.time() - start_time
        print(f"  Computation time: {elapsed:.3f} seconds")
        
        # Densify the flow field
        estimated_flow = densify(disp_sparse, I1.shape[0], I1.shape[1], patch_radius, stride, sigma)
        
        # Compute error metrics
        flow_diff = estimated_flow - true_flow
        epe = np.sqrt(flow_diff[:, :, 0]**2 + flow_diff[:, :, 1]**2)
        mean_epe = np.mean(epe)
        max_epe = np.max(epe)
        
        print(f"\nError Metrics:")
        print(f"  Mean End-Point Error (EPE): {mean_epe:.3f} pixels")
        print(f"  Max End-Point Error: {max_epe:.3f} pixels")
        print(f"  Flow range - X: [{estimated_flow[:,:,0].min():.2f}, {estimated_flow[:,:,0].max():.2f}]")
        print(f"  Flow range - Y: [{estimated_flow[:,:,1].min():.2f}, {estimated_flow[:,:,1].max():.2f}]")
        
        return estimated_flow, epe
    
    # Test 1: Simple translation
    print("=" * 60)
    print("Test 1: Simple Translation (Single-Level)")
    print("=" * 60)
    I1_trans, I2_trans, true_flow_trans = create_moving_gaussian_pair(
        height=256, width=256, 
        displacement=(3.0, 5.0),  # (dy, dx)
        sigma=20.0, 
        num_gaussians=5
    )
    
    estimated_flow_trans, epe_trans = evaluate_diso(
        I1_trans, I2_trans, true_flow_trans,
        patch_radius=9, stride=3, iters=30, eps=1e-3, lam=0.001, sigma=3.0
    )
    
    # Test 1b: Multi-level translation (old implementation)
    print("\n" + "=" * 60)
    print("Test 1b: Simple Translation (Multi-Level Old)")
    print("=" * 60)
    
    diso_ml = DISOInverseSearch(mode="cv2")
    import time
    start_time = time.time()
    estimated_flow_trans_ml = diso_ml.get_displacement_multilevel(
        I1_trans, I2_trans,
        patch_radius=7, stride=3, iters=15,
        eps=1e-3, lam=0.001, sigma=2.0,
        min_level=0, levels=30, eta=0.75
    )
    elapsed_ml = time.time() - start_time
    
    # Test 1c: DIS-Fast translation
    print("\n" + "=" * 60)
    print("Test 1c: Simple Translation (DIS-Fast)")
    print("=" * 60)
    
    dis_fast = DISFast(mode="cv2", eta=0.75, use_ic=True)
    start_time = time.time()
    estimated_flow_trans_fast = dis_fast.multilevel(
        I1_trans, I2_trans,
        patch_radius=7, stride=4, iters=5,  # Fewer iterations needed with IC
        lam=1e-4, eps=1e-3, sigma=2.0,
        min_level=20,
        use_fb_check=False,
        use_edge_aware=True
    )
    elapsed_fast = time.time() - start_time
    
    # Compute error for multi-level
    flow_diff_ml = estimated_flow_trans_ml - true_flow_trans
    epe_ml = np.sqrt(flow_diff_ml[:, :, 0]**2 + flow_diff_ml[:, :, 1]**2)
    mean_epe_ml = np.mean(epe_ml)
    max_epe_ml = np.max(epe_ml)
    
    print(f"Multi-level Old computation time: {elapsed_ml:.3f} seconds")
    print(f"Multi-level Old Mean EPE: {mean_epe_ml:.3f} pixels")
    print(f"Multi-level Old Max EPE: {max_epe_ml:.3f} pixels")
    
    # Compute error for DIS-Fast
    flow_diff_fast = estimated_flow_trans_fast - true_flow_trans
    epe_fast = np.sqrt(flow_diff_fast[:, :, 0]**2 + flow_diff_fast[:, :, 1]**2)
    mean_epe_fast = np.mean(epe_fast)
    max_epe_fast = np.max(epe_fast)
    
    print(f"DIS-Fast computation time: {elapsed_fast:.3f} seconds")
    print(f"DIS-Fast Mean EPE: {mean_epe_fast:.3f} pixels")
    print(f"DIS-Fast Max EPE: {max_epe_fast:.3f} pixels")
    print(f"Speedup: {elapsed_ml/elapsed_fast:.2f}x faster")
    
    # Test 2: Rotation
    print("\n" + "=" * 60)
    print("Test 2: Rotational Motion")
    print("=" * 60)
    I1_rot, I2_rot, true_flow_rot = create_rotation_test(
        height=256, width=256,
        angle_deg=3.0,
        sigma=20.0
    )
    
    estimated_flow_rot, epe_rot = evaluate_diso(
        I1_rot, I2_rot, true_flow_rot,
        patch_radius=9, stride=3, iters=30, eps=1e-3, lam=0.001, sigma=3.0
    )
    
    # Test 2b: Multi-level rotation
    print("\n" + "=" * 60)
    print("Test 2b: Rotational Motion (Multi-Level)")
    print("=" * 60)
    
    estimated_flow_rot_ml = diso_ml.get_displacement_multilevel(
        I1_rot, I2_rot,
        patch_radius=7, stride=3, iters=15,
        eps=1e-3, lam=0.001, sigma=2.0,
        min_level=0, levels=30, eta=0.75
    )
    
    flow_diff_rot_ml = estimated_flow_rot_ml - true_flow_rot
    epe_rot_ml = np.sqrt(flow_diff_rot_ml[:, :, 0]**2 + flow_diff_rot_ml[:, :, 1]**2)
    mean_epe_rot_ml = np.mean(epe_rot_ml)
    
    print(f"Multi-level Mean EPE (Rotation): {mean_epe_rot_ml:.3f} pixels")
    
    # Test rotation with DIS-Fast
    print("\n" + "=" * 60)
    print("Test 2c: Rotational Motion (DIS-Fast)")
    print("=" * 60)
    
    estimated_flow_rot_fast = dis_fast.multilevel(
        I1_rot, I2_rot,
        patch_radius=7, stride=4, iters=5,
        lam=1e-4, eps=1e-3, sigma=2.0,
        min_level=20,
        use_fb_check=False,
        use_edge_aware=True
    )
    
    flow_diff_rot_fast = estimated_flow_rot_fast - true_flow_rot
    epe_rot_fast = np.sqrt(flow_diff_rot_fast[:, :, 0]**2 + flow_diff_rot_fast[:, :, 1]**2)
    mean_epe_rot_fast = np.mean(epe_rot_fast)
    
    print(f"DIS-Fast Mean EPE (Rotation): {mean_epe_rot_fast:.3f} pixels")
    
    # Visualization
    print("\n" + "=" * 60)
    print("Creating Visualizations...")
    print("=" * 60)
    
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    
    # Row 1: Translation test - Input and ground truth
    axes[0, 0].imshow(I1_trans)
    axes[0, 0].set_title("I1 (Translation)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(I2_trans)
    axes[0, 1].set_title("I2 (Translation)")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(flow_to_color(true_flow_trans))
    axes[0, 2].set_title("True Flow")
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(flow_to_color(estimated_flow_trans))
    axes[0, 3].set_title("Single-Level")
    axes[0, 3].axis('off')
    
    axes[0, 4].imshow(flow_to_color(estimated_flow_trans_ml))
    axes[0, 4].set_title("Multi-Level Old")
    axes[0, 4].axis('off')
    
    axes[0, 5].imshow(flow_to_color(estimated_flow_trans_fast))
    axes[0, 5].set_title("DIS-Fast")
    axes[0, 5].axis('off')
    
    # Row 2: Translation error maps
    im1 = axes[1, 0].imshow(epe_trans, cmap='hot', vmin=0, vmax=2)
    axes[1, 0].set_title(f"EPE Single-Level\n(Mean: {np.mean(epe_trans):.3f})")
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(epe_ml, cmap='hot', vmin=0, vmax=2)
    axes[1, 1].set_title(f"EPE Multi-Level\n(Mean: {mean_epe_ml:.3f})")
    axes[1, 1].axis('off')
    
    im3 = axes[1, 2].imshow(epe_fast, cmap='hot', vmin=0, vmax=2)
    axes[1, 2].set_title(f"EPE DIS-Fast\n(Mean: {mean_epe_fast:.3f})")
    axes[1, 2].axis('off')
    
    # Quiver plots
    quiver_trans_sl = quiver_visualization(
        I1_trans, estimated_flow_trans,
        scale=0.5, downsample=0.05, 
        show_streamlines=False,
        backend='opencv'
    )
    axes[1, 3].imshow(quiver_trans_sl)
    axes[1, 3].set_title("Quiver (Single-Level)")
    axes[1, 3].axis('off')
    
    quiver_trans_ml = quiver_visualization(
        I1_trans, estimated_flow_trans_ml,
        scale=0.5, downsample=0.05,
        show_streamlines=False,
        backend='opencv'
    )
    axes[1, 4].imshow(quiver_trans_ml)
    axes[1, 4].set_title("Quiver (Multi-Level)")
    axes[1, 4].axis('off')
    
    quiver_trans_fast = quiver_visualization(
        I1_trans, estimated_flow_trans_fast,
        scale=0.5, downsample=0.05,
        show_streamlines=False,
        backend='opencv'
    )
    axes[1, 5].imshow(quiver_trans_fast)
    axes[1, 5].set_title("Quiver (DIS-Fast)")
    axes[1, 5].axis('off')
    
    # Row 3: Rotation test
    axes[2, 0].imshow(I1_rot)
    axes[2, 0].set_title("I1 (Rotation)")
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(I2_rot)
    axes[2, 1].set_title("I2 (Rotation)")
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(flow_to_color(true_flow_rot))
    axes[2, 2].set_title("True Flow")
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(flow_to_color(estimated_flow_rot))
    axes[2, 3].set_title("Single-Level DISO")
    axes[2, 3].axis('off')
    
    axes[2, 4].imshow(flow_to_color(estimated_flow_rot_ml))
    axes[2, 4].set_title("Multi-Level DISO")
    axes[2, 4].axis('off')
    
    axes[2, 5].imshow(flow_to_color(estimated_flow_rot_fast))
    axes[2, 5].set_title("DIS-Fast (Rotation)")
    axes[2, 5].axis('off')
    
    # Row 4: Rotation error maps
    im4 = axes[3, 0].imshow(epe_rot, cmap='hot', vmin=0, vmax=2)
    axes[3, 0].set_title(f"EPE Single-Level\n(Mean: {np.mean(epe_rot):.3f})")
    axes[3, 0].axis('off')
    
    im5 = axes[3, 1].imshow(epe_rot_ml, cmap='hot', vmin=0, vmax=2)
    axes[3, 1].set_title(f"EPE Multi-Level\n(Mean: {mean_epe_rot_ml:.3f})")
    axes[3, 1].axis('off')
    
    im6 = axes[3, 2].imshow(epe_rot_fast, cmap='hot', vmin=0, vmax=2)
    axes[3, 2].set_title(f"EPE DIS-Fast\n(Mean: {mean_epe_rot_fast:.3f})")
    axes[3, 2].axis('off')
    
    # Quiver visualizations for rotation
    quiver_rot_sl = quiver_visualization(
        I1_rot, estimated_flow_rot,
        scale=0.5, downsample=0.05,
        show_streamlines=False,
        backend='opencv'
    )
    axes[3, 3].imshow(quiver_rot_sl)
    axes[3, 3].set_title("Quiver (Single)")
    axes[3, 3].axis('off')
    
    quiver_rot_ml = quiver_visualization(
        I1_rot, estimated_flow_rot_ml,
        scale=0.5, downsample=0.05,
        show_streamlines=False,
        backend='opencv'
    )
    axes[3, 4].imshow(quiver_rot_ml)
    axes[3, 4].set_title("Quiver (Multi)")
    axes[3, 4].axis('off')
    
    quiver_rot_fast = quiver_visualization(
        I1_rot, estimated_flow_rot_fast,
        scale=0.5, downsample=0.05,
        show_streamlines=False,
        backend='opencv'
    )
    axes[3, 5].imshow(quiver_rot_fast)
    axes[3, 5].set_title("Quiver (DIS-Fast)")
    axes[3, 5].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization complete!")
