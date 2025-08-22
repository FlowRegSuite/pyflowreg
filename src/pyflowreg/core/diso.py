import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def _gauss_kernel(radius, sigma):
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


@njit(fastmath=True, cache=True)
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


class DISOInverseSearch:
    def __init__(self, mode="ncc"):
        self.mode = mode
        self.inv = build_diso_inverse_search(diso_map[mode])

    def __call__(self, I1, I2, patch_radius=3, stride=5, iters=4, eps=1e-3, lam=1e-4, sigma=1.5):
        inv = self.inv
        disp = inv(I1, I2, patch_radius, stride, iters, eps, lam, sigma)
        H, W = I1.shape[0], I1.shape[1]
        flow = densify(disp, H, W, patch_radius, stride, sigma)

        return flow


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from pyflowreg.util.visualization import quiver_visualization, flow_to_color
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
    print("Test 1: Simple Translation")
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
    
    # Visualization
    print("\n" + "=" * 60)
    print("Creating Visualizations...")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Translation test
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
    axes[0, 3].set_title("DISO Estimated Flow")
    axes[0, 3].axis('off')
    
    # Row 2: Rotation test
    axes[1, 0].imshow(I1_rot)
    axes[1, 0].set_title("I1 (Rotation)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(I2_rot)
    axes[1, 1].set_title("I2 (Rotation)")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(flow_to_color(true_flow_rot))
    axes[1, 2].set_title("True Flow")
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(flow_to_color(estimated_flow_rot))
    axes[1, 3].set_title("DISO Estimated Flow")
    axes[1, 3].axis('off')
    
    # Row 3: Error maps and quiver plots
    im1 = axes[2, 0].imshow(epe_trans, cmap='hot', vmin=0)
    axes[2, 0].set_title("EPE Map (Translation)")
    axes[2, 0].axis('off')
    plt.colorbar(im1, ax=axes[2, 0], fraction=0.046)
    
    im2 = axes[2, 1].imshow(epe_rot, cmap='hot', vmin=0)
    axes[2, 1].set_title("EPE Map (Rotation)")
    axes[2, 1].axis('off')
    plt.colorbar(im2, ax=axes[2, 1], fraction=0.046)
    
    # Quiver visualization for translation
    quiver_trans = quiver_visualization(
        I1_trans, estimated_flow_trans,
        scale=0.5, downsample=0.05, 
        show_streamlines=False,
        backend='opencv'
    )
    axes[2, 2].imshow(quiver_trans)
    axes[2, 2].set_title("Quiver Plot (Translation)")
    axes[2, 2].axis('off')
    
    # Quiver visualization for rotation
    quiver_rot = quiver_visualization(
        I1_rot, estimated_flow_rot,
        scale=0.5, downsample=0.05,
        show_streamlines=True,
        backend='opencv'
    )
    axes[2, 3].imshow(quiver_rot)
    axes[2, 3].set_title("Quiver + Streamlines (Rotation)")
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization complete!")
