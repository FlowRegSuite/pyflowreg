import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def _gauss_kernel(radius, sigma):
    k = np.empty((2*radius+1, 2*radius+1), np.float32)
    s2 = 2.0 * sigma * sigma
    idx = 0
    for y in range(-radius, radius+1):
        for x in range(-radius, radius+1):
            k.flat[idx] = np.exp(-(x*x + y*y)/s2)
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
            for x in range(1, W-1):
                gx[y, x, c] = 0.5*(img[y, x+1, c] - img[y, x-1, c])
        for y in range(1, H-1):
            for x in range(W):
                gy[y, x, c] = 0.5*(img[y+1, x, c] - img[y-1, x, c])
        for y in range(H):
            gx[y, 0, c] = img[y, 1, c] - img[y, 0, c]
            gx[y, W-1, c] = img[y, W-1, c] - img[y, W-2, c]
        for x in range(W):
            gy[0, x, c] = img[1, x, c] - img[0, x, c]
            gy[H-1, x, c] = img[H-1, x, c] - img[H-2, x, c]
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
    v0 = v00 + fx*(v01 - v00)
    v1 = v10 + fx*(v11 - v10)
    return v0 + fy*(v1 - v0)

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
    for dy in range(-radius, radius+1):
        y = yc + dy
        for dx in range(-radius, radius+1):
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
                rr += r*r
            ww = 1.0 / np.sqrt(rr + eps*eps)
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
def _solve2x2(H00, H01, H11, b0, b1, lam):
    H00 += lam
    H11 += lam
    det = H00*H11 - H01*H01
    if det <= 1e-12:
        return 0.0, 0.0
    inv00 =  H11 / det
    inv01 = -H01 / det
    inv11 =  H00 / det
    dx = inv00*b0 + inv01*b1
    dy = inv01*b0 + inv11*b1
    return dx, dy

@njit(fastmath=True, cache=True)
def dis_inverse_search_multi(I1, I2, patch_radius, stride, iters, eps, lam, sigma):
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
                H00, H01, H11, b0, b1 = _accumulate_system(I1, I2, gx2, gy2, yc, xc, (u0x, u0y), patch_radius, wpatch, eps)
                dx, dy = _solve2x2(H00, H01, H11, b0, b1, lam)
                u0x += dx
                u0y += dy
                if dx*dx + dy*dy < 1e-8:
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
        yc = patch_radius + iy*stride
        for ix in range(nx):
            xc = patch_radius + ix*stride
            u0x = disp[iy, ix, 0]
            u0y = disp[iy, ix, 1]
            idx = 0
            for dy in range(-patch_radius, patch_radius+1):
                y = yc + dy
                if y < 0 or y >= H:
                    idx += 2*patch_radius+1
                    continue
                for dx in range(-patch_radius, patch_radius+1):
                    x = xc + dx
                    w = wpatch.flat[idx]
                    idx += 1
                    if x < 0 or x >= W:
                        continue
                    wsum[y, x] += w
                    flow[y, x, 0] += w*u0x
                    flow[y, x, 1] += w*u0y
    for y in range(H):
        for x in range(W):
            s = wsum[y, x]
            if s > 0.0:
                s = 1.0/s
                flow[y, x, 0] *= s
                flow[y, x, 1] *= s
    return flow


if __name__ == "__main__":
    # Example usage
    I1 = np.random.rand(100, 100, 3).astype(np.float32)
    I2 = np.random.rand(100, 100, 3).astype(np.float32)
    patch_radius = 5
    stride = 10
    iters = 10
    eps = 1e-6
    lam = 0.1
    sigma = 1.0

    disp = dis_inverse_search_multi(I1, I2, patch_radius, stride, iters, eps, lam, sigma)
    flow = densify(disp, I1.shape[0], I1.shape[1], patch_radius, stride, sigma)
    print(flow.shape)
