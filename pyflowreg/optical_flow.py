import numpy as np
from skimage.transform import resize
from scipy.ndimage import median_filter
import cv2
from pyflowreg.src import compute_flow
import numpy as np
from skimage.transform import warp
from scipy.ndimage import map_coordinates


def imregister_wrapper(f2_level, u, v, f1_level, interpolation_method='cubic'):
    if f2_level.ndim == 2:
        f2_level = f2_level[:, :, None]
        f1_level = f1_level[:, :, None]
    #f2_level = f2_level[1:-1, 1:-1]
    #f1_level = f1_level[1:-1, 1:-1]
    #u = u[1:-1, 1:-1]
    #v = v[1:-1, 1:-1]
    H, W, C = f2_level.shape
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    map_x = (grid_x + u).astype(np.float32)
    map_y = (grid_y + v).astype(np.float32)
    out_of_bounds = (map_x < 0) | (map_x >= W) | (map_y < 0) | (map_y >= H)
    map_x_clipped = np.clip(map_x, 0, W - 1).astype(np.float32)
    map_y_clipped = np.clip(map_y, 0, H - 1).astype(np.float32)
    if interpolation_method.lower() == 'cubic':
        interp = cv2.INTER_CUBIC
    elif interpolation_method.lower() == 'linear':
        interp = cv2.INTER_LINEAR
    else:
        raise ValueError("Unsupported interpolation method. Use 'linear' or 'cubic'.")
    warped = np.empty_like(f2_level, dtype=np.float32)
    for c in range(C):
        warped[:, :, c] = cv2.remap(f2_level[:, :, c], map_x_clipped, map_y_clipped, interpolation=interp, borderMode=cv2.BORDER_REPLICATE)
    #idx = warped == -1
    #warped[idx] = f1_level[idx]
    for c in range(C):
        warped[:, :, c][out_of_bounds] = f1_level[:, :, c][out_of_bounds]
    if warped.shape[2] == 1:
        warped = warped[:, :, 0]
    return warped


def warpingDepth(eta, levels, m, n):
    min_dim = min(m, n)
    warpingdepth = 0
    for _ in range(levels):
        warpingdepth += 1
        min_dim *= eta
        if round(min_dim) < 10:
            break
    return warpingdepth

def add_boundary_old(f):
    # Pad and set boundary as in MATLAB
    g = np.pad(f, ((1, 1), (1, 1)), mode='symmetric')
    g[:, 0] = g[:, 1]
    g[:, -1] = g[:, -2]
    g[0, :] = g[1, :]
    g[-1, :] = g[-2, :]
    return g

def add_boundary(f):
    return cv2.copyMakeBorder(f, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT_101)

def imregister_wrapper_opencv(f2_level, u, v, f1_level, interpolation_method='cubic'):
    # Warp image f2_level towards f1_level using displacement fields u,v
    single_channel = False
    if f2_level.ndim == 2:
        f2_level = f2_level[:, :, np.newaxis]
        f1_level = f1_level[:, :, np.newaxis]
        single_channel = True

    H, W, C = f2_level.shape
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Remember: u is vertical displacement (add to y), v is horizontal (add to x)
    map_x = (grid_x + u).astype(np.float32)
    map_y = (grid_y + v).astype(np.float32)

    out_of_bounds = (map_x < 0) | (map_x >= W) | (map_y < 0) | (map_y >= H)
    map_x_clipped = np.clip(map_x, 0, W - 1).astype(np.float32)
    map_y_clipped = np.clip(map_y, 0, H - 1).astype(np.float32)

    if interpolation_method.lower() == 'cubic':
        interp = cv2.INTER_CUBIC
    elif interpolation_method.lower() == 'linear':
        interp = cv2.INTER_LINEAR
    else:
        raise ValueError("Unsupported interpolation method. Use 'linear' or 'cubic'.")

    warped = np.empty_like(f2_level, dtype=np.float32)
    for c in range(C):
        warped[:, :, c] = cv2.remap(
            f2_level[:, :, c],
            map_x_clipped, map_y_clipped,
            interpolation=interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1
        )

    idx = warped == -1
    warped[idx] = f1_level[idx]

    for c in range(C):
        warped[:, :, c][out_of_bounds] = f1_level[:, :, c][out_of_bounds]

    if single_channel:
        warped = warped[:, :, 0]

    return warped


def get_motion_tensor_gc_old(f1, f2, hx, hy):
    # Just like MATLAB:
    # MATLAB: [fx1, fy1] = gradient(f1, hx, hy)
    # MATLAB returns: fx1 = dF/dy (with spacing hx?), fy1 = dF/dx (with spacing hy?)
    # We'll do the exact same call in Python:
    # np.gradient returns [dF/drow, dF/dcol]
    # Here we pass hx as spacing for rows, hy for columns, just like MATLAB order
    fx1_grad = np.gradient(f1, hx, hy)  # returns [df1/dy, df1/dx]
    fx1 = fx1_grad[0]  # corresponds to df/dy in MATLAB (their first output)
    # The MATLAB code discards the second output for fx1, let's remain faithful
    # Actually, original code: [fx1, ~] = gradient(f1, hx, hy);
    # means fx1 is df/dy and we don't use df/dx here.

    fx2_grad = np.gradient(f2, hx, hy)
    fx2 = fx2_grad[0]  # similarly, df2/dy

    # average fx
    fx = 0.5 * (fx1 + fx2)

    ft = f2 - f1

    # Next: [~, fxy] = gradient(fx, hx, hy)
    # means we compute gradient(fx, hx, hy) and take the second output for fxy
    fxy_grad = np.gradient(fx, hx, hy)
    fxy = fxy_grad[1]  # fxy = derivative along x-direction

    # [fxt, fyt] = gradient(ft, hx, hy)
    # means fxt = first output (df/dy), fyt = second output (df/dx)
    ft_grad = np.gradient(ft, hx, hy)
    fxt = ft_grad[0]  # df/dy
    fyt = ft_grad[1]  # df/dx

    # [fxx1, fyy1] = gradient2(f1, hx, hy)
    # [fxx2, fyy2] = gradient2(f2, hx, hy)
    # same logic, gradient2 does second derivatives
    def gradient2(f, hx_, hy_):
        fxx = np.zeros_like(f)
        fyy = np.zeros_like(f)
        # second derivatives as in MATLAB
        fxx[1:-1, 1:-1] = (f[1:-1, 0:-2] - 2*f[1:-1, 1:-1] + f[1:-1, 2:]) / (hx_**2)
        fyy[1:-1, 1:-1] = (f[0:-2, 1:-1] - 2*f[1:-1, 1:-1] + f[2:, 1:-1]) / (hy_**2)
        return fxx, fyy

    f1p = np.pad(f1, ((1,1),(1,1)), mode='symmetric')
    f2p = np.pad(f2, ((1,1),(1,1)), mode='symmetric')

    # We must recompute fx,fy after padding if that was done originally?
    # The original code padded early. Let's follow original code exactly:
    # The original code got gradients after padding. We should do the same steps as original code:
    # Actually, let's trust original code structure:
    # The code snippet given was presumably already correct. We'll just trust the snippet:

    # Recompute with the correct approach:
    f1p = np.pad(f1, ((1, 1), (1, 1)), 'symmetric')
    f2p = np.pad(f2, ((1, 1), (1, 1)), 'symmetric')

    fy1p, fx1p = np.gradient(f1p, hx, hy)  # fy1p = dF1/dy, fx1p = dF1/dx
    fy2p, fx2p = np.gradient(f2p, hx, hy)  # fy2p = dF2/dy, fx2p = dF2/dx

    fx = 0.5*(fx1p + fx2p)
    ft = f2p - f1p

    # from original MATLAB code snippet:
    # [ ~ , fxy] = gradient(fx, hx, hy);
    # means do gradient and take second output for fxy:
    tmp_grad = np.gradient(fx, hx, hy)
    fxy = tmp_grad[1]

    ft_grad = np.gradient(ft, hx, hy)
    fxt = ft_grad[0]
    fyt = ft_grad[1]

    fxx1, fyy1 = gradient2(f1p, hx, hy)
    fxx2, fyy2 = gradient2(f2p, hx, hy)

    fxx = 0.5*(fxx1 + fxx2)
    fyy = 0.5*(fyy1 + fyy2)

    # reg_x, reg_y as in original code:
    reg_x = 1.0 / ((np.sqrt(fxx**2 + fxy**2)**2) + 0.000001)
    reg_y = 1.0 / ((np.sqrt(fxy**2 + fyy**2)**2) + 0.000001)

    J11 = reg_x * fxx**2 + reg_y * fxy**2
    J22 = reg_x * fxy**2 + reg_y * fyy**2
    J33 = reg_x * fxt**2 + reg_y * fyt**2
    J12 = reg_x * fxx * fxy + reg_y * fxy * fyy
    J13 = reg_x * fxx * fxt + reg_y * fxy * fyt
    J23 = reg_x * fxy * fxt + reg_y * fyy * fyt

    # set boundaries to 0
    for arr in [J11, J22, J33, J12, J13, J23]:
        arr[:, 0] = 0; arr[:, -1] = 0; arr[0, :] = 0; arr[-1, :] = 0

    return J11[1:-1, 1:-1], J22[1:-1, 1:-1], J33[1:-1, 1:-1], \
           J12[1:-1, 1:-1], J13[1:-1, 1:-1], J23[1:-1, 1:-1]


def get_motion_tensor_gc(f1, f2, hy, hx):
    f1p = np.pad(f1, ((1,1),(1,1)), mode='symmetric')
    f2p = np.pad(f2, ((1,1),(1,1)), mode='symmetric')
    _, fx1p = np.gradient(f1p, hy, hx)
    _, fx2p = np.gradient(f2p, hy, hx)
    fx = 0.5*(fx1p + fx2p)
    ft = f2p - f1p
    #fx = np.pad(fx[1:-1, 1:-1], 1, mode='symmetric')
    #ft = np.pad(ft[1:-1, 1:-1], 1, mode='symmetric')

    tmp_grad = np.gradient(fx, hy, hx)
    fxy = tmp_grad[0]
    ft_grad = np.gradient(ft, hy, hx)
    fxt = ft_grad[1]
    fyt = ft_grad[0]
    def gradient2(f, hx_, hy_):
        fxx = np.zeros_like(f)
        fyy = np.zeros_like(f)
        fxx[1:-1, 1:-1] = (f[1:-1, 0:-2] - 2*f[1:-1, 1:-1] + f[1:-1, 2:]) / (hx_**2)
        fyy[1:-1, 1:-1] = (f[0:-2, 1:-1] - 2*f[1:-1, 1:-1] + f[2:, 1:-1]) / (hy_**2)
        return fxx, fyy
    fxx1, fyy1 = gradient2(f1p, hy, hx)
    fxx2, fyy2 = gradient2(f2p, hy, hx)
    fxx = 0.5*(fxx1 + fxx2)
    fyy = 0.5*(fyy1 + fyy2)
    reg_x = 1.0 / ((np.sqrt(fxx**2 + fxy**2)**2) + 1e-6)
    reg_y = 1.0 / ((np.sqrt(fxy**2 + fyy**2)**2) + 1e-6)
    J11 = reg_x * fxx**2 + reg_y * fxy**2
    J22 = reg_x * fxy**2 + reg_y * fyy**2
    J33 = reg_x * fxt**2 + reg_y * fyt**2
    J12 = reg_x * fxx * fxy + reg_y * fxy * fyy
    J13 = reg_x * fxx * fxt + reg_y * fxy * fyt
    J23 = reg_x * fxy * fxt + reg_y * fyy * fyt
    for arr in [J11, J22, J33, J12, J13, J23]:
        arr[:, 0] = 0; arr[:, -1] = 0; arr[0, :] = 0; arr[-1, :] = 0
    return J11, J22, J33, J12, J13, J23



def level_solver(J11, J22, J33, J12, J13, J23, weight, u, v, alpha, iterations, update_lag, verbose, a_data, a_smooth, hx, hy):
    # call the C++ solver
    result = compute_flow(J11, J22, J33, J12, J13, J23, weight,
                          u[1:-1, 1:-1], v[1:-1, 1:-1],
                          alpha[0], alpha[1], iterations, update_lag,
                          a_data, a_smooth, hx, hy)

    # du = np.zeros_like(u)
    # dv = np.zeros_like(v)
    du = result[:, :, 0]
    dv = result[:, :, 1]
    return du, dv

def get_displacement_old(fixed, moving, alpha=(2,2), update_lag=10, iterations=20, min_level=0, levels=50, eta=0.8, a_smooth=0.5, a_data=0.45, const_assumption='gc', weight=None):
    fixed = fixed.astype(np.float64)
    moving = moving.astype(np.float64)
    if fixed.ndim == 3:
        m, n, n_channels = fixed.shape
    else:
        m, n = fixed.shape
        n_channels = 1
        fixed = fixed[:, :, np.newaxis]
        moving = moving[:, :, np.newaxis]
    u_init = np.zeros((m, n), dtype=np.float64)
    v_init = np.zeros((m, n), dtype=np.float64)
    if weight is None:
        weight = np.ones((m, n, n_channels), dtype=np.float64) / n_channels
    else:
        weight = weight.astype(np.float64)
        if weight.ndim < 3:
            weight = np.ones((m, n, n_channels), dtype=np.float64) * weight
    if not isinstance(a_data, np.ndarray):
        a_data_arr = np.full(n_channels, a_data, dtype=np.float64)
    else:
        a_data_arr = a_data
    a_data_arr = np.ascontiguousarray(a_data_arr)
    f1_low = fixed
    f2_low = moving
    max_level_y = warpingDepth(eta, levels, m, n)
    max_level_x = warpingDepth(eta, levels, m, n)
    max_level = min(max_level_x, max_level_y) * 4
    max_level_y = min(max_level_y, max_level)
    max_level_x = min(max_level_x, max_level)
    if max(max_level_x, max_level_y) <= min_level:
        min_level = max(max_level_x, max_level_y) - 1
    if min_level < 0:
        min_level = 0
    u = None
    v = None
    for i in range(max(max_level_x, max_level_y), min_level - 1, -1):
        level_size = (int(round(m * eta**(min(i, max_level_y)))), int(round(n * eta**(min(i, max_level_x)))))
        f1_level = cv2.resize(f1_low, (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC)
        f2_level = cv2.resize(f2_low, (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC)
        current_hx = float(m) / f1_level.shape[0]
        current_hy = float(n) / f1_level.shape[1]
        if i == max(max_level_x, max_level_y):
            u = add_boundary(cv2.resize(u_init, (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC))
            v = add_boundary(cv2.resize(v_init, (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC))
            tmp = f2_level.copy()
        else:
            u = add_boundary(cv2.resize(u[1:-1, 1:-1], (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC))
            v = add_boundary(cv2.resize(v[1:-1, 1:-1], (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC))
            tmp = imregister_wrapper(f2_level, u[1:-1,1:-1]/current_hy, v[1:-1,1:-1]/current_hx, f1_level)
        u = np.ascontiguousarray(u)
        v = np.ascontiguousarray(v)
        J11 = np.zeros_like(f1_level, dtype=np.float64)
        J22 = np.zeros_like(f1_level, dtype=np.float64)
        J33 = np.zeros_like(f1_level, dtype=np.float64)
        J12 = np.zeros_like(f1_level, dtype=np.float64)
        J13 = np.zeros_like(f1_level, dtype=np.float64)
        J23 = np.zeros_like(f1_level, dtype=np.float64)
        for ch in range(n_channels):
            J11_ch, J22_ch, J33_ch, J12_ch, J13_ch, J23_ch = get_motion_tensor_gc(f1_level[:, :, ch], tmp[:, :, ch], current_hx, current_hy)
            J11[:, :, ch] = J11_ch
            J22[:, :, ch] = J22_ch
            J33[:, :, ch] = J33_ch
            J12[:, :, ch] = J12_ch
            J13[:, :, ch] = J13_ch
            J23[:, :, ch] = J23_ch
        weight_level = weight
        if weight.shape[:2] != f1_level.shape[:2]:
            weight_level = cv2.resize(weight, (f1_level.shape[1], f1_level.shape[0]), interpolation=cv2.INTER_CUBIC)
        weight_level = cv2.copyMakeBorder(
            weight_level, 1, 1, 1, 1,
            borderType=cv2.BORDER_CONSTANT, value=0.0)
        du, dv = level_solver(np.ascontiguousarray(J11), np.ascontiguousarray(J22), np.ascontiguousarray(J33), np.ascontiguousarray(J12), np.ascontiguousarray(J13), np.ascontiguousarray(J23), np.ascontiguousarray(weight_level), u, v, alpha, iterations, update_lag, 0, a_data_arr, a_smooth, current_hx, current_hy)
        if min(level_size) > 5:
            du[1:-1, 1:-1] = median_filter(du[1:-1, 1:-1], size=(5, 5), mode='reflect')
            dv[1:-1, 1:-1] = median_filter(dv[1:-1, 1:-1], size=(5, 5), mode='reflect')
        u = u + du
        v = v + dv
    w = np.zeros((u.shape[0]-2, u.shape[1]-2, 2), dtype=np.float64)
    w[:, :, 0] = u[1:-1, 1:-1]
    w[:, :, 1] = v[1:-1, 1:-1]
    if min_level > 0:
        w = cv2.resize(w, (n, m), interpolation=cv2.INTER_CUBIC)
    return w


def get_displacement(fixed, moving, alpha=(2,2), update_lag=10, iterations=20, min_level=0, levels=50, eta=0.8, a_smooth=0.5, a_data=0.45, const_assumption='gc', weight=None):
    fixed = fixed.astype(np.float64)
    moving = moving.astype(np.float64)
    if fixed.ndim == 3:
        m, n, n_channels = fixed.shape
    else:
        m, n = fixed.shape
        n_channels = 1
        fixed = fixed[:, :, np.newaxis]
        moving = moving[:, :, np.newaxis]
    u_init = np.zeros((m, n), dtype=np.float64)
    v_init = np.zeros((m, n), dtype=np.float64)
    if weight is None:
        weight = np.ones((m, n, n_channels), dtype=np.float64) / n_channels
    else:
        weight = weight.astype(np.float64)
        if weight.ndim < 3:
            weight = np.ones((m, n, n_channels), dtype=np.float64) * weight
    if not isinstance(a_data, np.ndarray):
        a_data_arr = np.full(n_channels, a_data, dtype=np.float64)
    else:
        a_data_arr = a_data
    a_data_arr = np.ascontiguousarray(a_data_arr)
    f1_low = fixed
    f2_low = moving
    max_level_y = warpingDepth(eta, levels, m, n)
    max_level_x = warpingDepth(eta, levels, m, n)
    max_level = min(max_level_x, max_level_y) * 4
    max_level_y = min(max_level_y, max_level)
    max_level_x = min(max_level_x, max_level)
    if max(max_level_x, max_level_y) <= min_level:
        min_level = max(max_level_x, max_level_y) - 1
    if min_level < 0:
        min_level = 0
    u = None
    v = None
    for i in range(max(max_level_x, max_level_y), min_level - 1, -1):
        level_size = (int(round(m * eta**(min(i, max_level_y)))), int(round(n * eta**(min(i, max_level_x)))))
        f1_level = cv2.resize(f1_low, (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC)
        f2_level = cv2.resize(f2_low, (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC)
        if f1_level.ndim == 2:
            f1_level = f1_level[:, :, np.newaxis]
            f2_level = f2_level[:, :, np.newaxis]
        current_hx = float(m) / f1_level.shape[0]
        current_hy = float(n) / f1_level.shape[1]
        if i == max(max_level_x, max_level_y):
            u = add_boundary(cv2.resize(u_init, (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC))
            v = add_boundary(cv2.resize(v_init, (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC))
            tmp = f2_level.copy()
        else:
            u = add_boundary(cv2.resize(u[1:-1, 1:-1], (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC))
            v = add_boundary(cv2.resize(v[1:-1, 1:-1], (level_size[1], level_size[0]), interpolation=cv2.INTER_CUBIC))
            tmp = imregister_wrapper(f2_level, u[1:-1,1:-1]/current_hy, v[1:-1,1:-1]/current_hx, f1_level)
        if tmp.ndim == 2:
            tmp = tmp[:, :, np.newaxis]
        u = np.ascontiguousarray(u)
        v = np.ascontiguousarray(v)
        J_size = (f1_level.shape[0] + 2, f1_level.shape[1] + 2, n_channels)
        J11 = np.zeros(J_size, dtype=np.float64)
        J22 = np.zeros(J_size, dtype=np.float64)
        J33 = np.zeros(J_size, dtype=np.float64)
        J12 = np.zeros(J_size, dtype=np.float64)
        J13 = np.zeros(J_size, dtype=np.float64)
        J23 = np.zeros(J_size, dtype=np.float64)
        for ch in range(n_channels):
            J11_ch, J22_ch, J33_ch, J12_ch, J13_ch, J23_ch = get_motion_tensor_gc(f1_level[:, :, ch], tmp[:, :, ch], current_hx, current_hy)
            J11[:, :, ch] = J11_ch
            J22[:, :, ch] = J22_ch
            J33[:, :, ch] = J33_ch
            J12[:, :, ch] = J12_ch
            J13[:, :, ch] = J13_ch
            J23[:, :, ch] = J23_ch
        weight_level = weight
        if weight.shape[:2] != f1_level.shape[:2]:
            weight_level = cv2.resize(weight, (f1_level.shape[1], f1_level.shape[0]), interpolation=cv2.INTER_CUBIC)
        weight_level = cv2.copyMakeBorder(
            weight_level, 1, 1, 1, 1,
            borderType=cv2.BORDER_CONSTANT, value=0.0)

        if i == min_level:
            alpha_scaling = 1
        else:
            alpha_scaling = eta**(-0.5 * i)

        alpha_tmp = [alpha_scaling * alpha[i] for i in range(len(alpha))]

        du, dv = level_solver(np.ascontiguousarray(J11),
                              np.ascontiguousarray(J22),
                              np.ascontiguousarray(J33),
                              np.ascontiguousarray(J12),
                              np.ascontiguousarray(J13),
                              np.ascontiguousarray(J23),
                              np.ascontiguousarray(weight_level), u, v,
                              alpha_tmp, iterations, update_lag, 0, a_data_arr, a_smooth, current_hx, current_hy)
        if min(level_size) > 5:
            du[1:-1, 1:-1] = median_filter(du[1:-1, 1:-1], size=(5, 5), mode='reflect')
            dv[1:-1, 1:-1] = median_filter(dv[1:-1, 1:-1], size=(5, 5), mode='reflect')
        u = u + du
        v = v + dv
    w = np.zeros((u.shape[0]-2, u.shape[1]-2, 2), dtype=np.float64)
    w[:, :, 0] = u[1:-1, 1:-1]
    w[:, :, 1] = v[1:-1, 1:-1]
    if min_level > 0:
        w = cv2.resize(w, (n, m), interpolation=cv2.INTER_CUBIC)
    return w