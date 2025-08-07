import numpy as np
from skimage.transform import resize
from scipy.ndimage import median_filter
import cv2
from pyflowreg.src import compute_flow
import numpy as np
from skimage.transform import warp
from scipy.ndimage import map_coordinates
from skimage.transform import resize as ski_resize
from numba import njit


def matlab_gradient(f, spacing):
    """Match MATLAB's gradient exactly"""
    grad = np.zeros_like(f)
    # Interior: central differences
    grad[1:-1] = (f[2:] - f[:-2]) / (2 * spacing)
    # Boundaries: one-sided (MATLAB style)
    grad[0] = (f[1] - f[0]) / spacing
    grad[-1] = (f[-1] - f[-2]) / spacing
    return grad


def resize_image_skimage(image, size):
    """
    Resize an image to the specified size using the given interpolation method.

    Parameters:
    - image: Input image to be resized.
    - size: Tuple (width, height) specifying the new size.
    - interpolation: Interpolation method to use for resizing.

    Returns:
    - Resized image.
    """
    img = ski_resize(image, (size[1], size[0]), order=3, mode='edge', anti_aliasing=True)
    return img


def resize_image_cv2(image, size):
    """
    Resize an image to the specified size using the given interpolation method.

    Parameters:
    - image: Input image to be resized.
    - size: Tuple (width, height) specifying the new size.
    - interpolation: Interpolation method to use for resizing.

    Returns:
    - Resized image.
    """
    img = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LANCZOS4)
    return img


def resize_image_cv2_aa_gauss(image, size):
    h_orig, w_orig = image.shape[:2]
    h_new, w_new = size[:2]
    scale = min(h_new / h_orig, w_new / w_orig)

    if scale < 1:
        sigma = 0.6 / scale
        ksize = int(2 * np.ceil(2 * sigma) + 1)
        image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    img = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return img


def imresize_cv_exact(img, size):
    h, w = img.shape[:2]
    sx, sy = size[1] / w, size[0] / h
    scale = min(sx, sy)
    if scale < 1:
        sigma = 0.6 / scale
        k = int(2 * np.ceil(2 * sigma) + 1)
        g = cv2.getGaussianKernel(k, sigma)
        img = cv2.sepFilter2D(img, -1, g, g, borderType=cv2.BORDER_REFLECT101)
    return cv2.resize(img, size[1::-1], interpolation=cv2.INTER_CUBIC)


def resize_image_cv2_aa_blur(image, size):
    h_orig, w_orig = image.shape[:2]
    h_new, w_new = size[:2]

    if h_new < h_orig or w_new < w_orig:
        # Box filter - much faster than Gaussian, still removes aliasing
        scale = max(h_orig/h_new, w_orig/w_new)
        ksize = int(scale) | 1  # Ensure odd
        if ksize > 1:
            image = cv2.blur(image, (ksize, ksize))

    img = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return img


resize = imresize_cv_exact


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
    return cv2.copyMakeBorder(f, 1, 1, 1, 1, borderType=cv2.BORDER_REPLICATE)


def add_boundary(f):
    return np.pad(f, 1, mode='edge')


def get_motion_tensor_gc(f1, f2, hy, hx):
    f1p = np.pad(f1, ((1,1),(1,1)), mode='symmetric')
    f2p = np.pad(f2, ((1,1),(1,1)), mode='symmetric')
    _, fx1p = np.gradient(f1p, hy, hx)
    _, fx2p = np.gradient(f2p, hy, hx)
    fx = 0.5*(fx1p + fx2p)
    ft = f2p - f1p
    fx = np.pad(fx[1:-1, 1:-1], 1, mode='symmetric')
    ft = np.pad(ft[1:-1, 1:-1], 1, mode='symmetric')

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
        arr[:, 0] = 0
        arr[:, -1] = 0
        arr[0, :] = 0
        arr[-1, :] = 0
    return J11, J22, J33, J12, J13, J23


def level_solver(J11, J22, J33, J12, J13, J23, weight, u, v, alpha, iterations, update_lag, verbose, a_data, a_smooth, hx, hy):
    # call the C++ solver
    result = compute_flow(J11, J22, J33, J12, J13, J23, weight=weight,
                          u=u, v=v,
                          alpha_x=alpha[0], alpha_y=alpha[1], iterations=iterations, update_lag=update_lag,
                          a_data=a_data, a_smooth=a_smooth, hx=hx, hy=hy)

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
            du[1:-1, 1:-1] = median_filter(du[1:-1, 1:-1], size=(5, 5), mode='mirror')
            dv[1:-1, 1:-1] = median_filter(dv[1:-1, 1:-1], size=(5, 5), mode='mirror')
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
        f1_level = resize(f1_low, level_size)
        f2_level = resize(f2_low, level_size)
        if f1_level.ndim == 2:
            f1_level = f1_level[:, :, np.newaxis]
            f2_level = f2_level[:, :, np.newaxis]
        current_hx = float(m) / f1_level.shape[0]
        current_hy = float(n) / f1_level.shape[1]
        if i == max(max_level_x, max_level_y):
            u = add_boundary(resize(u_init, level_size))
            v = add_boundary(resize(v_init, level_size))
            tmp = f2_level.copy()
        else:
            u = add_boundary(resize(u[1:-1, 1:-1], level_size))
            v = add_boundary(resize(v[1:-1, 1:-1], level_size))
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
            weight_level = resize(weight, f1_level.shape)
        weight_level = cv2.copyMakeBorder(
            weight_level, 1, 1, 1, 1,
            borderType=cv2.BORDER_CONSTANT, value=0.0)

        if i == min_level:
            alpha_scaling = 1
        else:
            alpha_scaling = eta**(-0.5 * i)

        alpha_tmp = [alpha_scaling * alpha[j] for j in range(len(alpha))]

        du, dv = level_solver(np.ascontiguousarray(J11),
                              np.ascontiguousarray(J22),
                              np.ascontiguousarray(J33),
                              np.ascontiguousarray(J12),
                              np.ascontiguousarray(J13),
                              np.ascontiguousarray(J23),
                              np.ascontiguousarray(weight_level), u, v,
                              alpha_tmp, iterations, update_lag, 0, a_data_arr, a_smooth, current_hx, current_hy)
        if min(level_size) > 5:
            du[1:-1, 1:-1] = median_filter(du[1:-1, 1:-1], size=(5, 5), mode='mirror')
            dv[1:-1, 1:-1] = median_filter(dv[1:-1, 1:-1], size=(5, 5), mode='mirror')
        u = u + du
        v = v + dv
    w = np.zeros((u.shape[0]-2, u.shape[1]-2, 2), dtype=np.float64)
    w[:, :, 0] = u[1:-1, 1:-1]
    w[:, :, 1] = v[1:-1, 1:-1]
    if min_level > 0:
        w = cv2.resize(w, (n, m), interpolation=cv2.INTER_CUBIC)
    return w