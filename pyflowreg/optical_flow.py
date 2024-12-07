import numpy as np
from skimage.transform import resize
from scipy.ndimage import median_filter
from concurrent.futures import ProcessPoolExecutor



def get_displacements(c, c_ref, alpha=(2, 2), iterations=20, update_lag=10, a_data=0.45, a_smooth=0.5, hx=1.0, hy=1.0, weight=None):
    """
    Compute displacements for each frame in a sequence relative to a reference frame.

    Args:
        c (numpy.ndarray): Input array of shape (m, n, n_channels, t) representing a sequence of frames.
        c_ref (numpy.ndarray): Reference frame array of shape (m, n, n_channels).
        alpha (tuple): Smoothness parameters for the solver (alpha_x, alpha_y).
        iterations (int): Number of iterations for the solver.
        update_lag (int): Interval at which to update non-linearities.
        a_data (float): Data term exponent parameter.
        a_smooth (float): Smoothness term exponent parameter.
        hx (float): Spatial resolution in x-direction.
        hy (float): Spatial resolution in y-direction.
        weight (numpy.ndarray or None): Optional weight array of shape (m, n, n_channels).

    Returns:
        numpy.ndarray: Displacements array of shape (m, n, 2, t).
    """
    c = np.asarray(c)
    c_ref = np.asarray(c_ref)
    m, n, n_channels, t = c.shape
    if t == 1:
        return get_displacement(c_ref, c, alpha=alpha, iterations=iterations, update_lag=update_lag,
                                a_data=a_data, a_smooth=a_smooth, hx=hx, hy=hy, weight=weight)
    else:
        frames = [c[:, :, :, i] for i in range(t)]
        def process_frame(frame):
            return get_displacement(c_ref, frame, alpha=alpha, iterations=iterations, update_lag=update_lag,
                                    a_data=a_data, a_smooth=a_smooth, hx=hx, hy=hy, weight=weight)
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_frame, frames))
        return np.stack(results, axis=-1)


def warpingDepth(eta, levels, dim, dim_ref):
    """
    Compute the warping depth level for pyramid construction.

    Args:
        eta (float): Scaling factor per level.
        levels (int): Maximum number of levels.
        dim (int): Dimension of the current image size.
        dim_ref (int): Reference dimension.

    Returns:
        int: Computed warping depth.
    """
    return int(np.floor(np.log(dim/dim_ref) / np.log(1/eta))) if dim != dim_ref else 0


def add_boundary(f):
    """
    Add boundary conditions to an array by padding and adjusting edges.

    Args:
        f (numpy.ndarray): Input 2D array.

    Returns:
        numpy.ndarray: Padded array with adjusted boundaries.
    """
    g = np.pad(f, ((1, 1), (1, 1)), mode='edge')
    g[:, 0] = g[:, 1]
    g[:, -1] = g[:, -2]
    g[0, :] = g[1, :]
    g[-1, :] = g[-2, :]
    return g


def imregister_wrapper(f2_level, u, v, f1_level):
    """
    Warp one image (f2_level) toward another (f1_level) using displacement fields (u, v).

    Args:
        f2_level (numpy.ndarray): Image to be warped.
        u (numpy.ndarray): Displacements in y-direction.
        v (numpy.ndarray): Displacements in x-direction.
        f1_level (numpy.ndarray): Reference image.

    Returns:
        numpy.ndarray: Warped image.
    """
    coords_y, coords_x = np.indices(f2_level.shape[:2])
    coords_x_warp = coords_x + v
    coords_y_warp = coords_y + u
    coords_x_warp[coords_x_warp < 0] = 0
    coords_y_warp[coords_y_warp < 0] = 0
    coords_x_warp[coords_x_warp >= f2_level.shape[1]] = f2_level.shape[1] - 1
    coords_y_warp[coords_y_warp >= f2_level.shape[0]] = f2_level.shape[0] - 1
    coords_x_warp = coords_x_warp.astype(int)
    coords_y_warp = coords_y_warp.astype(int)
    if f2_level.ndim == 3:
        tmp = np.zeros_like(f2_level)
        for c in range(f2_level.shape[2]):
            tmp[:, :, c] = f2_level[coords_y_warp, coords_x_warp, c]
    else:
        tmp = f2_level[coords_y_warp, coords_x_warp]
    return tmp


def get_motion_tensor_gc(f1, f2, hx, hy):
    """
    Compute the motion tensor for gray value constancy assumption.

    Args:
        f1 (numpy.ndarray): First image (reference).
        f2 (numpy.ndarray): Second image (moving).
        hx (float): Spatial resolution in x-direction.
        hy (float): Spatial resolution in y-direction.

    Returns:
        tuple: (J11, J22, J33, J12, J13, J23) motion tensor components.
    """
    f1p = np.pad(f1, ((1, 1), (1, 1)), mode='symmetric')
    f2p = np.pad(f2, ((1, 1), (1, 1)), mode='symmetric')
    fx1, fy1 = np.gradient(f1p, hy, hx)
    fx2, fy2 = np.gradient(f2p, hy, hx)
    fx = 0.5 * (fx1 + fx2)
    ft = f2p - f1p
    fx = np.pad(fx[1:-1, 1:-1], ((1, 1), (1, 1)), mode='symmetric')
    ft = np.pad(ft[1:-1, 1:-1], ((1, 1), (1, 1)), mode='symmetric')
    _, fxy = np.gradient(fx, hy, hx)
    fxt, fyt = np.gradient(ft, hy, hx)

    def gradient2(f, hx_, hy_):
        fxx = np.zeros_like(f)
        fyy = np.zeros_like(f)
        fxx[1:-1, 1:-1] = (f[1:-1, 0:-2] - 2 * f[1:-1, 1:-1] + f[1:-1, 2:]) / hx_**2
        fyy[1:-1, 1:-1] = (f[0:-2, 1:-1] - 2 * f[1:-1, 1:-1] + f[2:, 1:-1]) / hy_**2
        return fxx, fyy

    fxx1, fyy1 = gradient2(f1p, hx, hy)
    fxx2, fyy2 = gradient2(f2p, hx, hy)
    fxx = 0.5 * (fxx1 + fxx2)
    fyy = 0.5 * (fyy1 + fyy2)
    reg_x = 1. / ((np.sqrt(fxx**2 + fxy**2)**2) + 0.000001)
    reg_y = 1. / ((np.sqrt(fxy**2 + fyy**2)**2) + 0.000001)
    J11 = reg_x * fxx**2 + reg_y * fxy**2
    J22 = reg_x * fxy**2 + reg_y * fyy**2
    J33 = reg_x * fxt**2 + reg_y * fyt**2
    J12 = reg_x * fxx * fxy + reg_y * fxy * fyy
    J13 = reg_x * fxx * fxt + reg_y * fxy * fyt
    J23 = reg_x * fxy * fxt + reg_y * fyy * fyt
    J11[:, 0] = 0; J11[:, -1] = 0; J11[0, :] = 0; J11[-1, :] = 0
    J22[:, 0] = 0; J22[:, -1] = 0; J22[0, :] = 0; J22[-1, :] = 0
    J33[:, 0] = 0; J33[:, -1] = 0; J33[0, :] = 0; J33[-1, :] = 0
    J12[:, 0] = 0; J12[:, -1] = 0; J12[0, :] = 0; J12[-1, :] = 0
    J13[:, 0] = 0; J13[:, -1] = 0; J13[0, :] = 0; J13[-1, :] = 0
    J23[:, 0] = 0; J23[:, -1] = 0; J23[0, :] = 0; J23[-1, :] = 0
    return J11[1:-1, 1:-1], J22[1:-1, 1:-1], J33[1:-1, 1:-1], J12[1:-1, 1:-1], J13[1:-1, 1:-1], J23[1:-1, 1:-1]


def level_solver(J11, J22, J33, J12, J13, J23, weight, u, v, alpha, iterations, update_lag, verbose, a_data, a_smooth, hx, hy, compute_flow_func):
    """
    Solve the optical flow problem at a given pyramid level.

    Args:
        J11, J22, J33, J12, J13, J23 (numpy.ndarray): Motion tensor components.
        weight (numpy.ndarray): Weight array.
        u (numpy.ndarray): Initial displacement field in y-direction with boundaries.
        v (numpy.ndarray): Initial displacement field in x-direction with boundaries.
        alpha (tuple): Smoothness parameters.
        iterations (int): Number of iterations for the solver.
        update_lag (int): Interval at which to update non-linearities.
        verbose (int): Verbosity level.
        a_data (numpy.ndarray): Data exponent array.
        a_smooth (float): Smoothness exponent.
        hx (float): Spatial resolution in x-direction.
        hy (float): Spatial resolution in y-direction.
        compute_flow_func (callable): C++ function to compute flow.

    Returns:
        tuple: (du, dv) displacement increments.
    """
    result = compute_flow_func(J11, J22, J33, J12, J13, J23, weight, u[1:-1, 1:-1], v[1:-1, 1:-1],
                               alpha[0], alpha[1], iterations, update_lag, a_data, a_smooth, hx, hy)
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    du[1:-1, 1:-1] = result[:, :, 0]
    dv[1:-1, 1:-1] = result[:, :, 1]
    return du, dv


def get_displacement(fixed, moving, alpha=(2, 2), update_lag=10, iterations=20, min_level=0, levels=50, eta=0.75,
                     a_smooth=0.5, a_data=0.45, const_assumption='gc', weight=None, compute_flow_func=None):
    """
    Compute the displacement field between a fixed and a moving image using a coarse-to-fine approach.

    Args:
        fixed (numpy.ndarray): Fixed reference image.
        moving (numpy.ndarray): Moving image to be registered.
        alpha (tuple): Smoothness parameters.
        update_lag (int): Interval at which to update non-linearities.
        iterations (int): Number of iterations at each pyramid level.
        min_level (int): Minimum level of the pyramid.
        levels (int): Maximum number of levels in the pyramid.
        eta (float): Scale factor per level in the pyramid.
        a_smooth (float): Smoothness exponent.
        a_data (float or numpy.ndarray): Data exponent parameter or array for multiple channels.
        const_assumption (str): Constancy assumption type ('gc', etc.).
        weight (numpy.ndarray or None): Optional weight array.
        compute_flow_func (callable or None): Function to compute flow at the core (C++ extension).

    Returns:
        numpy.ndarray: Displacement field of shape (m, n, 2).
    """
    fixed = fixed.astype(np.float64)
    moving = moving.astype(np.float64)

    if fixed.ndim == 3:
        m, n, n_channels = fixed.shape
    else:
        m, n = fixed.shape
        n_channels = 1
        fixed = fixed[:, :, None]
        moving = moving[:, :, None]

    u_init = np.zeros((m, n), dtype=np.float64)
    v_init = np.zeros((m, n), dtype=np.float64)

    if weight is None:
        weight = np.ones((m, n, n_channels), dtype=np.float64) / n_channels
    else:
        weight = weight.astype(np.float64)

    if not isinstance(a_data, np.ndarray):
        a_data_arr = np.full(n_channels, a_data, dtype=np.float64)
    else:
        a_data_arr = a_data

    f1_low = fixed
    f2_low = moving

    method = 'bicubic'
    max_level_y = warpingDepth(eta, levels, m, m)
    max_level_x = warpingDepth(eta, levels, n, n)
    max_level = min(max_level_x, max_level_y) * 4

    if max(max_level_x, max_level_y) <= min_level:
        min_level = max(max_level_x, max_level_y) - 1
    if min_level < 0:
        min_level = 0

    for i in range(max(max_level_x, max_level_y), min_level - 1, -1):
        level_size = (int(round(m * eta**(min(i, max_level_y)))), int(round(n * eta**(min(i, max_level_x)))))

        f1_level = resize(f1_low, level_size + (n_channels,), order=3, mode='edge', anti_aliasing=True)
        f2_level = resize(f2_low, level_size + (n_channels,), order=3, mode='edge', anti_aliasing=True)

        hx = m / f1_level.shape[0]
        hy = n / f1_level.shape[1]

        if i == max(max_level_x, max_level_y):
            u = add_boundary(resize(u_init, level_size, order=3, mode='edge', anti_aliasing=True))
            v = add_boundary(resize(v_init, level_size, order=3, mode='edge', anti_aliasing=True))
            tmp = f2_level.copy()
        else:
            u = add_boundary(resize(u[1:-1, 1:-1], level_size, order=3, mode='edge', anti_aliasing=True))
            v = add_boundary(resize(v[1:-1, 1:-1], level_size, order=3, mode='edge', anti_aliasing=True))
            tmp = imregister_wrapper(f2_level, u[1:-1, 1:-1]/hx, v[1:-1, 1:-1]/hy, f1_level)

        J11 = np.zeros(f1_level.shape, dtype=np.float64)
        J22 = np.zeros(f1_level.shape, dtype=np.float64)
        J33 = np.zeros(f1_level.shape, dtype=np.float64)
        J12 = np.zeros(f1_level.shape, dtype=np.float64)
        J13 = np.zeros(f1_level.shape, dtype=np.float64)
        J23 = np.zeros(f1_level.shape, dtype=np.float64)

        for ch in range(n_channels):
            (J11_ch, J22_ch, J33_ch, J12_ch, J13_ch, J23_ch) = get_motion_tensor_gc(f1_level[:, :, ch],
                                                                                   tmp[:, :, ch],
                                                                                   hx, hy)
            J11[:, :, ch] = J11_ch
            J22[:, :, ch] = J22_ch
            J33[:, :, ch] = J33_ch
            J12[:, :, ch] = J12_ch
            J13[:, :, ch] = J13_ch
            J23[:, :, ch] = J23_ch

        weight_level = resize(weight, (f1_level.shape[0], f1_level.shape[1], n_channels), order=3, mode='edge', anti_aliasing=True)

        du, dv = level_solver(J11, J22, J33, J12, J13, J23, weight_level, u, v, alpha, iterations, update_lag, 0,
                              a_data_arr, a_smooth, hx, hy, compute_flow_func)

        if min(level_size) > 5:
            du[1:-1, 1:-1] = median_filter(du[1:-1, 1:-1], size=(5, 5), mode='reflect')
            dv[1:-1, 1:-1] = median_filter(dv[1:-1, 1:-1], size=(5, 5), mode='reflect')

        u = u + du
        v = v + dv

    w = np.zeros((u.shape[0]-2, u.shape[1]-2, 2), dtype=np.float64)
    w[:, :, 0] = u[1:-1, 1:-1]
    w[:, :, 1] = v[1:-1, 1:-1]

    if min_level > 0:
        w = resize(w, (m, n, 2), order=3, mode='edge', anti_aliasing=True)

    return w
