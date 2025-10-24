"""
Optical Flow Implementation
============================

This module implements the variational optical flow computation using
a pyramid-based multi-scale approach with gradient constancy assumption.

This is the default 'flowreg' backend implementation that maintains
compatibility with the MATLAB Flow-Registration toolbox.

Functions
---------
get_displacement
    Main API function for computing optical flow between two frames
get_motion_tensor_gc
    Compute motion tensor components for gradient constancy
imregister_wrapper
    Warp an image using computed displacement fields
warpingDepth
    Calculate pyramid depth based on image dimensions
add_boundary
    Add boundary padding to arrays

Notes
-----
The implementation uses a coarse-to-fine pyramid scheme where flow is
computed at progressively finer resolutions, with each level initialized
from the previous coarser level.

See Also
--------
pyflowreg.core.level_solver.compute_flow : Low-level flow solver
"""

import cv2
import numpy as np
from scipy.ndimage import median_filter

from pyflowreg.core import compute_flow
from pyflowreg.core.warping import imregister_wrapper, warpingDepth
from pyflowreg.util.resize_util import imresize_fused_gauss_cubic as resize


def add_boundary(f):
    """
    Add 1-pixel boundary padding with edge replication.

    Pads array with 1 pixel on all sides by replicating edge values,
    implementing Neumann boundary conditions for the flow solver.

    Parameters
    ----------
    f : np.ndarray
        Input array to pad

    Returns
    -------
    padded : np.ndarray
        Padded array with shape increased by 2 in each dimension

    Notes
    -----
    Uses np.pad with mode='edge', which replicates the values at the edges
    to provide Neumann (zero-derivative) boundary conditions.
    """
    return np.pad(f, 1, mode="edge")


def get_motion_tensor_gc(f1, f2, hy, hx):
    """
    Compute motion tensor components for gradient constancy assumption.

    Calculates the motion tensor components that describe the linearized
    optical flow constraints under the gradient constancy assumption, where
    image gradients are assumed to remain constant between frames. The motion
    tensor encodes motion through temporal derivatives and frame-averaged
    spatial derivatives between f1 and f2.

    Parameters
    ----------
    f1 : ndarray
        Reference frame (2D array).
    f2 : ndarray
        Moving frame (2D array).
    hy : float
        Spatial grid spacing in y-direction.
    hx : float
        Spatial grid spacing in x-direction.

    Returns
    -------
    J11, J22, J33, J12, J13, J23 : ndarray
        Motion tensor components used in the optical flow solver. These
        components encode the linearized gradient constancy constraints
        with adaptive regularization based on local gradient structure.

    Notes
    -----
    The gradient constancy assumption extends the classic brightness constancy
    by requiring that spatial gradients also remain constant during motion.
    This provides additional robustness in textured regions.

    The tensor components are computed with adaptive regularization that
    weights contributions based on local gradient magnitudes, making the
    estimation robust to noise while preserving motion details.

    References
    ----------
    .. [2] Brox, T., Bruhn, A., Papenberg, N., and Weickert, J. "High
       Accuracy Optical Flow Estimation Based on a Theory for Warping",
       ECCV 2004.
    .. [3] Flotho, P., et al. "Software for Non-Parametric Image
       Registration of 2-Photon Imaging Data", J. Biophotonics, 2022.
    """
    f1p = np.pad(f1, ((1, 1), (1, 1)), mode="symmetric")
    f2p = np.pad(f2, ((1, 1), (1, 1)), mode="symmetric")
    _, fx1p = np.gradient(f1p, hy, hx)
    _, fx2p = np.gradient(f2p, hy, hx)
    fx = 0.5 * (fx1p + fx2p)
    ft = f2p - f1p
    fx = np.pad(fx[1:-1, 1:-1], 1, mode="symmetric")
    ft = np.pad(ft[1:-1, 1:-1], 1, mode="symmetric")

    tmp_grad = np.gradient(fx, hy, hx)
    fxy = tmp_grad[0]
    ft_grad = np.gradient(ft, hy, hx)
    fxt = ft_grad[1]
    fyt = ft_grad[0]

    def gradient2(f, hx_, hy_):
        fxx = np.zeros_like(f)
        fyy = np.zeros_like(f)
        fxx[1:-1, 1:-1] = (f[1:-1, 0:-2] - 2 * f[1:-1, 1:-1] + f[1:-1, 2:]) / (hx_**2)
        fyy[1:-1, 1:-1] = (f[0:-2, 1:-1] - 2 * f[1:-1, 1:-1] + f[2:, 1:-1]) / (hy_**2)
        return fxx, fyy

    fxx1, fyy1 = gradient2(f1p, hy, hx)
    fxx2, fyy2 = gradient2(f2p, hy, hx)
    fxx = 0.5 * (fxx1 + fxx2)
    fyy = 0.5 * (fyy1 + fyy2)
    reg_x = 1.0 / ((np.sqrt(fxx**2 + fxy**2) ** 2) + 1e-6)
    reg_y = 1.0 / ((np.sqrt(fxy**2 + fyy**2) ** 2) + 1e-6)
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


def level_solver(
    J11,
    J22,
    J33,
    J12,
    J13,
    J23,
    weight,
    u,
    v,
    alpha,
    iterations,
    update_lag,
    verbose,
    a_data,
    a_smooth,
    hx,
    hy,
):
    result = compute_flow(
        J11,
        J22,
        J33,
        J12,
        J13,
        J23,
        weight=weight,
        u=u,
        v=v,
        alpha_x=alpha[0],
        alpha_y=alpha[1],
        iterations=iterations,
        update_lag=update_lag,
        a_data=a_data,
        a_smooth=a_smooth,
        hx=hx,
        hy=hy,
    )
    du = result[:, :, 0]
    dv = result[:, :, 1]
    return du, dv


def get_displacement(
    fixed,
    moving,
    alpha=(2, 2),
    update_lag=5,
    iterations=50,
    min_level=0,
    levels=50,
    eta=0.8,
    a_smooth=1.0,
    a_data=0.45,
    const_assumption="gc",
    uv=None,
    weight=None,
    level_solver_backend=None,
):
    """
    Compute optical flow displacement field using variational approach.

    This function implements the main pyramid-based variational optical flow
    algorithm using gradient constancy assumption with non-linear diffusion
    regularization. Flow is computed from coarse to fine scales, with each
    level initialized from the upsampled result of the previous coarser level.

    Parameters
    ----------
    fixed : np.ndarray
        Reference (fixed) image, shape (H, W) or (H, W, C)
    moving : np.ndarray
        Moving image to register, shape (H, W) or (H, W, C)
    alpha : tuple of float, default=(2, 2)
        Regularization strength (alpha_x, alpha_y) controlling smoothness.
        Larger values enforce smoother flow fields.
    update_lag : int, default=5
        Number of iterations between updates of non-linearity weights (psi).
        Smaller values update more frequently (slower, potentially more accurate).
        Larger values update less frequently (faster convergence).
    iterations : int, default=50
        Number of SOR iterations per pyramid level
    min_level : int, default=0
        Minimum (finest) pyramid level to compute. 0 = full resolution.
    levels : int, default=50
        Maximum number of pyramid levels attempted
    eta : float, default=0.8
        Pyramid downsampling factor per level (0 < eta <= 1).
        Each level is eta times the size of the previous level.
    a_smooth : float, default=1.0
        Exponent for generalized Charbonnier penalty on smoothness term.
        Controls robustness of smoothness regularization via ρ(s²) = (s²)^a:
        - a = 1.0: quadratic (L2) penalty, assumes smooth flow everywhere
        - a = 0.5: linear (L1) penalty
        - 0.5 < a < 1.0: sublinear, robust to local discontinuities
    a_data : float, default=0.45
        Exponent for generalized Charbonnier penalty on data term.
        Controls robustness to noise and outliers via ρ(d²) = (d²)^a:
        - a = 1.0: quadratic (L2) penalty
        - a = 0.5: linear (L1) penalty
        - a = 0.45: sublinear, robust to noisy microscopy data
    const_assumption : str, default='gc'
        Constancy assumption: 'gc' for gradient constancy (only option implemented)
    uv : np.ndarray, optional
        Initial displacement field (H, W, 2) with [u, v] components to initialize
        the coarsest (highest) pyramid level. If None, initializes with zeros.
    weight : np.ndarray or list, optional
        Channel weights for multi-channel registration. Can be:
        - 1D array of length C: per-channel weights (normalized to sum to 1)
        - 2D array (H, W): spatial weights broadcast to all channels
        - 3D array (H, W, C): full spatial and channel weights
        If None, uses equal weights (1/C) for all channels.
    level_solver_backend : Callable, optional
        Custom level solver function to use instead of the default CPU solver.
        Used by GPU backends to inject accelerated solvers. Must have the same
        signature as the default level_solver function.

    Returns
    -------
    flow : np.ndarray
        Displacement field, shape (H, W, 2) where [:, :, 0] is horizontal (u)
        and [:, :, 1] is vertical (v) displacement in pixels

    Notes
    -----
    The pyramid depth is computed independently for each dimension (height and width),
    stopping when that dimension becomes < 10 pixels. This allows narrow ROIs to
    achieve large displacements along their longer dimension without being limited
    by the shorter dimension.

    The algorithm uses successive over-relaxation (SOR) with omega=1.95 for
    fast convergence at each pyramid level.

    When a_smooth=1.0 (quadratic), the smoothness penalty computation is
    optimized by skipping weight updates (psi_smooth = 1.0).

    This implementation maintains compatibility with MATLAB Flow-Registration.

    See Also
    --------
    pyflowreg.motion_correction.compensate_arr : High-level API with OFOptions
    get_motion_tensor_gc : Compute motion tensor for gradient constancy

    References
    ----------
    .. [4] Flotho et al. "Software for Non-Parametric Image Registration of
       2-Photon Imaging Data", J Biophotonics, 2022.
       https://doi.org/10.1002/jbio.202100330
    """
    # Ensure fixed and moving have the same number of dimensions
    assert (
        fixed.ndim == moving.ndim
    ), f"Fixed and moving must have same dimensions: fixed.shape={fixed.shape}, moving.shape={moving.shape}"
    fixed = fixed.astype(np.float64)
    moving = moving.astype(np.float64)
    if fixed.ndim == 3:
        m, n, n_channels = fixed.shape
    else:
        m, n = fixed.shape
        n_channels = 1
        fixed = fixed[:, :, np.newaxis]
        moving = moving[:, :, np.newaxis]
    if uv is not None:
        u_init = uv[:, :, 0]
        v_init = uv[:, :, 1]
    else:
        u_init = np.zeros((m, n), dtype=np.float64)
        v_init = np.zeros((m, n), dtype=np.float64)
    if weight is None:
        weight = np.ones((m, n, n_channels), dtype=np.float64) / n_channels
    else:
        weight = weight.astype(np.float64)
        if weight.ndim < 3:
            # Handle 1D weight array
            if weight.ndim == 1:
                # If weight has fewer elements than channels, pad with 1/n_channels
                if len(weight) < n_channels:
                    # Use default value for missing channels (MATLAB behavior)
                    default_weight = 1.0 / n_channels
                    weight_expanded = np.full(
                        n_channels, default_weight, dtype=np.float64
                    )
                    weight_expanded[: len(weight)] = weight
                    weight = weight_expanded
                elif len(weight) > n_channels:
                    # Truncate if more weights than channels
                    weight = weight[:n_channels]
                # Normalize weights to sum to 1
                weight = weight / weight.sum()
                # Broadcast to spatial dimensions
                weight = np.ones((m, n, n_channels), dtype=np.float64) * weight.reshape(
                    1, 1, -1
                )
            else:
                # 2D spatial weight - broadcast to all channels
                weight = (
                    np.ones((m, n, n_channels), dtype=np.float64)
                    * weight[..., np.newaxis]
                )
    if not isinstance(a_data, np.ndarray):
        a_data_arr = np.full(n_channels, a_data, dtype=np.float64)
    else:
        a_data_arr = a_data
    a_data_arr = np.ascontiguousarray(a_data_arr)
    f1_low = fixed
    f2_low = moving
    max_level_y = warpingDepth(eta, levels, m, m)
    max_level_x = warpingDepth(eta, levels, n, n)
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
        level_size = (
            int(round(m * eta ** (min(i, max_level_y)))),
            int(round(n * eta ** (min(i, max_level_x)))),
        )
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
            tmp = imregister_wrapper(
                f2_level,
                u[1:-1, 1:-1] / current_hy,
                v[1:-1, 1:-1] / current_hx,
                f1_level,
            )
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
            J11_ch, J22_ch, J33_ch, J12_ch, J13_ch, J23_ch = get_motion_tensor_gc(
                f1_level[:, :, ch], tmp[:, :, ch], current_hx, current_hy
            )
            J11[:, :, ch] = J11_ch
            J22[:, :, ch] = J22_ch
            J33[:, :, ch] = J33_ch
            J12[:, :, ch] = J12_ch
            J13[:, :, ch] = J13_ch
            J23[:, :, ch] = J23_ch

        weight_level = resize(weight, f1_level.shape[:2])
        weight_level = cv2.copyMakeBorder(
            weight_level, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0.0
        )
        if weight_level.ndim < 3:
            weight_level = weight_level[:, :, np.newaxis]

        if i == min_level:
            alpha_scaling = 1
        else:
            alpha_scaling = eta ** (-0.5 * i)

        alpha_tmp = [alpha_scaling * alpha[j] for j in range(len(alpha))]

        # Use custom level solver if provided, otherwise use default
        solver_func = (
            level_solver_backend if level_solver_backend is not None else level_solver
        )

        du, dv = solver_func(
            np.ascontiguousarray(J11),
            np.ascontiguousarray(J22),
            np.ascontiguousarray(J33),
            np.ascontiguousarray(J12),
            np.ascontiguousarray(J13),
            np.ascontiguousarray(J23),
            np.ascontiguousarray(weight_level),
            u,
            v,
            alpha_tmp,
            iterations,
            update_lag,
            0,
            a_data_arr,
            a_smooth,
            current_hx,
            current_hy,
        )
        if min(level_size) > 5:
            du[1:-1, 1:-1] = median_filter(du[1:-1, 1:-1], size=(5, 5), mode="mirror")
            dv[1:-1, 1:-1] = median_filter(dv[1:-1, 1:-1], size=(5, 5), mode="mirror")
        u = u + du
        v = v + dv
    w = np.zeros((u.shape[0] - 2, u.shape[1] - 2, 2), dtype=np.float64)
    w[:, :, 0] = u[1:-1, 1:-1]
    w[:, :, 1] = v[1:-1, 1:-1]
    if min_level > 0:
        w = cv2.resize(w, (n, m), interpolation=cv2.INTER_CUBIC)
    return w
