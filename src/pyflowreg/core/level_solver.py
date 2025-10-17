"""
Pyramid Level Solver for Variational Optical Flow
==================================================

This module implements the computationally intensive solver for optical flow
estimation at each pyramid level. All functions are numba-optimized with JIT
compilation for high performance.

The solver uses successive over-relaxation (SOR) to iteratively solve the
linearized optical flow equations with non-linear diffusion regularization.
The implementation follows the generalized Charbonnier penalty framework for
robustness to noise and motion discontinuities.

Functions
---------
compute_flow
    Main iterative solver for flow field computation at a pyramid level
set_boundary_2d
    Apply Neumann boundary conditions to 2D arrays
nonlinearity_smoothness_2d
    Compute non-linearity weights for smoothness term

Notes
-----
All functions use numba JIT compilation with cache=True for performance.
The cache is beneficial since these functions are called repeatedly for each
image, but it means the functions will fail if dtypes change between calls.

For optimal performance, input arrays should be contiguous in memory. Numba
performs significantly better with contiguous arrays and can be slow with
non-contiguous data due to inability to use optimized SIMD instructions.

The solver implements coupled updates for the u and v components of the flow
field using SOR with OMEGA=1.95.

See Also
--------
pyflowreg.core.optical_flow : High-level optical flow computation
"""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def set_boundary_2d(f):
    """
    Apply Neumann boundary conditions to 2D array in-place.

    Replicates edge values to provide zero-derivative (Neumann) boundary
    conditions for the flow solver. This enables one-sided finite differences
    at boundaries (forward differences at left/top edges, backward differences
    at right/bottom edges) without going out of bounds.

    Parameters
    ----------
    f : ndarray
        2D array to apply boundary conditions to (modified in-place).
    """
    m, n = f.shape
    for i in range(n):
        f[0, i] = f[1, i]
        f[m - 1, i] = f[m - 2, i]
    for j in range(m):
        f[j, 0] = f[j, 1]
        f[j, n - 1] = f[j, n - 2]


@njit(fastmath=True, cache=True)
def nonlinearity_smoothness_2d(psi_smooth, u, du, v, dv, m, n, a, hx, hy):
    """
    Compute derivative of smoothness penalty (non-linearity weights).

    Calculates ψ(s²) = a(s² + ε)^(a-1), the derivative of the Charbonnier
    smoothness penalty ρ(s²) = (s² + ε)^a, where s² = ||∇u||² + ||∇v||².
    This non-linearity arises from the variational derivative of the energy
    functional.

    Parameters
    ----------
    psi_smooth : ndarray
        Output array for smoothness weights (m, n), modified in-place.
    u : ndarray
        Current u-component of flow field (m, n).
    du : ndarray
        Incremental update to u-component (m, n).
    v : ndarray
        Current v-component of flow field (m, n).
    dv : ndarray
        Incremental update to v-component (m, n).
    m : int
        Height of arrays.
    n : int
        Width of arrays.
    a : float
        Charbonnier penalty exponent for smoothness term.
    hx : float
        Spatial grid spacing in x-direction.
    hy : float
        Spatial grid spacing in y-direction.
    """
    eps = 0.00001
    u_full = u + du
    v_full = v + dv
    ux = np.zeros((m, n))
    uy = np.zeros((m, n))
    vx = np.zeros((m, n))
    vy = np.zeros((m, n))

    for i in range(n):
        for j in range(m):
            # ux
            if n > 1:
                if i == 0:
                    ux[j, i] = (u_full[j, i + 1] - u_full[j, i]) / hx
                elif i == n - 1:
                    ux[j, i] = (u_full[j, i] - u_full[j, i - 1]) / hx
                else:
                    ux[j, i] = (u_full[j, i + 1] - u_full[j, i - 1]) / (2.0 * hx)
            # vx
            if n > 1:
                if i == 0:
                    vx[j, i] = (v_full[j, i + 1] - v_full[j, i]) / hx
                elif i == n - 1:
                    vx[j, i] = (v_full[j, i] - v_full[j, i - 1]) / hx
                else:
                    vx[j, i] = (v_full[j, i + 1] - v_full[j, i - 1]) / (2.0 * hx)
            # uy
            if m > 1:
                if j == 0:
                    uy[j, i] = (u_full[j + 1, i] - u_full[j, i]) / hy
                elif j == m - 1:
                    uy[j, i] = (u_full[j, i] - u_full[j - 1, i]) / hy
                else:
                    uy[j, i] = (u_full[j + 1, i] - u_full[j - 1, i]) / (2.0 * hy)
            # vy
            if m > 1:
                if j == 0:
                    vy[j, i] = (v_full[j + 1, i] - v_full[j, i]) / hy
                elif j == m - 1:
                    vy[j, i] = (v_full[j, i] - v_full[j - 1, i]) / hy
                else:
                    vy[j, i] = (v_full[j + 1, i] - v_full[j - 1, i]) / (2.0 * hy)

    for i in range(n):
        for j in range(m):
            tmp = (
                ux[j, i] * ux[j, i]
                + uy[j, i] * uy[j, i]
                + vx[j, i] * vx[j, i]
                + vy[j, i] * vy[j, i]
            )
            if tmp < 0.0:
                tmp = 0.0
            psi_smooth[j, i] = a * (tmp + eps) ** (a - 1.0)


@njit(fastmath=True, cache=True)
def compute_flow(
    J11,
    J22,
    J33,
    J12,
    J13,
    J23,
    weight,
    u,
    v,
    alpha_x,
    alpha_y,
    iterations,
    update_lag,
    a_data,
    a_smooth,
    hx,
    hy,
):
    """
    Iterative solver for optical flow at a single pyramid level.

    Solves the linearized optical flow equations using successive over-relaxation
    (SOR) with non-linear diffusion regularization. The solver minimizes an energy
    functional combining data fidelity (from motion tensor) and smoothness terms,
    both with generalized Charbonnier penalties.

    Parameters
    ----------
    J11, J22, J33, J12, J13, J23 : ndarray
        Motion tensor components, shape (m, n, n_channels). Encode the linearized
        gradient constancy constraints.
    weight : ndarray
        Channel weights for multi-channel data, shape (m, n, n_channels).
    u, v : ndarray
        Initial flow field components from coarser pyramid level, shape (m, n).
    alpha_x, alpha_y : float
        Regularization weights for smoothness term in x and y directions.
    iterations : int
        Number of SOR iterations to perform.
    update_lag : int
        Update non-linearity weights every update_lag iterations.
    a_data : ndarray
        Charbonnier exponents for data term, length n_channels.
    a_smooth : float
        Charbonnier exponent for smoothness term.
    hx, hy : float
        Spatial grid spacing in x and y directions.

    Returns
    -------
    flow : ndarray
        Computed flow field, shape (m, n, 2) where flow[:,:,0] is u-component
        and flow[:,:,1] is v-component.

    Notes
    -----
    Uses SOR with relaxation parameter OMEGA=1.95 for convergence acceleration.
    The solver updates flow increments (du, dv) relative to the input (u, v)
    using a coupled Gauss-Seidel scheme with immediate u-updates affecting v.
    """
    m, n, n_channels = J11.shape
    du = np.zeros((m, n))
    dv = np.zeros((m, n))
    psi = np.ones((m, n, n_channels))
    psi_smooth = np.ones((m, n))

    OMEGA = 1.95
    alpha = np.array([alpha_x, alpha_y], dtype=np.float64)

    for iteration_counter in range(iterations):
        if (iteration_counter + 1) % update_lag == 0:
            # Update psi (non-linearities for data term)
            for k in range(n_channels):
                for i in range(n):
                    for j in range(m):
                        val = (
                            J11[j, i, k] * du[j, i] * du[j, i]
                            + J22[j, i, k] * dv[j, i] * dv[j, i]
                            + J23[j, i, k] * dv[j, i]
                            + 2.0 * J12[j, i, k] * du[j, i] * dv[j, i]
                            + 2.0 * J13[j, i, k] * du[j, i]
                            + J23[j, i, k] * dv[j, i]
                            + J33[j, i, k]
                        )
                        if val < 0.0:
                            val = 0.0
                        psi[j, i, k] = a_data[k] * (val + 0.00001) ** (a_data[k] - 1.0)

            if a_smooth != 1.0:
                nonlinearity_smoothness_2d(
                    psi_smooth, u, du, v, dv, m, n, a_smooth, hx, hy
                )
            else:
                for i in range(n):
                    for j in range(m):
                        psi_smooth[j, i] = 1.0

        set_boundary_2d(du)
        set_boundary_2d(dv)

        for i in range(1, n - 1):
            for j in range(1, m - 1):
                denom_u = 0.0
                denom_v = 0.0
                num_u = 0.0
                num_v = 0.0

                # neighbors:
                # left = (j, i-1)
                # right = (j, i+1)
                # down = (j+1, i)
                # up = (j-1, i)
                left = (j, i - 1)
                right = (j, i + 1)
                down = (j + 1, i)
                up = (j - 1, i)

                if a_smooth != 1.0:
                    tmp = (
                        0.5
                        * (psi_smooth[j, i] + psi_smooth[left])
                        * (alpha[0] / (hx * hx))
                    )
                    num_u += tmp * (u[left] + du[left] - u[j, i])
                    num_v += tmp * (v[left] + dv[left] - v[j, i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = (
                        0.5
                        * (psi_smooth[j, i] + psi_smooth[right])
                        * (alpha[0] / (hx * hx))
                    )
                    num_u += tmp * (u[right] + du[right] - u[j, i])
                    num_v += tmp * (v[right] + dv[right] - v[j, i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = (
                        0.5
                        * (psi_smooth[j, i] + psi_smooth[down])
                        * (alpha[1] / (hy * hy))
                    )
                    num_u += tmp * (u[down] + du[down] - u[j, i])
                    num_v += tmp * (v[down] + dv[down] - v[j, i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = (
                        0.5
                        * (psi_smooth[j, i] + psi_smooth[up])
                        * (alpha[1] / (hy * hy))
                    )
                    num_u += tmp * (u[up] + du[up] - u[j, i])
                    num_v += tmp * (v[up] + dv[up] - v[j, i])
                    denom_u += tmp
                    denom_v += tmp
                else:
                    tmp = alpha[0] / (hx * hx)
                    num_u += tmp * (u[left] + du[left] - u[j, i])
                    num_v += tmp * (v[left] + dv[left] - v[j, i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = alpha[0] / (hx * hx)
                    num_u += tmp * (u[right] + du[right] - u[j, i])
                    num_v += tmp * (v[right] + dv[right] - v[j, i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = alpha[1] / (hy * hy)
                    num_u += tmp * (u[down] + du[down] - u[j, i])
                    num_v += tmp * (v[down] + dv[down] - v[j, i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = alpha[1] / (hy * hy)
                    num_u += tmp * (u[up] + du[up] - u[j, i])
                    num_v += tmp * (v[up] + dv[up] - v[j, i])
                    denom_u += tmp
                    denom_v += tmp

                for k in range(n_channels):
                    val_u = (
                        weight[j, i, k]
                        * psi[j, i, k]
                        * (J13[j, i, k] + J12[j, i, k] * dv[j, i])
                    )
                    num_u -= val_u
                    denom_u += weight[j, i, k] * psi[j, i, k] * J11[j, i, k]
                    denom_v += weight[j, i, k] * psi[j, i, k] * J22[j, i, k]

                du_kp1 = num_u / denom_u if denom_u != 0.0 else 0.0
                du[j, i] = (1.0 - OMEGA) * du[j, i] + OMEGA * du_kp1

                num_v2 = num_v
                for k in range(n_channels):
                    num_v2 -= (
                        weight[j, i, k]
                        * psi[j, i, k]
                        * (J23[j, i, k] + J12[j, i, k] * du[j, i])
                    )

                dv_kp1 = num_v2 / denom_v if denom_v != 0.0 else 0.0
                dv[j, i] = (1.0 - OMEGA) * dv[j, i] + OMEGA * dv_kp1

    flow = np.zeros((m, n, 2))
    flow[:, :, 0] = du
    flow[:, :, 1] = dv
    return flow
