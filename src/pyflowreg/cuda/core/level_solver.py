"""
CUDA-accelerated Red-Black Gauss-Seidel solver for variational optical flow.

This module implements the level solver using CuPy and raw CUDA kernels for
GPU acceleration. It is functionally equivalent to the torch.level_solver
implementation but uses explicit CUDA kernel code for maximum performance.
"""

import numpy as np
import cupy as cp


def level_solver_rbgs_cuda(
    J11,
    J22,
    J33,
    J12,
    J13,
    J23,
    weight_level,
    u,
    v,
    alpha,
    iterations,
    update_lag,
    a_data,
    a_smooth,
    hx,
    hy,
    omega=1.95,
    eps=1e-6,
    update_lag_semantics="torch",
):
    """
    Solve for flow increments using Red-Black Gauss-Seidel relaxation on GPU.

    Implements the variational optical flow solver with non-linear diffusion
    regularization using a Red-Black Gauss-Seidel scheme. This CUDA implementation
    uses explicit kernels for boundary handling, psi computation, and relaxation.

    The solver minimizes the energy functional:
        E = E_data + alpha * E_smooth
    where E_data uses the gradient constancy assumption and E_smooth is a
    non-linear diffusion regularizer.

    Parameters
    ----------
    J11 : ndarray, shape (m, n, K)
        Structure tensor component: ∇I_x · ∇I_x (summed over frames if multichannel).
    J22 : ndarray, shape (m, n, K)
        Structure tensor component: ∇I_y · ∇I_y.
    J33 : ndarray, shape (m, n, K)
        Structure tensor component: ∇I_t · ∇I_t.
    J12 : ndarray, shape (m, n, K)
        Structure tensor component: ∇I_x · ∇I_y.
    J13 : ndarray, shape (m, n, K)
        Structure tensor component: ∇I_x · ∇I_t.
    J23 : ndarray, shape (m, n, K)
        Structure tensor component: ∇I_y · ∇I_t.
    weight_level : ndarray, shape (K,) or (m, n, K)
        Per-channel weights. Automatically broadcast to (m, n, K) if needed.
        For multi-channel data, weights should sum to 1 across channels.
    u : ndarray, shape (m, n)
        Initial horizontal flow field (from previous pyramid level).
    v : ndarray, shape (m, n)
        Initial vertical flow field (from previous pyramid level).
    alpha : tuple of float, length 2
        Regularization weights [alpha_x, alpha_y] for smoothness term.
    iterations : int
        Number of RBGS iterations to perform.
    update_lag : int
        Number of iterations between updates of psi_data and psi_smooth.
        Higher values improve speed but may reduce convergence quality.
    a_data : float or ndarray, shape (K,)
        Exponent for data term robustness: psi_data = a * E^(a-1).
        Common values: 1.0 (quadratic), 0.5 (robust L1-like).
    a_smooth : float
        Exponent for smoothness term robustness: psi_smooth = a * |∇u|^(a-1).
        Use 1.0 for isotropic diffusion, <1.0 for edge-preserving anisotropic diffusion.
    hx : float
        Spatial grid spacing in x-direction (typically 1.0).
    hy : float
        Spatial grid spacing in y-direction (typically 1.0).
    omega : float, optional
        SOR (Successive Over-Relaxation) parameter, default 1.95.
        Values in (1, 2) accelerate convergence; 1.0 gives standard Gauss-Seidel.
    eps : float, optional
        Small epsilon for numerical stability, default 1e-6.
    update_lag_semantics : {'torch', 'matlab'}, optional
        Semantics for update_lag timing, default 'torch'.
        - 'torch': update when (iteration % update_lag) == 0
        - 'matlab': update when ((iteration + 1) % update_lag) == 0

    Returns
    -------
    du : ndarray, shape (m, n)
        Horizontal flow increment computed at this level.
    dv : ndarray, shape (m, n)
        Vertical flow increment computed at this level.

    Notes
    -----
    - Uses Neumann boundary conditions (zero derivative at boundaries)
    - Red-Black ordering enables parallel updates within each color
    - CUDA kernels process pixels with (i+j) % 2 == parity for red/black split
    - All computations performed in float64 for numerical accuracy
    - Arrays automatically transferred to/from GPU

    See Also
    --------
    pyflowreg.core.level_solver : CPU implementation (numba)
    pyflowreg.core.torch.level_solver : PyTorch implementation

    References
    ----------
    Based on the Flow-Registration toolbox:
    Flotho et al. (2023), Journal of Biophotonics, DOI: 10.1002/jbio.202100330
    """
    m, n, K = J11.shape
    cp_dtype = cp.float64
    J11 = cp.asarray(J11, dtype=cp_dtype, order="C")
    J22 = cp.asarray(J22, dtype=cp_dtype, order="C")
    J33 = cp.asarray(J33, dtype=cp_dtype, order="C")
    J12 = cp.asarray(J12, dtype=cp_dtype, order="C")
    J13 = cp.asarray(J13, dtype=cp_dtype, order="C")
    J23 = cp.asarray(J23, dtype=cp_dtype, order="C")
    if weight_level.ndim == 3 and weight_level.shape == (m, n, K):
        W = cp.asarray(weight_level, dtype=cp_dtype, order="C")
    else:
        W = cp.broadcast_to(
            cp.asarray(weight_level, dtype=cp_dtype).reshape(1, 1, -1), (m, n, K)
        ).copy()

    u = cp.asarray(u, dtype=cp_dtype, order="C")
    v = cp.asarray(v, dtype=cp_dtype, order="C")
    du = cp.zeros((m, n), dtype=cp_dtype)
    dv = cp.zeros((m, n), dtype=cp_dtype)

    if isinstance(a_data, (float, int)):
        a_vec = cp.full((K,), float(a_data), dtype=cp_dtype)
    else:
        a_vec = cp.asarray(a_data, dtype=cp_dtype).reshape(K)

    psi_data = cp.ones((m, n, K), dtype=cp_dtype)
    psi_smooth = cp.ones((m, n), dtype=cp_dtype)
    denom_u = cp.zeros((m, n), dtype=cp_dtype)
    denom_v = cp.zeros((m, n), dtype=cp_dtype)

    ax = float(alpha[0]) / (hx * hx)
    ay = float(alpha[1]) / (hy * hy)

    mod = cp.RawModule(
        code=r"""
    extern "C" {

    __global__ void boundary_rows(double* A, int m, int n){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n){
            A[i] = A[n + i];
            A[(m-1)*n + i] = A[(m-2)*n + i];
        }
    }

    __global__ void boundary_cols(double* A, int m, int n){
        int j = blockDim.x * blockIdx.x + threadIdx.x;
        if (j < m){
            A[j*n + 0] = A[j*n + 1];
            A[j*n + (n-1)] = A[j*n + (n-2)];
        }
    }

    __global__ void compute_psi_data(
        const double* __restrict__ J11,
        const double* __restrict__ J22,
        const double* __restrict__ J33,
        const double* __restrict__ J12,
        const double* __restrict__ J13,
        const double* __restrict__ J23,
        const double* __restrict__ du,
        const double* __restrict__ dv,
        const double* __restrict__ a_vec,
        double eps,
        double* __restrict__ psi_data,
        int m, int n, int K)
    {
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (j >= m || i >= n) return;
        int idx2 = j * n + i;
        int base = idx2 * K;
        double duv = du[idx2];
        double dvv = dv[idx2];
        for (int k=0; k<K; ++k){
            int idx3 = base + k;
            double E = J11[idx3]*duv*duv + J22[idx3]*dvv*dvv
                     + 2.0*J12[idx3]*duv*dvv + 2.0*J13[idx3]*duv + 2.0*J23[idx3]*dvv
                     + J33[idx3];
            if (E < 0.0) E = 0.0;
            double a = a_vec[k];
            psi_data[idx3] = a * pow(E + eps, a - 1.0);
        }
    }

    __global__ void compute_denoms(
        const double* __restrict__ psi_data,
        const double* __restrict__ W,
        const double* __restrict__ J11,
        const double* __restrict__ J22,
        double* __restrict__ denom_u,
        double* __restrict__ denom_v,
        int m, int n, int K, double eps)
    {
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (j >= m || i >= n) return;
        int idx2 = j * n + i;
        int base = idx2 * K;
        double su = 0.0, sv = 0.0;
        for (int k=0; k<K; ++k){
            int idx3 = base + k;
            double w = W[idx3];
            double psi = psi_data[idx3];
            su += w * psi * J11[idx3];
            sv += w * psi * J22[idx3];
        }
        denom_u[idx2] = fmax(su, eps);
        denom_v[idx2] = fmax(sv, eps);
    }

    __global__ void compute_psi_smooth(
        const double* __restrict__ u,
        const double* __restrict__ du,
        const double* __restrict__ v,
        const double* __restrict__ dv,
        double* __restrict__ psi_smooth,
        int m, int n, double a_s, double hx, double hy, double eps)
    {
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (j <= 0 || j >= m-1 || i <= 0 || i >= n-1) return;
        int idx = j*n + i;
        double uc_c = u[idx] + du[idx];
        double vc_c = v[idx] + dv[idx];

        int idxL = j*n + (i-1);
        int idxR = j*n + (i+1);
        int idxT = (j-1)*n + i;
        int idxB = (j+1)*n + i;

        double ux = ((u[idxR] + du[idxR]) - (u[idxL] + du[idxL])) / (2.0*hx);
        double vx = ((v[idxR] + dv[idxR]) - (v[idxL] + dv[idxL])) / (2.0*hx);
        double uy = ((u[idxB] + du[idxB]) - (u[idxT] + du[idxT])) / (2.0*hy);
        double vy = ((v[idxB] + dv[idxB]) - (v[idxT] + dv[idxT])) / (2.0*hy);

        double mag = ux*ux + uy*uy + vx*vx + vy*vy;
        if (mag < 0.0) mag = 0.0;
        psi_smooth[idx] = a_s * pow(mag + eps, a_s - 1.0);
    }

    __global__ void relax_rbgs(
        double* __restrict__ du,
        double* __restrict__ dv,
        const double* __restrict__ u,
        const double* __restrict__ v,
        const double* __restrict__ psi_data,
        const double* __restrict__ W,
        const double* __restrict__ J11,
        const double* __restrict__ J22,
        const double* __restrict__ J12,
        const double* __restrict__ J13,
        const double* __restrict__ J23,
        const double* __restrict__ psi_smooth,
        const double* __restrict__ denom_u,
        const double* __restrict__ denom_v,
        int m, int n, int K,
        double ax, double ay, double omega, double eps,
        int use_aniso, int parity)
    {
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (j <= 0 || j >= m-1 || i <= 0 || i >= n-1) return;
        if ( ((i + j) & 1) != parity ) return;

        int idx = j*n + i;
        int idxL = j*n + (i-1);
        int idxR = j*n + (i+1);
        int idxT = (j-1)*n + i;
        int idxB = (j+1)*n + i;

        double wL = use_aniso ? 0.5*(psi_smooth[idx] + psi_smooth[idxL]) * ax : ax;
        double wR = use_aniso ? 0.5*(psi_smooth[idx] + psi_smooth[idxR]) * ax : ax;
        double wT = use_aniso ? 0.5*(psi_smooth[idx] + psi_smooth[idxT]) * ay : ay;
        double wB = use_aniso ? 0.5*(psi_smooth[idx] + psi_smooth[idxB]) * ay : ay;

        double num_u = wL*(u[idxL] + du[idxL] - u[idx]) +
                       wR*(u[idxR] + du[idxR] - u[idx]) +
                       wT*(u[idxT] + du[idxT] - u[idx]) +
                       wB*(u[idxB] + du[idxB] - u[idx]);

        double num_v = wL*(v[idxL] + dv[idxL] - v[idx]) +
                       wR*(v[idxR] + dv[idxR] - v[idx]) +
                       wT*(v[idxT] + dv[idxT] - v[idx]) +
                       wB*(v[idxB] + dv[idxB] - v[idx]);

        int base = idx * K;
        double s_u = 0.0, s_v = 0.0;
        double du_here = du[idx];
        double dv_here = dv[idx];
        for (int k=0; k<K; ++k){
            int idx3 = base + k;
            double w = W[idx3];
            double psi = psi_data[idx3];
            s_u += w * psi * (J13[idx3] + J12[idx3] * dv_here);
            s_v += w * psi * (J23[idx3] + J12[idx3] * du_here);
        }
        num_u -= s_u;
        num_v -= s_v;

        double den_u = denom_u[idx] + (wL + wR + wT + wB);
        double den_v = denom_v[idx] + (wL + wR + wT + wB);

        double new_du = (1.0 - omega) * du_here + omega * (num_u / fmax(den_u, eps));
        double new_dv = (1.0 - omega) * dv_here + omega * (num_v / fmax(den_v, eps));
        du[idx] = new_du;
        dv[idx] = new_dv;
    }

    } // extern "C"
    """
    )

    k_brow = mod.get_function("boundary_rows")
    k_bcol = mod.get_function("boundary_cols")
    k_psi_data = mod.get_function("compute_psi_data")
    k_denoms = mod.get_function("compute_denoms")
    k_psi_s = mod.get_function("compute_psi_smooth")
    k_relax = mod.get_function("relax_rbgs")

    bx, by = 16, 16
    gx = (n + bx - 1) // bx
    gy = (m + by - 1) // by
    g2 = (gx, gy, 1)
    b2 = (bx, by, 1)
    g1_rows = ((n + 255) // 256, 1, 1)
    g1_cols = ((m + 255) // 256, 1, 1)
    b1 = (256, 1, 1)

    use_aniso = 0 if float(a_smooth) == 1.0 else 1

    for it in range(int(iterations)):
        bool_tick = (
            (it % int(update_lag) == 0)
            if (update_lag_semantics == "torch")
            else (((it + 1) % int(update_lag)) == 0)
        )
        if bool_tick:
            k_psi_data(
                g2,
                b2,
                (
                    J11,
                    J22,
                    J33,
                    J12,
                    J13,
                    J23,
                    du,
                    dv,
                    a_vec,
                    np.float64(eps),
                    psi_data,
                    np.int32(m),
                    np.int32(n),
                    np.int32(K),
                ),
            )
            k_denoms(
                g2,
                b2,
                (
                    psi_data,
                    W,
                    J11,
                    J22,
                    denom_u,
                    denom_v,
                    np.int32(m),
                    np.int32(n),
                    np.int32(K),
                    np.float64(eps),
                ),
            )
            if use_aniso:
                psi_smooth.fill(0.0)
                k_psi_s(
                    g2,
                    b2,
                    (
                        u,
                        du,
                        v,
                        dv,
                        psi_smooth,
                        np.int32(m),
                        np.int32(n),
                        np.float64(a_smooth),
                        np.float64(hx),
                        np.float64(hy),
                        np.float64(eps),
                    ),
                )
                k_brow(g1_rows, b1, (psi_smooth, np.int32(m), np.int32(n)))
                k_bcol(g1_cols, b1, (psi_smooth, np.int32(m), np.int32(n)))
            else:
                psi_smooth.fill(1.0)

        k_brow(g1_rows, b1, (du, np.int32(m), np.int32(n)))
        k_bcol(g1_cols, b1, (du, np.int32(m), np.int32(n)))
        k_brow(g1_rows, b1, (dv, np.int32(m), np.int32(n)))
        k_bcol(g1_cols, b1, (dv, np.int32(m), np.int32(n)))

        k_relax(
            g2,
            b2,
            (
                du,
                dv,
                u,
                v,
                psi_data,
                W,
                J11,
                J22,
                J12,
                J13,
                J23,
                psi_smooth,
                denom_u,
                denom_v,
                np.int32(m),
                np.int32(n),
                np.int32(K),
                np.float64(ax),
                np.float64(ay),
                np.float64(omega),
                np.float64(eps),
                np.int32(use_aniso),
                np.int32(0),
            ),
        )
        k_brow(g1_rows, b1, (du, np.int32(m), np.int32(n)))
        k_bcol(g1_cols, b1, (du, np.int32(m), np.int32(n)))
        k_brow(g1_rows, b1, (dv, np.int32(m), np.int32(n)))
        k_bcol(g1_cols, b1, (dv, np.int32(m), np.int32(n)))

        k_relax(
            g2,
            b2,
            (
                du,
                dv,
                u,
                v,
                psi_data,
                W,
                J11,
                J22,
                J12,
                J13,
                J23,
                psi_smooth,
                denom_u,
                denom_v,
                np.int32(m),
                np.int32(n),
                np.int32(K),
                np.float64(ax),
                np.float64(ay),
                np.float64(omega),
                np.float64(eps),
                np.int32(use_aniso),
                np.int32(1),
            ),
        )

    return cp.asnumpy(du), cp.asnumpy(dv)
