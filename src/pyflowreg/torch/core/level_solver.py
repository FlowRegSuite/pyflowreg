import torch


def _set_boundary2d_(A):
    A[:, 0] = A[:, 1]
    A[:, -1] = A[:, -2]
    A[0, :] = A[1, :]
    A[-1, :] = A[-2, :]


def level_solver_rbgs_torch(
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
    update_lag_semantics: str = "torch",
):
    with torch.no_grad():
        m, n, K = J11.shape
        du = torch.zeros_like(u)
        dv = torch.zeros_like(v)
        ax = torch.as_tensor(alpha[0], dtype=u.dtype, device=u.device) / (hx * hx)
        ay = torch.as_tensor(alpha[1], dtype=u.dtype, device=u.device) / (hy * hy)
        epsv = torch.as_tensor(eps, dtype=u.dtype, device=u.device)
        if isinstance(a_data, (float, int)):
            A_vec = torch.full(
                (1, 1, K), float(a_data), dtype=J11.dtype, device=J11.device
            )
        else:
            A_vec = torch.as_tensor(a_data, dtype=J11.dtype, device=J11.device).view(
                1, 1, -1
            )
        W = (
            weight_level
            if (weight_level.dim() == 3 and weight_level.shape == J11.shape)
            else weight_level.view(1, 1, -1)
        )
        psi_data = torch.ones_like(J11)
        psi_smooth = torch.ones_like(u)
        denom_u_data = torch.zeros_like(u)
        denom_v_data = torch.zeros_like(u)
        c1 = slice(1, m - 1)
        c2 = slice(1, n - 1)
        ii = torch.arange(1, m - 1, device=u.device).view(-1, 1)
        jj = torch.arange(1, n - 1, device=u.device).view(1, -1)
        Rmask = (ii + jj) % 2 == 0
        Bmask = ~Rmask
        num_u = torch.empty_like(u)
        num_v = torch.empty_like(v)
        den_u = torch.empty_like(u)
        den_v = torch.empty_like(v)
        for it in range(iterations):
            upd_tick = (
                (it % update_lag) == 0
                if update_lag_semantics == "torch"
                else ((it + 1) % update_lag) == 0
            )
            if upd_tick:
                du3 = du.unsqueeze(-1)
                dv3 = dv.unsqueeze(-1)
                E = (
                    J11 * (du3 * du3)
                    + J22 * (dv3 * dv3)
                    + 2 * J12 * (du3 * dv3)
                    + 2 * J13 * du3
                    + 2 * J23 * dv3
                    + J33
                )
                E.clamp_min_(0)
                psi_data = A_vec * (E + epsv) ** (A_vec - 1)
                if a_smooth != 1:
                    uc = u + du
                    vc = v + dv
                    ux = (uc[c1, 2:n] - uc[c1, 0 : n - 2]) / (2 * hx)
                    uy = (uc[2:m, c2] - uc[0 : m - 2, c2]) / (2 * hy)
                    vx = (vc[c1, 2:n] - vc[c1, 0 : n - 2]) / (2 * hx)
                    vy = (vc[2:m, c2] - vc[0 : m - 2, c2]) / (2 * hy)
                    mag = torch.zeros_like(u)
                    mag[c1, c2] = ux * ux + uy * uy + vx * vx + vy * vy
                    psi_smooth.zero_()
                    a_s = torch.as_tensor(a_smooth, dtype=u.dtype, device=u.device)
                    psi_smooth[c1, c2] = a_s * (mag[c1, c2] + epsv) ** (a_s - 1)
                    _set_boundary2d_(psi_smooth)
                else:
                    psi_smooth.fill_(1)
                denom_u_data = torch.sum(W * psi_data * J11, dim=2)
                denom_v_data = torch.sum(W * psi_data * J22, dim=2)
                denom_u_data = torch.maximum(denom_u_data, epsv)
                denom_v_data = torch.maximum(denom_v_data, epsv)
            _set_boundary2d_(du)
            _set_boundary2d_(dv)
            if a_smooth != 1:
                psiC = psi_smooth[c1, c2]
                wL_u = 0.5 * (psiC + psi_smooth[c1, 0 : n - 2]) * ax
                wR_u = 0.5 * (psiC + psi_smooth[c1, 2:n]) * ax
                wT_u = 0.5 * (psiC + psi_smooth[0 : m - 2, c2]) * ay
                wB_u = 0.5 * (psiC + psi_smooth[2:m, c2]) * ay
                wL_v = wL_u
                wR_v = wR_u
                wT_v = wT_u
                wB_v = wB_u
            else:
                wL_u = wR_u = ax
                wT_u = wB_u = ay
                wL_v = wR_v = ax
                wT_v = wB_v = ay
            num_u[c1, c2] = (
                -torch.sum(W * psi_data * (J13 + J12 * dv.unsqueeze(-1)), dim=2)[c1, c2]
                + wL_u * (u[c1, 0 : n - 2] + du[c1, 0 : n - 2] - u[c1, c2])
                + wR_u * (u[c1, 2:n] + du[c1, 2:n] - u[c1, c2])
                + wT_u * (u[0 : m - 2, c2] + du[0 : m - 2, c2] - u[c1, c2])
                + wB_u * (u[2:m, c2] + du[2:m, c2] - u[c1, c2])
            )
            den_u[c1, c2] = denom_u_data[c1, c2] + (wL_u + wR_u + wT_u + wB_u)
            num_v[c1, c2] = (
                -torch.sum(W * psi_data * (J23 + J12 * du.unsqueeze(-1)), dim=2)[c1, c2]
                + wL_v * (v[c1, 0 : n - 2] + dv[c1, 0 : n - 2] - v[c1, c2])
                + wR_v * (v[c1, 2:n] + dv[c1, 2:n] - v[c1, c2])
                + wT_v * (v[0 : m - 2, c2] + dv[0 : m - 2, c2] - v[c1, c2])
                + wB_v * (v[2:m, c2] + dv[2:m, c2] - v[c1, c2])
            )
            den_v[c1, c2] = denom_v_data[c1, c2] + (wL_v + wR_v + wT_v + wB_v)
            upd_u = du[c1, c2]
            upd_v = dv[c1, c2]
            new_u = (1 - omega) * upd_u + omega * (
                num_u[c1, c2] / torch.maximum(den_u[c1, c2], epsv)
            )
            new_v = (1 - omega) * upd_v + omega * (
                num_v[c1, c2] / torch.maximum(den_v[c1, c2], epsv)
            )
            tmp = upd_u.clone()
            tmp[Rmask] = new_u[Rmask]
            du[c1, c2] = tmp
            tmp = upd_v.clone()
            tmp[Rmask] = new_v[Rmask]
            dv[c1, c2] = tmp
            _set_boundary2d_(du)
            _set_boundary2d_(dv)
            num_u[c1, c2] = (
                -torch.sum(W * psi_data * (J13 + J12 * dv.unsqueeze(-1)), dim=2)[c1, c2]
                + wL_u * (u[c1, 0 : n - 2] + du[c1, 0 : n - 2] - u[c1, c2])
                + wR_u * (u[c1, 2:n] + du[c1, 2:n] - u[c1, c2])
                + wT_u * (u[0 : m - 2, c2] + du[0 : m - 2, c2] - u[c1, c2])
                + wB_u * (u[2:m, c2] + du[2:m, c2] - u[c1, c2])
            )
            den_u[c1, c2] = denom_u_data[c1, c2] + (wL_u + wR_u + wT_u + wB_u)
            num_v[c1, c2] = (
                -torch.sum(W * psi_data * (J23 + J12 * du.unsqueeze(-1)), dim=2)[c1, c2]
                + wL_v * (v[c1, 0 : n - 2] + dv[c1, 0 : n - 2] - v[c1, c2])
                + wR_v * (v[c1, 2:n] + dv[c1, 2:n] - v[c1, c2])
                + wT_v * (v[0 : m - 2, c2] + dv[0 : m - 2, c2] - v[c1, c2])
                + wB_v * (v[2:m, c2] + dv[2:m, c2] - v[c1, c2])
            )
            den_v[c1, c2] = denom_v_data[c1, c2] + (wL_v + wR_v + wT_v + wB_v)
            upd_u = du[c1, c2]
            upd_v = dv[c1, c2]
            new_u = (1 - omega) * upd_u + omega * (
                num_u[c1, c2] / torch.maximum(den_u[c1, c2], epsv)
            )
            new_v = (1 - omega) * upd_v + omega * (
                num_v[c1, c2] / torch.maximum(den_v[c1, c2], epsv)
            )
            tmp = upd_u.clone()
            tmp[Bmask] = new_u[Bmask]
            du[c1, c2] = tmp
            tmp = upd_v.clone()
            tmp[Bmask] = new_v[Bmask]
            dv[c1, c2] = tmp
            _set_boundary2d_(du)
            _set_boundary2d_(dv)
        return du, dv
