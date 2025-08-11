import numpy as np
from numba import njit

@njit
def set_boundary_2d(f):
    m, n = f.shape
    for i in range(n):
        f[0, i] = f[1, i]
        f[m-1, i] = f[m-2, i]
    for j in range(m):
        f[j, 0] = f[j, 1]
        f[j, n-1] = f[j, n-2]

@njit
def nonlinearity_smoothness_2d(psi_smooth, u, du, v, dv, m, n, a, hx, hy):
    eps = 0.00001
    u_full = u + du
    v_full = v + dv
    ux = np.zeros((m,n))
    uy = np.zeros((m,n))
    vx = np.zeros((m,n))
    vy = np.zeros((m,n))

    for j in range(m):
        for i in range(n):
            # ux
            if n > 1:
                if i == 0:
                    ux[j,i] = (u_full[j,i+1]-u_full[j,i])/hx
                elif i == n-1:
                    ux[j,i] = (u_full[j,i]-u_full[j,i-1])/hx
                else:
                    ux[j,i] = (u_full[j,i+1]-u_full[j,i-1])/(2.0*hx)
            # vx
            if n > 1:
                if i == 0:
                    vx[j,i] = (v_full[j,i+1]-v_full[j,i])/hx
                elif i == n-1:
                    vx[j,i] = (v_full[j,i]-v_full[j,i-1])/hx
                else:
                    vx[j,i] = (v_full[j,i+1]-v_full[j,i-1])/(2.0*hx)
            # uy
            if m > 1:
                if j == 0:
                    uy[j,i] = (u_full[j+1,i]-u_full[j,i])/hy
                elif j == m-1:
                    uy[j,i] = (u_full[j,i]-u_full[j-1,i])/hy
                else:
                    uy[j,i] = (u_full[j+1,i]-u_full[j-1,i])/(2.0*hy)
            # vy
            if m > 1:
                if j == 0:
                    vy[j,i] = (v_full[j+1,i]-v_full[j,i])/hy
                elif j == m-1:
                    vy[j,i] = (v_full[j,i]-v_full[j-1,i])/hy
                else:
                    vy[j,i] = (v_full[j+1,i]-v_full[j-1,i])/(2.0*hy)

    for j in range(m):
        for i in range(n):
            tmp = ux[j,i]*ux[j,i] + uy[j,i]*uy[j,i] + vx[j,i]*vx[j,i] + vy[j,i]*vy[j,i]
            if tmp < 0.0:
                tmp = 0.0
            psi_smooth[j,i] = a * (tmp+eps)**(a-1.0)

@njit
def compute_flow_2d(
    J11, J22, J33, J12, J13, J23,
    weight, u, v,
    alpha_x, alpha_y,
    iterations, update_lag,
    a_data, a_smooth, hx, hy
):
    m, n, n_channels = J11.shape
    du = np.zeros((m,n))
    dv = np.zeros((m,n))
    psi = np.ones((m,n,n_channels))
    psi_smooth = np.ones((m,n))

    OMEGA = 1.95
    alpha = np.array([alpha_x, alpha_y], dtype=np.float64)

    for iteration_counter in range(iterations):
        if (iteration_counter+1) % update_lag == 0:
            # Update psi
            for k in range(n_channels):
                for j in range(m):
                    for i in range(n):
                        val = (J11[j,i,k]*du[j,i]*du[j,i] +
                               J22[j,i,k]*dv[j,i]*dv[j,i] +
                               J23[j,i,k]*dv[j,i] +
                               2.0*J12[j,i,k]*du[j,i]*dv[j,i] +
                               2.0*J13[j,i,k]*du[j,i] +
                               J23[j,i,k]*dv[j,i] +
                               J33[j,i,k])
                        if val < 0.0:
                            val = 0.0
                        psi[j,i,k] = a_data[k]*(val+0.00001)**(a_data[k]-1.0)

            if a_smooth != 1.0:
                nonlinearity_smoothness_2d(psi_smooth, u, du, v, dv, m, n, a_smooth, hx, hy)
            else:
                for j in range(m):
                    for i in range(n):
                        psi_smooth[j,i] = 1.0

        set_boundary_2d(du)
        set_boundary_2d(dv)

        for i in range(1,n-1):
            for j in range(1,m-1):
                denom_u=0.0
                denom_v=0.0
                num_u=0.0
                num_v=0.0

                # neighbors: left(i-1),right(i+1),down(j+1),up(j-1)
                left = (j, i-1)
                right = (j, i+1)
                down = (j+1, i)
                up = (j-1, i)

                if a_smooth != 1.0:
                    # left/right: alpha[0]/hx^2, up/down: alpha[1]/hy^2
                    tmp = 0.5*(psi_smooth[j,i]+psi_smooth[left])*(alpha[0]/(hx*hx))
                    num_u += tmp*(u[left]+du[left] - u[j,i])
                    num_v += tmp*(v[left]+dv[left] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = 0.5*(psi_smooth[j,i]+psi_smooth[right])*(alpha[0]/(hx*hx))
                    num_u += tmp*(u[right]+du[right] - u[j,i])
                    num_v += tmp*(v[right]+dv[right] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = 0.5*(psi_smooth[j,i]+psi_smooth[down])*(alpha[1]/(hy*hy))
                    num_u += tmp*(u[down]+du[down] - u[j,i])
                    num_v += tmp*(v[down]+dv[down] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = 0.5*(psi_smooth[j,i]+psi_smooth[up])*(alpha[1]/(hy*hy))
                    num_u += tmp*(u[up]+du[up] - u[j,i])
                    num_v += tmp*(v[up]+dv[up] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp
                else:
                    tmp = alpha[0]/(hx*hx)
                    num_u += tmp*(u[left]+du[left] - u[j,i])
                    num_v += tmp*(v[left]+dv[left] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = alpha[0]/(hx*hx)
                    num_u += tmp*(u[right]+du[right] - u[j,i])
                    num_v += tmp*(v[right]+dv[right] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = alpha[1]/(hy*hy)
                    num_u += tmp*(u[down]+du[down] - u[j,i])
                    num_v += tmp*(v[down]+dv[down] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = alpha[1]/(hy*hy)
                    num_u += tmp*(u[up]+du[up] - u[j,i])
                    num_v += tmp*(v[up]+dv[up] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                # Data terms
                for k in range(n_channels):
                    val_u = weight[j,i,k]*psi[j,i,k]*(J13[j,i,k] + J12[j,i,k]*dv[j,i])
                    num_u -= val_u
                    denom_u += weight[j,i,k]*psi[j,i,k]*J11[j,i,k]
                    denom_v += weight[j,i,k]*psi[j,i,k]*J22[j,i,k]

                du_kp1 = num_u/denom_u if denom_u!=0 else 0.0
                du[j,i] = (1.0-OMEGA)*du[j,i] + OMEGA*du_kp1

                num_v2 = num_v
                for k in range(n_channels):
                    num_v2 -= weight[j,i,k]*psi[j,i,k]*(J23[j,i,k] + J12[j,i,k]*du[j,i])

                dv_kp1 = num_v2/denom_v if denom_v!=0 else 0.0
                dv[j,i] = (1.0-OMEGA)*dv[j,i] + OMEGA*dv_kp1

    # Return final DU, DV
    result = np.zeros((m,n,2))
    result[:,:,0] = du
    result[:,:,1] = dv
    return result

def benchmark_python():
    m = 3
    n = 3
    hx = 2.0
    hy = 0.5
    iterations = 10
    update_lag = 1
    alpha_x, alpha_y = 2.0, 2.0
    a_data = np.array([0.45,0.7],dtype=np.float64)
    a_smooth = 0.75

    # Manually match the MATLAB arrays:
    # shape (3,3,2)
    J11 = np.stack([
        np.array([[2.1,2.1,2.1],[2.1,2.2,2.1],[2.1,2.1,2.1]]),
        np.array([[2.0,2.0,2.0],[2.0,2.05,2.0],[2.0,2.0,2.0]])
    ], axis=2)
    J22 = np.stack([
        np.array([[2.0,2.0,2.0],[2.0,2.1,2.0],[2.0,2.0,2.0]]),
        np.array([[2.1,2.1,2.1],[2.1,2.15,2.1],[2.1,2.1,2.1]])
    ], axis=2)
    J33 = np.stack([
        np.full((3,3),2.0),
        np.full((3,3),2.0)
    ], axis=2)
    J12 = np.stack([
        np.array([[0.0,0.0,0.0],[0.0,0.05,0.0],[0.0,0.0,0.0]]),
        np.array([[0.0,0.0,0.0],[0.0,-0.05,0.0],[0.0,0.0,0.0]])
    ], axis=2)
    J13 = np.stack([
        np.array([[0.1,0.1,0.1],[0.1,0.15,0.1],[0.1,0.1,0.1]]),
        np.array([[0.2,0.2,0.2],[0.2,0.25,0.2],[0.2,0.2,0.2]])
    ], axis=2)
    J23 = np.stack([
        np.array([[-0.1,-0.1,-0.1],[-0.1,-0.15,-0.1],[-0.1,-0.1,-0.1]]),
        np.array([[-0.2,-0.2,-0.2],[-0.2,-0.25,-0.2],[-0.2,-0.2,-0.2]])
    ], axis=2)

    weight = np.stack([np.ones((3,3)), np.ones((3,3))], axis=2)
    u_init = np.zeros((3,3))
    v_init = np.zeros((3,3))
    u_init[1,1] = 0.01
    v_init[1,1] = -0.01

    flow = compute_flow_2d(J11, J22, J33, J12, J13, J23,
                           weight, u_init, v_init,
                           alpha_x, alpha_y,
                           iterations, update_lag,
                           a_data, a_smooth, hx, hy)
    du = flow[:,:,0]
    dv = flow[:,:,1]
    print("Python DU after completion:")
    print(du)
    print("Python DV after completion:")
    print(dv)

if __name__ == "__main__":
    benchmark_python()
