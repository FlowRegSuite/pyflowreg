// level_solver.cpp
// A faithful Pybind11 translation of your original MEX code without suspicious changes.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <cassert>

namespace py = pybind11;

static void set_boundary(double* f, ptrdiff_t m, ptrdiff_t n) {
    // Duplicate boundaries as in the MEX code
    for (ptrdiff_t i = 0; i < n; i++) {
        f[0 + i*m] = f[1 + i*m];
        f[(m-1) + i*m] = f[(m-2) + i*m];
    }
    for (ptrdiff_t j = 0; j < m; j++) {
        f[j + 0*m] = f[j + 1*m];
        f[j + (n-1)*m] = f[j + (n-2)*m];
    }
}

static void nonlinearity_smoothness(double* psi_smooth,
                                    const double* u, const double* du,
                                    const double* v, const double* dv,
                                    ptrdiff_t m, ptrdiff_t n, double a, double hx, double hy) {
    double eps = 0.00001;

    std::vector<double> u_full(m*n, 0.0);
    std::vector<double> v_full(m*n, 0.0);
    for (ptrdiff_t idx = 0; idx < m*n; idx++) {
        u_full[idx] = u[idx] + du[idx];
        v_full[idx] = v[idx] + dv[idx];
    }

    std::vector<double> ux(m*n,0.0), uy(m*n,0.0), vx(m*n,0.0), vy(m*n,0.0);

    // Central differences for gradient approximation
    for (ptrdiff_t j = 0; j < m; j++) {
        for (ptrdiff_t i = 0; i < n; i++) {
            ptrdiff_t idx = j + i*m;
            // ux
            if (n > 1) {
                if (i == 0)
                    ux[idx] = (u_full[j+(i+1)*m]-u_full[idx])/hx;
                else if (i == n-1)
                    ux[idx] = (u_full[idx]-u_full[j+(i-1)*m])/hx;
                else
                    ux[idx] = (u_full[j+(i+1)*m]-u_full[j+(i-1)*m])/(2.0*hx);
            }

            // vx
            if (n > 1) {
                if (i == 0)
                    vx[idx] = (v_full[j+(i+1)*m]-v_full[idx])/hx;
                else if (i == n-1)
                    vx[idx] = (v_full[idx]-v_full[j+(i-1)*m])/hx;
                else
                    vx[idx] = (v_full[j+(i+1)*m]-v_full[j+(i-1)*m])/(2.0*hx);
            }

            // uy
            if (m > 1) {
                if (j == 0)
                    uy[idx] = (u_full[(j+1)+i*m]-u_full[idx])/hy;
                else if (j == m-1)
                    uy[idx] = (u_full[idx]-u_full[(j-1)+i*m])/hy;
                else
                    uy[idx] = (u_full[(j+1)+i*m]-u_full[(j-1)+i*m])/(2.0*hy);
            }

            // vy
            if (m > 1) {
                if (j == 0)
                    vy[idx] = (v_full[(j+1)+i*m]-v_full[idx])/hy;
                else if (j == m-1)
                    vy[idx] = (v_full[idx]-v_full[(j-1)+i*m])/hy;
                else
                    vy[idx] = (v_full[(j+1)+i*m]-v_full[(j-1)+i*m])/(2.0*hy);
            }
        }
    }

    for (ptrdiff_t i = 0; i < m*n; i++) {
        double tmp = ux[i]*ux[i] + uy[i]*uy[i] + vx[i]*vx[i] + vy[i]*vy[i];
        if (tmp < 0.0) tmp = 0.0;
        psi_smooth[i] = a * std::pow(tmp+eps, a-1.0);
    }
}

py::array_t<double> compute_flow(
    py::array_t<double> j11_,
    py::array_t<double> j22_,
    py::array_t<double> j33_,
    py::array_t<double> j12_,
    py::array_t<double> j13_,
    py::array_t<double> j23_,
    py::array_t<double> weight_,
    py::array_t<double> u_,
    py::array_t<double> v_,
    double alpha_x,
    double alpha_y,
    int iterations,
    int update_lag,
    py::array_t<double> a_data_,
    double a_smooth,
    double hx,
    double hy
) {
    auto j11_buf = j11_.request();
    auto j22_buf = j22_.request();
    auto j33_buf = j33_.request();
    auto j12_buf = j12_.request();
    auto j13_buf = j13_.request();
    auto j23_buf = j23_.request();
    auto w_buf = weight_.request();
    auto u_buf = u_.request();
    auto v_buf = v_.request();
    auto a_data_buf = a_data_.request();

    double* j11 = (double*)j11_buf.ptr;
    double* j22 = (double*)j22_buf.ptr;
    double* j33 = (double*)j33_buf.ptr;
    double* j12 = (double*)j12_buf.ptr;
    double* j13 = (double*)j13_buf.ptr;
    double* j23 = (double*)j23_buf.ptr;
    double* data_weight = (double*)w_buf.ptr;
    double* u = (double*)u_buf.ptr;
    double* v = (double*)v_buf.ptr;
    double* a_data = (double*)a_data_buf.ptr;

    ptrdiff_t m = u_buf.shape[0];
    ptrdiff_t n = u_buf.shape[1];

    int n_channels = 1;
    if (j11_buf.ndim == 3) {
        n_channels = (int)j11_buf.shape[2];
    }

    // Create du, dv arrays
    std::vector<py::ssize_t> shape_mn = {(py::ssize_t)m, (py::ssize_t)n};
    py::array_t<double> du_(shape_mn);
    py::array_t<double> dv_(shape_mn);
    auto du_buf = du_.request();
    auto dv_buf = dv_.request();
    double* du = (double*)du_buf.ptr;
    double* dv = (double*)dv_buf.ptr;
    for (ptrdiff_t i = 0; i < m*n; i++) {
        du[i] = 0.0;
        dv[i] = 0.0;
    }

    // Create psi array (m,n,n_channels)
    std::vector<py::ssize_t> shape_mn_ch = { (py::ssize_t)m, (py::ssize_t)n, (py::ssize_t)n_channels };
    py::array_t<double> psi_(shape_mn_ch);
    auto psi_buf = psi_.request();
    double* psi = (double*)psi_buf.ptr;
    for (ptrdiff_t i = 0; i < m*n*n_channels; i++) {
        psi[i] = 1.0;
    }

    py::array_t<double> psi_smooth_(shape_mn);
    auto psi_smooth_buf = psi_smooth_.request();
    double* psi_smooth = (double*)psi_smooth_buf.ptr;
    for (ptrdiff_t i = 0; i < m*n; i++) {
        psi_smooth[i] = 1.0;
    }

    double OMEGA = 1.95;
    double alpha[2];
    alpha[0] = alpha_x;
    alpha[1] = alpha_y;

    int iteration_counter = 0;

    while (iteration_counter < iterations) {
        iteration_counter++;

        if ((iteration_counter % update_lag) == 0) {
            for (int k = 0; k < n_channels; k++) {
                for (ptrdiff_t i = 0; i < m*n; i++) {
                    ptrdiff_t idx = i + k*m*n;
                    double val = j11[idx]*du[i]*du[i] + j22[idx]*dv[i]*dv[i] + j23[idx]*dv[i]
                               + 2.0*j12[idx]*du[i]*dv[i] + 2.0*j13[idx]*du[i] + j23[idx]*dv[i] + j33[idx];
                    if (val < 0.0) val = 0.0;
                    psi[idx] = a_data[k]*std::pow(val+0.00001, a_data[k]-1.0);
                }
            }

            if (a_smooth != 1.0) {
                nonlinearity_smoothness(psi_smooth, u, du, v, dv, m, n, a_smooth, hx, hy);
            } else {
                for (ptrdiff_t i = 0; i < m*n; i++) {
                    psi_smooth[i] = 1.0;
                }
            }
        }

        set_boundary(du, m, n);
        set_boundary(dv, m, n);

        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < m-1; j++) {
                ptrdiff_t idx = j + i*m;
                ptrdiff_t s_idx[4];
                s_idx[0] = j + (i-1)*m; // left
                s_idx[1] = j + (i+1)*m; // right
                s_idx[2] = (j+1) + i*m; // down
                s_idx[3] = (j-1) + i*m; // up

                double denom_u=0.0, denom_v=0.0, num_u=0.0, num_v=0.0;

                if (a_smooth != 1.0) {
                    double tmp = 0.5*(psi_smooth[idx]+psi_smooth[s_idx[0]])*(alpha[0]/(hx*hx));
                    num_u += tmp*(u[s_idx[0]]+du[s_idx[0]] - u[idx]);
                    num_v += tmp*(v[s_idx[0]]+dv[s_idx[0]] - v[idx]);
                    denom_u += tmp;
                    denom_v += tmp;

                    tmp = 0.5*(psi_smooth[idx]+psi_smooth[s_idx[1]])*(alpha[0]/(hx*hx));
                    num_u += tmp*(u[s_idx[1]]+du[s_idx[1]] - u[idx]);
                    num_v += tmp*(v[s_idx[1]]+dv[s_idx[1]] - v[idx]);
                    denom_u += tmp;
                    denom_v += tmp;

                    tmp = 0.5*(psi_smooth[idx]+psi_smooth[s_idx[2]])*(alpha[1]/(hy*hy));
                    num_u += tmp*(u[s_idx[2]]+du[s_idx[2]] - u[idx]);
                    num_v += tmp*(v[s_idx[2]]+dv[s_idx[2]] - v[idx]);
                    denom_u += tmp;
                    denom_v += tmp;

                    tmp = 0.5*(psi_smooth[idx]+psi_smooth[s_idx[3]])*(alpha[1]/(hy*hy));
                    num_u += tmp*(u[s_idx[3]]+du[s_idx[3]] - u[idx]);
                    num_v += tmp*(v[s_idx[3]]+dv[s_idx[3]] - v[idx]);
                    denom_u += tmp;
                    denom_v += tmp;
                } else {
                    double tmp = alpha[0]/(hx*hx);
                    num_u += tmp*(u[s_idx[0]]+du[s_idx[0]] - u[idx]);
                    num_v += tmp*(v[s_idx[0]]+dv[s_idx[0]] - v[idx]);
                    denom_u += tmp;
                    denom_v += tmp;

                    tmp = alpha[0]/(hx*hx);
                    num_u += tmp*(u[s_idx[1]]+du[s_idx[1]] - u[idx]);
                    num_v += tmp*(v[s_idx[1]]+dv[s_idx[1]] - v[idx]);
                    denom_u += tmp;
                    denom_v += tmp;

                    tmp = alpha[1]/(hy*hy);
                    num_u += tmp*(u[s_idx[2]]+du[s_idx[2]] - u[idx]);
                    num_v += tmp*(v[s_idx[2]]+dv[s_idx[2]] - v[idx]);
                    denom_u += tmp;
                    denom_v += tmp;

                    tmp = alpha[1]/(hy*hy);
                    num_u += tmp*(u[s_idx[3]]+du[s_idx[3]] - u[idx]);
                    num_v += tmp*(v[s_idx[3]]+dv[s_idx[3]] - v[idx]);
                    denom_u += tmp;
                    denom_v += tmp;
                }

                for (int k = 0; k < n_channels; k++) {
                    ptrdiff_t nd_idx = idx + k*m*n;
                    num_u -= data_weight[nd_idx]*psi[nd_idx]*(j13[nd_idx] + j12[nd_idx]*dv[idx]);
                    denom_u += data_weight[nd_idx]*psi[nd_idx]*j11[nd_idx];
                    denom_v += data_weight[nd_idx]*psi[nd_idx]*j22[nd_idx];
                }

                double OMEGA = 1.95;
                double du_kp1 = (denom_u!=0.0) ? (num_u/denom_u) : 0.0;
                du[idx] = (1.0-OMEGA)*du[idx] + OMEGA*du_kp1;

                for (int k = 0; k < n_channels; k++) {
                    ptrdiff_t nd_idx = idx + k*m*n;
                    num_v -= data_weight[nd_idx]*psi[nd_idx]*(j23[nd_idx] + j12[nd_idx]*du[idx]);
                }
                double dv_kp1 = (denom_v!=0.0) ? (num_v/denom_v) : 0.0;
                dv[idx] = (1.0-OMEGA)*dv[idx] + OMEGA*dv_kp1;
            }
        }
    }

    std::vector<py::ssize_t> shape_result = {(py::ssize_t)m, (py::ssize_t)n, (py::ssize_t)2};
    py::array_t<double> result_(shape_result);
    auto result_buf = result_.request();
    double* result = (double*)result_buf.ptr;

    for (ptrdiff_t i = 0; i < m; i++) {
        for (ptrdiff_t j = 0; j < n; j++) {
            ptrdiff_t idx = i*n + j;
            ptrdiff_t out_idx = (i*n + j)*2;
            result[out_idx] = du[idx];
            result[out_idx+1] = dv[idx];
        }
    }

    return result_;
}

PYBIND11_MODULE(_level_solver, m) {
    m.doc() = "Literal MEX to Pybind11 Optical Flow Level Solver";
    m.def("compute_flow", &compute_flow, R"pbdoc(
        Compute the optical flow displacement field using a literal translation of the MEX code.
    )pbdoc");
}
