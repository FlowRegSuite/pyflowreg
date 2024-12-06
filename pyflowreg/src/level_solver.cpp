// Author   : Philipp Flotho
// Copyright 2024 by Philipp Flotho, All rights reserved.
//
// This is a refactored version of the original MEX code for the optical flow level solver,
// adapted to be called from Python using pybind11.
//
// In this refactoring:
// - All MATLAB/MEX specific calls (mxArray, mexErrMsgIdAndTxt, mxGetPr, etc.) have been removed.
// - Input/Output is now handled via function arguments using pybind11::array_t<double>.
// - The functionality that previously relied on mexCallMATLAB for gradients and array operations
//   has been replaced with direct C++ implementations.
//
// Note: Ensure that pybind11 and NumPy are properly set up in your environment.
//       This code assumes that inputs are provided as NumPy arrays with appropriate shapes.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <math.h>
#include <stdlib.h>
#include <vector>

namespace py = pybind11;

static void set_boundary(double* f, int m, int n) {
    for (int i = 0; i < n; i++) {
        f[0 + i*m] = f[1 + i*m];
        f[(m-1) + i*m] = f[(m-2) + i*m];
    }
    for (int j = 0; j < m; j++) {
        f[j + 0*m] = f[j + 1*m];
        f[j + (n-1)*m] = f[j + (n-2)*m];
    }
}

static void nonlinearity_smoothness(double* psi_smooth,
                                    const double* u, const double* du,
                                    const double* v, const double* dv,
                                    int m, int n, double a, double hx, double hy) {
    double eps = 0.00001;
    // Compute gradients of (u+du) and (v+dv)
    std::vector<double> u_full(m*n), v_full(m*n);
    for (int i = 0; i < m*n; i++) {
        u_full[i] = u[i] + du[i];
        v_full[i] = v[i] + dv[i];
    }

    std::vector<double> ux(m*n), uy(m*n), vx(m*n), vy(m*n);

    // gradient w.r.t x
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int idx = j + i*m;
            int idx_r = j + (i == n-1 ? i : i+1)*m;
            ux[idx] = (u_full[idx_r] - u_full[idx]) / ( (i == n-1) ? hx : hx );
            vx[idx] = (v_full[idx_r] - v_full[idx]) / ( (i == n-1) ? hx : hx );
        }
    }

    // gradient w.r.t y
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int idx = j + i*m;
            int idx_b = (j == m-1 ? j : j+1) + i*m;
            uy[idx] = (u_full[idx_b] - u_full[idx]) / ( (j == m-1) ? hy : hy );
            vy[idx] = (v_full[idx_b] - v_full[idx]) / ( (j == m-1) ? hy : hy );
        }
    }

    for (int i = 0; i < m*n; i++) {
        double tmp = ux[i]*ux[i] + uy[i]*uy[i] + vx[i]*vx[i] + vy[i]*vy[i];
        tmp = tmp < 0 ? 0 : tmp;
        psi_smooth[i] = a * pow(tmp + eps, a - 1);
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
    // input arguments
    // checking number of inputs
    // setting inputs
    py::buffer_info j11_buf = j11_.request();
    py::buffer_info j22_buf = j22_.request();
    py::buffer_info j33_buf = j33_.request();
    py::buffer_info j12_buf = j12_.request();
    py::buffer_info j13_buf = j13_.request();
    py::buffer_info j23_buf = j23_.request();
    py::buffer_info w_buf = weight_.request();
    py::buffer_info u_buf = u_.request();
    py::buffer_info v_buf = v_.request();
    py::buffer_info a_data_buf = a_data_.request();

    int m = (int)j11_buf.shape[0];
    int n = (int)j11_buf.shape[1];

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

    // We assume a_data has length equal to number of channels.
    // n_channels determined by dimensions of J11.
    int n_channels = 1;
    if (j11_buf.ndim == 3) {
        n_channels = (int)j11_buf.shape[2];
    }

    // creating output arrays du, dv
    py::array_t<double> du_( { (size_t)m, (size_t)n } );
    py::array_t<double> dv_( { (size_t)m, (size_t)n } );
    py::buffer_info du_buf = du_.request();
    py::buffer_info dv_buf = dv_.request();
    double* du = (double*)du_buf.ptr;
    double* dv = (double*)dv_buf.ptr;

    for (int i = 0; i < m*n; i++) {
        du[i] = 0;
        dv[i] = 0;
    }

    // updating the non-linearities etc.
    py::array_t<double> psi_(j11_.request().size);
    double* psi = (double*)psi_.request().ptr;
    for (int i = 0; i < m*n*((int)n_channels); i++) {
        psi[i] = 1;
    }

    py::array_t<double> psi_smooth_( { (size_t)m, (size_t)n } );
    double* psi_smooth = (double*)psi_smooth_.request().ptr;
    for (int i = 0; i < m*n; i++) {
        psi_smooth[i] = 1;
    }

    double OMEGA = 1.95;

    int iteration_counter = 0;

    double alpha[2];
    alpha[0] = alpha_x;
    alpha[1] = alpha_y;

    double tmp1 = alpha[0] / (hx*hx);
    double tmp2 = alpha[1] / (hy*hy);

    // main iteration loop from Flow-Registration without MEX calls.
    while (iteration_counter++ < iterations) {

        if (iteration_counter % update_lag == 0) {
            for (int k = 0; k < n_channels; k++) {
                for (int i = 0; i < m*n; i++) {
                    int idx = i + k*m*n;
                    double val = j11[idx]*du[i]*du[i] + j22[idx]*dv[i]*dv[i] + j23[idx]*dv[i]
                                 +2*j12[idx]*du[i]*dv[i] + 2*j13[idx]*du[i] + j23[idx]*dv[i] + j33[idx];
                    val = val < 0 ? 0 : val;
                    psi[idx] = a_data[k] * pow(val + 0.00001, a_data[k] - 1);
                }
            }

            if (a_smooth != 1) {
                nonlinearity_smoothness(psi_smooth, u, du, v, dv, m, n, a_smooth, hx, hy);
            } else {
                for (int i = 0; i < m*n; i++) {
                    psi_smooth[i] = 1;
                }
            }
        }

        set_boundary(du, m, n);
        set_boundary(dv, m, n);

        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < m-1; j++) {

                int idx = j + i*m;

                int s_idx[4];
                s_idx[0] = j + (i-1)*m;
                s_idx[1] = j + (i+1)*m;
                s_idx[2] = (j+1) + i*m;
                s_idx[3] = (j-1) + i*m;

                double denom_u = 0;
                double denom_v = 0;
                double num_u = 0;
                double num_v = 0;

                if (a_smooth != 1) {
                    for (int d = 0; d < 4; d++) {
                        double tmp = 0.5*(psi_smooth[idx] + psi_smooth[s_idx[d]]);
                        if (d == 0 || d == 1) tmp *= tmp1;
                        else tmp *= tmp2;

                        num_u += tmp * (u[s_idx[d]] + du[s_idx[d]] - u[idx]);
                        num_v += tmp * (v[s_idx[d]] + dv[s_idx[d]] - v[idx]);
                        denom_u += tmp;
                        denom_v += tmp;
                    }
                } else {
                    for (int d = 0; d < 4; d++) {
                        double tmp = (d < 2) ? tmp1 : tmp2;
                        num_u += tmp * (u[s_idx[d]] + du[s_idx[d]] - u[idx]);
                        num_v += tmp * (v[s_idx[d]] + dv[s_idx[d]] - v[idx]);
                        denom_u += tmp;
                        denom_v += tmp;
                    }
                }

                for (int k = 0; k < n_channels; k++) {
                    int nd_idx = idx + k*m*n;
                    num_u -= ((n_channels == 1) ? data_weight[nd_idx] : data_weight[nd_idx]) * psi[nd_idx] * (j13[nd_idx] + j12[nd_idx]*dv[idx]);
                    denom_u += ((n_channels == 1) ? data_weight[nd_idx] : data_weight[nd_idx]) * psi[nd_idx] * j11[nd_idx];
                    denom_v += ((n_channels == 1) ? data_weight[nd_idx] : data_weight[nd_idx]) * psi[nd_idx] * j22[nd_idx];
                }

                double du_kp1 = num_u / denom_u;
                du[idx] = (1 - OMEGA)*du[idx] + OMEGA*du_kp1;

                for (int k = 0; k < n_channels; k++) {
                    int nd_idx = idx + k*m*n;
                    num_v -= ((n_channels == 1) ? data_weight[nd_idx] : data_weight[nd_idx]) * psi[nd_idx]*(j23[nd_idx] + j12[nd_idx]*du[idx]);
                }
                double dv_kp1 = num_v / denom_v;
                dv[idx] = (1 - OMEGA)*dv[idx] + OMEGA*dv_kp1;
            }
        }
    }

    // return du and dv as a combined result
    // For convenience, we return a 3D array with shape (m,n,2).
    py::array_t<double> result_( { (size_t)m, (size_t)n, (size_t)2 } );
    double* result = (double*)result_.request().ptr;
    for (int i = 0; i < m*n; i++) {
        result[i] = du[i];
        result[i + m*n] = dv[i];
    }

    return result_;
}

PYBIND11_MODULE(_level_solver, m) {
    m.def("compute_flow", &compute_flow, "Compute optical flow displacements");
}
