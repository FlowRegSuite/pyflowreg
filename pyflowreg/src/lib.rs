use ndarray::{Array2, Array3, s};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use std::f64;

#[pyfunction]
fn compute_flow(
    j11: PyReadonlyArrayDyn<f64>,
    j22: PyReadonlyArrayDyn<f64>,
    j33: PyReadonlyArrayDyn<f64>,
    j12: PyReadonlyArrayDyn<f64>,
    j13: PyReadonlyArrayDyn<f64>,
    j23: PyReadonlyArrayDyn<f64>,
    weight: PyReadonlyArrayDyn<f64>,
    u_in: PyReadonlyArrayDyn<f64>,
    v_in: PyReadonlyArrayDyn<f64>,
    alpha_x: f64,
    alpha_y: f64,
    iterations: i32,
    update_lag: i32,
    a_data: PyReadonlyArrayDyn<f64>,
    a_smooth: f64,
    hx: f64,
    hy: f64,
    py: Python
) -> PyResult<Py<PyArrayDyn<f64>>> {
    // Assume inputs are C-contiguous and shaped as in the Numba code.
    // j11, j22, etc. are shape [m, n, n_channels]
    let j11 = j11.as_array();
    let j22 = j22.as_array();
    let j33 = j33.as_array();
    let j12 = j12.as_array();
    let j13 = j13.as_array();
    let j23 = j23.as_array();
    let weight = weight.as_array();
    let a_data = a_data.as_array();

    // Get dimensions. We assume that u and v are 2D arrays.
    let u_arr = u_in.as_array();
    let v_arr = v_in.as_array();
    let m = u_arr.dim().0;
    let n = u_arr.dim().1;
    let n_channels = if j11.ndim() == 3 { j11.shape()[2] } else { 1 };

    // Create working arrays
    let mut du = Array2::<f64>::zeros((m, n));
    let mut dv = Array2::<f64>::zeros((m, n));
    let mut psi = Array3::<f64>::from_elem((m, n, n_channels), 1.0);
    let mut psi_smooth = Array2::<f64>::from_elem((m, n), 1.0);
    // Copy initial u, v (we assume these are the current displacement fields)
    let mut u = u_arr.to_owned();
    let mut v = v_arr.to_owned();

    let omega = 1.95;
    let alpha = [alpha_x, alpha_y];

    // Helper: set boundaries of a 2D array
    let mut set_boundary = |arr: &mut Array2<f64>| {
        for i in 0..n {
            arr[[0, i]] = arr[[1, i]];
            arr[[m - 1, i]] = arr[[m - 2, i]];
        }
        for j in 0..m {
            arr[[j, 0]] = arr[[j, 1]];
            arr[[j, n - 1]] = arr[[j, n - 2]];
        }
    };

    // Main iteration loop
    for iter in 0..iterations {
        if ((iter + 1) % update_lag as usize) == 0 {
            // Update psi over all channels
            for k in 0..n_channels {
                for j in 0..m {
                    for i in 0..n {
                        let val = j11[[j, i, k]] * du[[j, i]] * du[[j, i]]
                            + j22[[j, i, k]] * dv[[j, i]] * dv[[j, i]]
                            + j23[[j, i, k]] * dv[[j, i]]
                            + 2.0 * j12[[j, i, k]] * du[[j, i]] * dv[[j, i]]
                            + 2.0 * j13[[j, i, k]] * du[[j, i]]
                            + j23[[j, i, k]] * dv[[j, i]]
                            + j33[[j, i, k]];
                        let val = if val < 0.0 { 0.0 } else { val };
                        psi[[j, i, k]] = a_data[[k]] * (val + 1e-5).powf(a_data[[k]] - 1.0);
                    }
                }
            }
            // Update psi_smooth using nonlinearity_smoothness equivalent.
            {
                let eps = 1e-5;
                let u_full = &u + &du;
                let v_full = &v + &dv;
                // Create temporary arrays for gradients.
                let mut ux = Array2::<f64>::zeros((m, n));
                let mut uy = Array2::<f64>::zeros((m, n));
                let mut vx_arr = Array2::<f64>::zeros((m, n));
                let mut vy_arr = Array2::<f64>::zeros((m, n));
                for j in 0..m {
                    for i in 0..n {
                        // Compute ux
                        if i == 0 {
                            ux[[j, i]] = (u_full[[j, i + 1]] - u_full[[j, i]]) / hx;
                        } else if i == n - 1 {
                            ux[[j, i]] = (u_full[[j, i]] - u_full[[j, i - 1]]) / hx;
                        } else {
                            ux[[j, i]] = (u_full[[j, i + 1]] - u_full[[j, i - 1]]) / (2.0 * hx);
                        }
                        // Compute vx
                        if i == 0 {
                            vx_arr[[j, i]] = (v_full[[j, i + 1]] - v_full[[j, i]]) / hx;
                        } else if i == n - 1 {
                            vx_arr[[j, i]] = (v_full[[j, i]] - v_full[[j, i - 1]]) / hx;
                        } else {
                            vx_arr[[j, i]] = (v_full[[j, i + 1]] - v_full[[j, i - 1]]) / (2.0 * hx);
                        }
                        // Compute uy
                        if j == 0 {
                            uy[[j, i]] = (u_full[[j + 1, i]] - u_full[[j, i]]) / hy;
                        } else if j == m - 1 {
                            uy[[j, i]] = (u_full[[j, i]] - u_full[[j - 1, i]]) / hy;
                        } else {
                            uy[[j, i]] = (u_full[[j + 1, i]] - u_full[[j - 1, i]]) / (2.0 * hy);
                        }
                        // Compute vy
                        if j == 0 {
                            vy_arr[[j, i]] = (v_full[[j + 1, i]] - v_full[[j, i]]) / hy;
                        } else if j == m - 1 {
                            vy_arr[[j, i]] = (v_full[[j, i]] - v_full[[j - 1, i]]) / hy;
                        } else {
                            vy_arr[[j, i]] = (v_full[[j + 1, i]] - v_full[[j - 1, i]]) / (2.0 * hy);
                        }
                    }
                }
                for j in 0..m {
                    for i in 0..n {
                        let tmp = ux[[j, i]].powi(2)
                            + uy[[j, i]].powi(2)
                            + vx_arr[[j, i]].powi(2)
                            + vy_arr[[j, i]].powi(2);
                        psi_smooth[[j, i]] = a_smooth * (tmp + eps).powf(a_smooth - 1.0);
                    }
                }
            }
        }

        // Set boundaries before the interior update (virtual nodes)
        set_boundary(&mut du);
        set_boundary(&mut dv);

        // SOR update over interior points only
        for j in 1..(m - 1) {
            for i in 1..(n - 1) {
                let mut num_u = 0.0;
                let mut num_v = 0.0;
                let mut denom_u = 0.0;
                let mut denom_v = 0.0;

                // Define neighbors: left, right, down, up
                let left = (j, i - 1);
                let right = (j, i + 1);
                let down = (j + 1, i);
                let up = (j - 1, i);

                if (a_smooth - 1.0).abs() > 1e-12 {
                    let tmp = 0.5 * (psi_smooth[[j, i]] + psi_smooth[[left.0, left.1]]) * (alpha[0] / (hx * hx));
                    num_u += tmp * ((u[[left.0, left.1]] + du[[left.0, left.1]]) - u[[j, i]]);
                    num_v += tmp * ((v[[left.0, left.1]] + dv[[left.0, left.1]]) - v[[j, i]]);
                    denom_u += tmp;
                    denom_v += tmp;

                    let tmp = 0.5 * (psi_smooth[[j, i]] + psi_smooth[[right.0, right.1]]) * (alpha[0] / (hx * hx));
                    num_u += tmp * ((u[[right.0, right.1]] + du[[right.0, right.1]]) - u[[j, i]]);
                    num_v += tmp * ((v[[right.0, right.1]] + dv[[right.0, right.1]]) - v[[j, i]]);
                    denom_u += tmp;
                    denom_v += tmp;

                    let tmp = 0.5 * (psi_smooth[[j, i]] + psi_smooth[[down.0, down.1]]) * (alpha[1] / (hy * hy));
                    num_u += tmp * ((u[[down.0, down.1]] + du[[down.0, down.1]]) - u[[j, i]]);
                    num_v += tmp * ((v[[down.0, down.1]] + dv[[down.0, down.1]]) - v[[j, i]]);
                    denom_u += tmp;
                    denom_v += tmp;

                    let tmp = 0.5 * (psi_smooth[[j, i]] + psi_smooth[[up.0, up.1]]) * (alpha[1] / (hy * hy));
                    num_u += tmp * ((u[[up.0, up.1]] + du[[up.0, up.1]]) - u[[j, i]]);
                    num_v += tmp * ((v[[up.0, up.1]] + dv[[up.0, up.1]]) - v[[j, i]]);
                    denom_u += tmp;
                    denom_v += tmp;
                } else {
                    let tmp = alpha[0] / (hx * hx);
                    num_u += tmp * ((u[[left.0, left.1]] + du[[left.0, left.1]]) - u[[j, i]]);
                    num_v += tmp * ((v[[left.0, left.1]] + dv[[left.0, left.1]]) - v[[j, i]]);
                    denom_u += tmp;
                    denom_v += tmp;

                    let tmp = alpha[0] / (hx * hx);
                    num_u += tmp * ((u[[right.0, right.1]] + du[[right.0, right.1]]) - u[[j, i]]);
                    num_v += tmp * ((v[[right.0, right.1]] + dv[[right.0, right.1]]) - v[[j, i]]);
                    denom_u += tmp;
                    denom_v += tmp;

                    let tmp = alpha[1] / (hy * hy);
                    num_u += tmp * ((u[[down.0, down.1]] + du[[down.0, down.1]]) - u[[j, i]]);
                    num_v += tmp * ((v[[down.0, down.1]] + dv[[down.0, down.1]]) - v[[j, i]]);
                    denom_u += tmp;
                    denom_v += tmp;

                    let tmp = alpha[1] / (hy * hy);
                    num_u += tmp * ((u[[up.0, up.1]] + du[[up.0, up.1]]) - u[[j, i]]);
                    num_v += tmp * ((v[[up.0, up.1]] + dv[[up.0, up.1]]) - v[[j, i]]);
                    denom_u += tmp;
                    denom_v += tmp;
                }
                for k in 0..n_channels {
                    let val_u = weight[[j, i, k]] * psi[[j, i, k]]
                        * (j13[[j, i, k]] + j12[[j, i, k]] * dv[[j, i]]);
                    num_u -= val_u;
                    denom_u += weight[[j, i, k]] * psi[[j, i, k]] * j11[[j, i, k]];
                    denom_v += weight[[j, i, k]] * psi[[j, i, k]] * j22[[j, i, k]];
                }
                let du_kp1 = if denom_u != 0.0 { num_u / denom_u } else { 0.0 };
                du[[j, i]] = (1.0 - omega) * du[[j, i]] + omega * du_kp1;
                let mut num_v2 = num_v;
                for k in 0..n_channels {
                    num_v2 -= weight[[j, i, k]] * psi[[j, i, k]]
                        * (j23[[j, i, k]] + j12[[j, i, k]] * du[[j, i]]);
                }
                let dv_kp1 = if denom_v != 0.0 { num_v2 / denom_v } else { 0.0 };
                dv[[j, i]] = (1.0 - omega) * dv[[j, i]] + omega * dv_kp1;
            }
        }
        // (No extra boundary update at the end, since our loop excludes boundaries.)
    }
    // Compose final flow: shape (m, n, 2)
    let mut flow = Array3::<f64>::zeros((m, n, 2));
    for j in 0..m {
        for i in 0..n {
            flow[[j, i, 0]] = du[[j, i]];
            flow[[j, i, 1]] = dv[[j, i]];
        }
    }
    Ok(flow.into_pyarray(py).to_owned())
}

#[pymodule]
fn level_solver(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_flow, m)?)?;
    Ok(())
}
