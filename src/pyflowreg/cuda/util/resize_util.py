import numpy as np
import cupy as cp


def _cubic(x):
    A = -0.75
    ax = np.abs(x)
    y = np.where(ax < 1.0, (A + 2.0) * ax**3 - (A + 3.0) * ax**2 + 1.0, 0.0)
    m = (ax >= 1.0) & (ax < 2.0)
    y[m] = A * ax[m] ** 3 - 5.0 * A * ax[m] ** 2 + 8.0 * A * ax[m] - 4.0 * A
    return y


def _reflect_idx(j, n):
    if n <= 1:
        return np.zeros_like(j, dtype=np.int32)
    j = j.astype(np.int64, copy=True)
    while True:
        mask = (j < 0) | (j >= n)
        if not np.any(mask):
            break
        j[mask] = np.where(j[mask] < 0, -j[mask] - 1, 2 * n - 1 - j[mask])
    return j.astype(np.int32, copy=False)


def _precompute_fused_gauss_cubic(in_len, out_len, sigma):
    if sigma <= 0.0:
        R = 0
        g = np.array([1.0], dtype=np.float32)
    else:
        R = int(np.ceil(2.0 * sigma))
        x = np.arange(-R, R + 1, dtype=np.float32)
        g = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
        g /= g.sum()
    P = 2 * R + 4
    idx = np.empty((out_len, P), np.int32)
    wt = np.empty((out_len, P), np.float32)
    scale = out_len / in_len
    for i in range(out_len):
        x = (i + 0.5) / scale - 0.5
        left = int(np.floor(x - 2.0)) - R
        js = left + np.arange(P, dtype=np.int32)
        jj = _reflect_idx(js, in_len)
        idx[i, :] = jj
        d = x - js.astype(np.float32)
        acc = np.zeros(P, dtype=np.float32)
        for u in range(-R, R + 1):
            acc += g[u + R] * _cubic(d - u).astype(np.float32)
        ssum = float(acc.sum())
        wt[i, :] = acc * (1.0 / ssum if ssum != 0.0 else 0.0)
    return idx, wt


def imresize_fused_gauss_cubic(img, size, sigma_coeff=0.6, per_axis=False):
    oh, ow = int(size[0]), int(size[1])
    x = img.astype(np.float32, copy=False)
    H, W = x.shape[:2]
    sy = oh / H
    sx = ow / W
    if per_axis:
        sigx = (sigma_coeff / sx) if sx < 1.0 else 0.0
        sigy = (sigma_coeff / sy) if sy < 1.0 else 0.0
    else:
        s = sy if sy < sx else sx
        sigx = sigy = (sigma_coeff / s) if s < 1.0 else 0.0

    idx_x_np, wt_x_np = _precompute_fused_gauss_cubic(W, ow, sigx)
    idx_y_np, wt_y_np = _precompute_fused_gauss_cubic(H, oh, sigy)

    idx_x = cp.asarray(idx_x_np, dtype=cp.int32)
    wt_x = cp.asarray(wt_x_np, dtype=cp.float32)
    idx_y = cp.asarray(idx_y_np, dtype=cp.int32)
    wt_y = cp.asarray(wt_y_np, dtype=cp.float32)

    if x.ndim == 2:
        X = cp.asarray(x, dtype=cp.float32)
        tmp = (X[:, idx_x] * wt_x[None, :, :]).sum(axis=2)
        out = (tmp[idx_y, :] * wt_y[:, :, None]).sum(axis=1)
        y = cp.asnumpy(out)
    else:
        X = cp.asarray(x, dtype=cp.float32)
        tmp = (X[:, idx_x, :] * wt_x[None, :, :, None]).sum(axis=2)
        out = (tmp[idx_y, :, :] * wt_y[:, :, None, None]).sum(axis=1)
        y = cp.asnumpy(out)

    return y.astype(img.dtype, copy=False)
