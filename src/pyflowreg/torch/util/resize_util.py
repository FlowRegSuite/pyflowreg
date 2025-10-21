import numpy as np
import torch


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


def imresize_fused_gauss_cubic(img, size, sigma_coeff=0.6, per_axis=False, device=None):
    oh, ow = int(size[0]), int(size[1])
    x_np = img.astype(np.float32, copy=False)
    H, W = x_np.shape[:2]
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

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    idx_x = torch.from_numpy(idx_x_np.astype(np.int64)).to(device)
    wt_x = torch.from_numpy(wt_x_np).to(device=device, dtype=dtype)
    idx_y = torch.from_numpy(idx_y_np.astype(np.int64)).to(device)
    wt_y = torch.from_numpy(wt_y_np).to(device=device, dtype=dtype)

    if x_np.ndim == 2:
        X = torch.from_numpy(x_np).to(device=device, dtype=dtype).unsqueeze(-1)  # H W 1
    else:
        X = torch.from_numpy(x_np).to(device=device, dtype=dtype)  # H W C
    Ht, Wt, C = X.shape

    # Horizontal pass: gather P taps for each of ow outputs in one shot
    P = idx_x.shape[1]
    X1 = X.unsqueeze(1).expand(Ht, ow, Wt, C)  # H, ow, W, C
    idxX = idx_x.view(1, ow, P, 1).expand(Ht, ow, P, C)  # H, ow, P, C
    taps = X1.gather(2, idxX)  # H, ow, P, C
    tmp = (taps * wt_x.view(1, ow, P, 1)).sum(dim=2)  # H, ow, C

    # Vertical pass
    P2 = idx_y.shape[1]
    T1 = tmp.unsqueeze(0).expand(oh, Ht, ow, C)  # oh, H, ow, C
    idxY = idx_y.view(oh, P2, 1, 1).expand(oh, P2, ow, C)  # oh, P, ow, C
    taps_y = T1.gather(1, idxY)  # oh, P, ow, C
    out = (taps_y * wt_y.view(oh, P2, 1, 1)).sum(dim=1)  # oh, ow, C

    y = out.squeeze(-1).contiguous().to("cpu").numpy()
    return y.astype(img.dtype, copy=False)
