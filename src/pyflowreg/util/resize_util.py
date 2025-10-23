import cv2
import numpy as np
from numba import njit
from skimage.transform import resize as ski_resize

A = -0.75


def resize_image_skimage(image, size):
    """
    Resize an image to the specified size using the given interpolation method.

    Parameters:
    - image: Input image to be resized.
    - size: Tuple (width, height) specifying the new size.
    - interpolation: Interpolation method to use for resizing.

    Returns:
    - Resized image.
    """
    img = ski_resize(
        image, (size[1], size[0]), order=3, mode="edge", anti_aliasing=True
    )
    return img


def resize_image_cv2(image, size):
    """
    Resize an image to the specified size using the given interpolation method.

    Parameters:
    - image: Input image to be resized.
    - size: Tuple (width, height) specifying the new size.
    - interpolation: Interpolation method to use for resizing.

    Returns:
    - Resized image.
    """
    img = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LANCZOS4)
    return img


def resize_image_cv2_aa_gauss(image, size):
    h_orig, w_orig = image.shape[:2]
    h_new, w_new = size[:2]
    scale = min(h_new / h_orig, w_new / w_orig)

    if scale < 1:
        sigma = 0.6 / scale
        ksize = int(2 * np.ceil(2 * sigma) + 1)
        image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    img = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return img


def imresize_cv2_aa_gauss_sep(img, size):
    h, w = img.shape[:2]
    sx, sy = size[1] / w, size[0] / h
    scale = min(sx, sy)
    if scale < 1:
        sigma = 0.6 / scale
        k = int(2 * np.ceil(2 * sigma) + 1)
        g = cv2.getGaussianKernel(k, sigma)
        img = cv2.sepFilter2D(img, -1, g, g, borderType=cv2.BORDER_REFLECT101)
    return cv2.resize(img, size[1::-1], interpolation=cv2.INTER_CUBIC)


@njit(inline="always")
def _cubic(x):
    ax = abs(x)
    if ax < 1.0:
        return (A + 2.0) * ax**3 - (A + 3.0) * ax**2 + 1.0
    elif ax < 2.0:
        return A * ax**3 - 5.0 * A * ax**2 + 8.0 * A * ax - 4.0 * A
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def _fill_tables(idx, wt, in_len, out_len, s, w, scaled):
    P = idx.shape[1]
    for i in range(out_len):
        x = (i + 0.5) / s - 0.5
        left = int(np.floor(x - 0.5 * w))
        sumw = 0.0
        for p in range(P):
            j = left + p
            jj = 0 if j < 0 else (in_len - 1 if j > in_len - 1 else j)
            idx[i, p] = jj
            d = x - j
            wgt = s * _cubic(s * d) if scaled else _cubic(d)
            wt[i, p] = wgt
            sumw += wgt
        inv = 1.0 / sumw
        for p in range(P):
            wt[i, p] *= inv


def _precompute(in_len, out_len):
    scale = out_len / in_len
    if scale < 1.0:
        W = 4.0 / scale
        shrink = True
    else:
        W = 4.0
        shrink = False

    P = int(np.ceil(W)) + 1
    idx = np.empty((out_len, P), np.int32)
    wt = np.empty((out_len, P), np.float32)

    _fill_tables(idx, wt, in_len, out_len, scale, W, shrink)
    return idx, wt


@njit(fastmath=True, cache=True)
def _resize_h(src, idx, wt):
    h = src.shape[0]
    ow, P = wt.shape
    dst = np.empty((h, ow), src.dtype)
    for y in range(h):
        for x in range(ow):
            s = 0.0
            for p in range(P):
                s += src[y, idx[x, p]] * wt[x, p]
            dst[y, x] = s
    return dst


@njit(fastmath=True, cache=True)
def _resize_v(src, idx, wt):
    w = src.shape[1]
    oh, P = wt.shape
    dst = np.empty((oh, w), src.dtype)
    for x in range(w):
        for y in range(oh):
            s = 0.0
            for p in range(P):
                s += src[idx[y, p], x] * wt[y, p]
            dst[y, x] = s
    return dst


def imresize_numba(img, size):
    oh, ow = size[:2]

    work = img.astype(np.float32, copy=False)

    idx_x, wt_x = _precompute(work.shape[1], ow)
    idx_y, wt_y = _precompute(work.shape[0], oh)

    if work.ndim == 2:
        out = np.empty((oh, ow), np.float32)
        tmp = _resize_h(work, idx_x, wt_x)
        out[:] = _resize_v(tmp, idx_y, wt_y)

    else:
        ch = work.shape[2]
        out = np.empty((oh, ow, ch), np.float32)
        for c in range(ch):  # 2-D slice per kernel call
            tmp = _resize_h(work[:, :, c], idx_x, wt_x)
            out[:, :, c] = _resize_v(tmp, idx_y, wt_y)

    return out.astype(img.dtype, copy=False)


def resize_image_cv2_aa_blur(image, size):
    h_orig, w_orig = image.shape[:2]
    h_new, w_new = size[:2]

    if h_new < h_orig or w_new < w_orig:
        # Box filter - much faster than Gaussian, still removes aliasing
        scale = max(h_orig / h_new, w_orig / w_new)
        ksize = int(scale) | 1  # Ensure odd
        if ksize > 1:
            image = cv2.blur(image, (ksize, ksize))

    img = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return img


@njit(inline="always")
def _reflect_idx(j, n):
    if n <= 1:
        return 0
    while j < 0 or j >= n:
        if j < 0:
            j = -j - 1
        else:
            j = 2 * n - 1 - j
    return j


@njit(fastmath=True, cache=True)
def _fill_tables_fused_gauss_cubic_reflect(idx, wt, in_len, out_len, scale, g, R):
    P = 2 * R + 4
    for i in range(out_len):
        x = (i + 0.5) / scale - 0.5
        left = int(np.floor(x - 2.0)) - R
        ssum = 0.0
        for p in range(P):
            j = left + p
            jj = _reflect_idx(j, in_len)
            idx[i, p] = jj
            d = x - j
            acc = 0.0
            for u in range(-R, R + 1):
                acc += g[u + R] * _cubic(d - u)
            wt[i, p] = acc
            ssum += acc
        inv = 1.0 / ssum
        for p in range(P):
            wt[i, p] *= inv


def _precompute_fused_gauss_cubic(in_len, out_len, sigma):
    scale = out_len / in_len
    if sigma <= 0.0:
        R = 0
        g = np.array([1.0], dtype=np.float32)
    else:
        R = int(np.ceil(2.0 * sigma))
        x = np.arange(-R, R + 1, dtype=np.float32)
        g = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
        g /= g.sum()
    idx = np.empty((out_len, 2 * R + 4), np.int32)
    wt = np.empty((out_len, 2 * R + 4), np.float32)
    _fill_tables_fused_gauss_cubic_reflect(idx, wt, in_len, out_len, scale, g, R)
    return idx, wt


def imresize_fused_gauss_cubic(img, size, sigma_coeff=0.6, per_axis=False):
    oh, ow = size[:2]
    x = img.astype(np.float32, copy=False)
    sy = oh / x.shape[0]
    sx = ow / x.shape[1]
    if per_axis:
        sigx = sigma_coeff / sx if sx < 1.0 else 0.0
        sigy = sigma_coeff / sy if sy < 1.0 else 0.0
    else:
        s = sy if sy < sx else sx
        sigx = sigy = (sigma_coeff / s) if s < 1.0 else 0.0
    idx_x, wt_x = _precompute_fused_gauss_cubic(x.shape[1], ow, sigx)
    idx_y, wt_y = _precompute_fused_gauss_cubic(x.shape[0], oh, sigy)
    if x.ndim == 2:
        tmp = _resize_h(x, idx_x, wt_x)
        y = _resize_v(tmp, idx_y, wt_y)
    else:
        c = x.shape[2]
        y = np.empty((oh, ow, c), np.float32)
        for k in range(c):
            tmp = _resize_h(x[:, :, k], idx_x, wt_x)
            y[:, :, k] = _resize_v(tmp, idx_y, wt_y)
    return y.astype(img.dtype, copy=False)
