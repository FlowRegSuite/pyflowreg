"""
Evaluate pyflowreg on synthetic data with known ground truth. Replicates the
evaluation in the original paper.
"""

import time

import h5py
import numpy as np
import pyflowreg as pfr
from scipy.ndimage import gaussian_filter
from pyflowreg.util.download import download_demo_data


def preprocess(frame):
    frame1 = frame[0]
    frame2 = frame[1]
    frame1 = np.permute_dims(frame1, (1, 2, 0)).astype(float)
    frame2 = np.permute_dims(frame2, (1, 2, 0)).astype(float)

    mins = frame1.min(axis=(0, 1))[None, None, :]  # shape (1,1,C)
    maxs = frame1.max(axis=(0, 1))[None, None, :]  # shape (1,1,C)

    ranges = maxs - mins

    frame1 = (frame1 - mins) / ranges
    frame2 = (frame2 - mins) / ranges

    return frame1, frame2


def epe(gt, est, b=25):
    gt_c = gt[b:-b, b:-b, :2]
    est_c = est[b:-b, b:-b, :2]
    return float(np.mean(np.linalg.norm(gt_c - est_c, axis=-1)))


def run(flow_params, f1, f2):
    t0 = time.perf_counter()
    w = pfr.get_displacement(f1, f2, **flow_params)
    t = time.perf_counter() - t0
    print(f"Elapsed time is {t:.6f} seconds.")
    return w


if __name__ == "__main__":
    # Download data to data/ folder (default location)
    data_file = download_demo_data("synth_frames.h5")

    with h5py.File(data_file, "r") as f:
        clean = f["clean"][:]
        n35 = f["noisy35db"][:]
        n30 = f["noisy30db"][:]
        w_ref = np.moveaxis(f["w"][:], 0, -1)
        w_ref = w_ref[..., ::-1]

    datasets = [("clean", clean), ("noise35db", n35), ("noise30db", n30)]
    b = 25

    cold_f1 = np.zeros((32, 32, 2), np.float32)
    pfr.get_displacement(
        cold_f1,
        cold_f1,
        alpha=(2, 2),
        levels=50,
        min_level=5,
        iterations=50,
        a_data=0.45,
        a_smooth=1,
        weight=np.array([0.6, 0.4]),
    )

    base_params = dict(
        alpha=(8, 8),
        iterations=100,
        a_data=0.45,
        a_smooth=1.0,
        weight=np.array([0.5, 0.5], np.float32),
        levels=50,
        eta=0.8,
        update_lag=5,
    )

    for name, data in datasets:
        data = gaussian_filter(data, (0.00001, 0.00001, 1, 1), truncate=4)

        f1, f2 = preprocess(data)

        w = run({**base_params, "min_level": 0}, f1, f2)
        print(f"{epe(w_ref, w, b):.2f} for {name}, default, ch 1 + 2")

        w = run({**base_params, "min_level": 5}, f1, f2)
        print(f"{epe(w_ref, w, b):.2f} for {name}, fast, ch 1 + 2")

        for ch in (0, 1):
            f1c = f1[..., ch : ch + 1]
            f2c = f2[..., ch : ch + 1]

            w = run({**base_params, "min_level": 0}, f1c, f2c)
            print(f"{epe(w_ref, w, b):.2f} for {name}, default, ch {ch+1}")

            w = run({**base_params, "min_level": 5}, f1c, f2c)
            print(f"{epe(w_ref, w, b):.2f} for {name}, fast, ch {ch+1}")
