import numpy as np
import h5py
import cv2
import pyflowreg as pfr
from os.path import join, dirname
import os


if __name__ == "__main__":
    input_folder = join(dirname(dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    with h5py.File(join(input_folder, "synth_frames.h5"), "r") as f:
        clean = f["clean"][:]
        noisy30db = f["noisy30db"][:]
        noisy35db = f["noisy35db"][:]
        w = f["w"][:]

    print(clean.shape)

    frame1 = np.permute_dims(clean[0], (1, 2, 0))
    frame2 = np.permute_dims(clean[1], (1, 2, 0))

    u, v = pfr.get_displacement(
        frame1, frame2, alpha=(2, 2),
        iterations=20, update_lag=10, a_data=0.45, a_smooth=0.5)

    print(frame1.shape)

    pass

    # Create or load two frames as numpy arrays
    # For this example, let's just create synthetic frames.
    # Suppose m=64, n=64, single channel
    m, n = 64, 64
    fixed = np.zeros((m, n), dtype=np.float64)
    moving = np.zeros((m, n), dtype=np.float64)

    # Introduce a simple synthetic shift in 'moving' frame
    # e.g., shift one pixel to the right
    moving[:, 1:] = 1.0

    # Compute displacement field
    du, dv = get_displacement(
        fixed,
        moving,
        alpha=(2,2),
        iterations=20,
        update_lag=10,
        a_data=0.45,
        a_smooth=0.5,
        hx=1.0,
        hy=1.0
    )

    # Print or analyze the results
    # du, dv are displacement fields in the vertical (u) and horizontal (v) directions respectively
    magnitude = np.sqrt(du**2 + dv**2)
    print("Displacement magnitude:", magnitude.mean())
    print("U-field mean:", du.mean())
    print("V-field mean:", dv.mean())
