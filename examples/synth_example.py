import numpy as np
import h5py
import cv2
import pyflowreg as pfr
from os.path import join, dirname
import os
from pyflowreg.core.optical_flow import imregister_wrapper
from time import time


if __name__ == "__main__":
    input_folder = join(dirname(dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    with h5py.File(join(input_folder, "synth_frames.h5"), "r") as f:
        #clean = f["clean"][:]
        clean = f[("clean")][:]
        noisy35db = f["noisy35db"][:]
        clean = f[("noisy30db")][:]

        w = f["w"][:]

    frame1 = np.permute_dims(clean[0], (1, 2, 0)).astype(float)
    frame2 = np.permute_dims(clean[1], (1, 2, 0)).astype(float)
    frame1 = cv2.GaussianBlur(frame1, None, 1.5)
    frame2 = cv2.GaussianBlur(frame2, None, 1.5)

    eps = 1e-6
    mins = frame1.min(axis=(0, 1))[None, None, :]  # shape (1,1,C)
    maxs = frame1.max(axis=(0, 1))[None, None, :]  # shape (1,1,C)

    ranges = maxs - mins
    ranges[ranges < eps] = 1.0

    frame1 = (frame1 - mins) / ranges
    frame2 = (frame2 - mins) / ranges


    #min_ref = frame1.min((0, 1))[None, None]
    #max_ref = frame1.max((0, 1))[None, None]

    #frame1 = (frame1 - min_ref) / (max_ref - min_ref)
    #frame2 = (frame2 - min_ref) / (max_ref - min_ref)

    #for i in range(2):
    #    frame1[:, :, i] = cv2.normalize(frame1[..., i], None, 0, 1, cv2.NORM_MINMAX)
    #    frame2[:, :, i] = cv2.normalize(frame2[..., i], None, 0, 1, cv2.NORM_MINMAX)
    w = np.permute_dims(w, (1, 2, 0))

    print(frame1.shape)

    start = time()
    w = pfr.get_displacement(
        frame1[..., :], frame2[..., :], alpha=(2, 2), levels=50, min_level=5,
        iterations=50, a_data=0.45, a_smooth=1, weight=np.array([0.6, 0.4]))
    print("Elapsed time:", time() - start)

    times = []
    for i in range(2):
        start = time()
        w = pfr.get_displacement(
            frame1[..., :], frame2[..., :],
            alpha=(8, 8),
            levels=50,
            min_level=0,
            iterations=50,
            a_data=0.45,
            a_smooth=1,
            weight=np.array([0.6, 0.4])
        )
        elapsed = time() - start
        if i > 0:
            times.append(elapsed)

    print("Average elapsed time over 5 runs:", sum(times) / len(times))

    print(w.shape)

    img1 = w[..., 0]
    img2 = w[..., 1]
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)

    print(frame1.shape)

    warped_frame2 = imregister_wrapper(frame2, w[..., 0], w[..., 1], frame1, interpolation_method='cubic')

    warped_display = cv2.normalize(warped_frame2[..., 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    frame1_display = cv2.normalize(frame1[..., 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    frame2_display = cv2.normalize(frame2[..., 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow("Frame1 (Reference)", frame1_display)
    cv2.imshow("Frame2 (Original)", frame2_display)
    cv2.imshow("Warped Frame2", warped_display)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
