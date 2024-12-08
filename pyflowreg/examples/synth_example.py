import numpy as np
import h5py
import cv2
import pyflowreg as pfr
from os.path import join, dirname
import os
from pyflowreg.optical_flow import imregister_wrapper


if __name__ == "__main__":
    input_folder = join(dirname(dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    with h5py.File(join(input_folder, "synth_frames.h5"), "r") as f:
        clean = f["clean"][:]
        noisy30db = f["noisy30db"][:]
        noisy35db = f["noisy35db"][:]
        w = f["w"][:]

    frame1 = np.permute_dims(clean[0], (1, 2, 0))
    frame2 = np.permute_dims(clean[1], (1, 2, 0))
    frame1 = cv2.GaussianBlur(frame1, (5, 5), 1)
    frame1 = cv2.normalize(frame1, None, 0, 1, cv2.NORM_MINMAX).astype(np.float64)
    frame2 = cv2.GaussianBlur(frame2, (5, 5), 1)
    frame2 = cv2.normalize(frame2, None, 0, 1, cv2.NORM_MINMAX).astype(np.float64)
    w = np.permute_dims(w, (1, 2, 0))

    w = pfr.get_displacement(
        frame1, frame2, alpha=(250, 250), levels=1,
        iterations=50, update_lag=10, a_data=0.45, a_smooth=1)

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
