from pyflowreg.util.io.mdf import MDFFileReader
from pyflowreg.util.io.hdf5 import HDF5FileWriter
import cv2


if __name__ == "__main__":
    input_file = "D:\\2025_OIST\\Shinobu\\RFPonly\\190403_001.MDF"
    file_reader = MDFFileReader(input_file, buffer_size=1000, bin_size=1)

    print(f"Number of frames: {len(file_reader)}")

    idx = range(8000, 10000, 10)

    for i in idx:
        frame = file_reader[i]
        cv2.imshow("Frame", cv2.normalize(frame[:, :, 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.waitKey(1)