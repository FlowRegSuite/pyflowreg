# Docs page : docs/user_guide/file_formats.md ("Multi-File Formats")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsMultifileWriter
# Inputs    : none -- frames are generated inline
# [docs:start]
import numpy as np

from pyflowreg.util.io import get_video_file_writer

frames = np.random.rand(8, 32, 32, 2).astype(np.float32)  # (T, H, W, C)

with get_video_file_writer("multifile/compensated.h5", "MULTIFILE_HDF5") as writer:
    writer.write_frames(frames)

# Creates multifile/compensated_ch1.HDF5 and multifile/compensated_ch2.HDF5
# [docs:end]
