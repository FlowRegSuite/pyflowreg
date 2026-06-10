# Docs page : docs/user_guide/file_formats.md ("Context Manager")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsWriterContextManager
# Inputs    : none -- frames are generated inline
# [docs:start]
import numpy as np

from pyflowreg.util.io import get_video_file_writer

frames = np.random.rand(10, 64, 64, 2).astype(np.float32)  # (T, H, W, C)

with get_video_file_writer("output.h5", "HDF5") as writer:
    writer.write_frames(frames)
# Automatically closed
# [docs:end]
