# Docs page : docs/user_guide/file_formats.md ("In-Memory Output")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsMemoryWriters
# Inputs    : none -- frames are generated inline
# [docs:start]
import numpy as np

from pyflowreg.util.io import get_video_file_writer

frames = np.random.rand(5, 64, 64, 2).astype(np.float32)  # (T, H, W, C)

# ARRAY: accumulate frames in memory
array_writer = get_video_file_writer("", "ARRAY")
array_writer.write_frames(frames)
video = array_writer.get_array()  # (5, 64, 64, 2)

# NULL: discard frames (callback-only processing)
null_writer = get_video_file_writer("", "NULL")
null_writer.write_frames(frames)
# [docs:end]
