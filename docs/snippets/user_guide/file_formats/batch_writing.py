# Docs page : docs/user_guide/file_formats.md ("Batch Writing")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsBatchWriting
# Inputs    : none -- frames are generated inline
# [docs:start]
import numpy as np

from pyflowreg.util.io import get_video_file_writer

# Three batches of 10 frames each
video_batches = [np.random.rand(10, 64, 64, 2).astype(np.float32) for _ in range(3)]

writer = get_video_file_writer("output.h5", "HDF5")

# Write in batches
for batch in video_batches:
    writer.write_frames(batch)  # batch: (T, H, W, C)

writer.close()
# [docs:end]
