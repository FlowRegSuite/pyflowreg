# Docs page : docs/user_guide/file_formats.md ("HDF5 Optimization")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsHdf5WriterOptions
# Inputs    : none -- frames are generated inline
# [docs:start]
import numpy as np

from pyflowreg.util.io import get_video_file_writer

frames = np.random.rand(10, 64, 64, 1).astype(np.float32)  # (T, H, W, C)

# Use compression for storage savings
with get_video_file_writer(
    "compressed.h5", "HDF5", compression="gzip", compression_level=4
) as writer:
    writer.write_frames(frames)

# chunk_size sets the number of frames per chunk (default 1)
with get_video_file_writer("chunked.h5", "HDF5", chunk_size=10) as writer:
    writer.write_frames(frames)
# [docs:end]
