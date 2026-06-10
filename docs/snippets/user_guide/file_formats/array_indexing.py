# Docs page : docs/user_guide/file_formats.md ("Array-Like Indexing")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsArrayIndexing
# Inputs    : video.h5 -- created by the test harness
# [docs:start]
from pyflowreg.util.io import get_video_file_reader

reader = get_video_file_reader("video.h5")

# Single frame: returns (H, W, C)
frame = reader[0]

# Slice: returns (T, H, W, C)
frames = reader[10:20]

# List indexing: returns (T, H, W, C)
frames = reader[[0, 10, 20, 30]]

# Spatial subset: the first element indexes frames,
# the remaining elements index into (H, W, C)
frames = reader[0:10, 100:200, 100:200, :]
# [docs:end]
