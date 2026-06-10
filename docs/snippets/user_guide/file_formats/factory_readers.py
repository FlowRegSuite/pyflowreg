# Docs page : docs/user_guide/file_formats.md ("Creating Readers")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsFactoryReaders
# Inputs    : video.h5, ch1.tif, ch2.tif -- created by the test harness
# [docs:start]
from pyflowreg.util.io import get_video_file_reader

# Automatic format detection from extension
file_reader = get_video_file_reader("video.h5", buffer_size=500, bin_size=1)
video_array = file_reader[:]  # (T, H, W, C)
file_reader.close()

# Numpy array input
array_reader = get_video_file_reader(video_array)

# Multi-channel from separate files
multichannel_reader = get_video_file_reader(["ch1.tif", "ch2.tif"])
# [docs:end]
