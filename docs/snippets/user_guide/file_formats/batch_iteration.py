# Docs page : docs/user_guide/file_formats.md ("Batch Iteration")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsBatchIteration
# Inputs    : video.h5 -- created by the test harness
# [docs:start]
from pyflowreg.util.io import get_video_file_reader

reader = get_video_file_reader("video.h5", buffer_size=100)

n_frames = 0
for batch in reader:
    # batch shape: (100, H, W, C) or fewer for the last batch
    n_frames += batch.shape[0]

reader.close()
# [docs:end]
