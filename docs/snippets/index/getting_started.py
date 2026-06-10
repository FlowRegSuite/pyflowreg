# Docs page : docs/index.md ("Getting Started")
# Test      : tests/docs/test_index.py::TestIndexGettingStarted
# Inputs    : my_video.tif -- created by the test harness
# [docs:start]
import numpy as np
from pyflowreg.motion_correction import compensate_arr, OFOptions
from pyflowreg.util.io import get_video_file_reader

# Load video using PyFlowReg's video readers
reader = get_video_file_reader("my_video.tif")
video = reader[:]
reader.close()

# Create reference from frames 10-20
reference = np.mean(video[10:21], axis=0)

# Configure and run motion correction
options = OFOptions(quality_setting="balanced")
registered, flow = compensate_arr(video, reference, options)
# [docs:end]
