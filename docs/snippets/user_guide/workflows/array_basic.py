# Docs page : docs/user_guide/workflows.md ("Array-Based Workflow")
# Test      : tests/docs/user_guide/test_workflows.py::TestWorkflowsArrayBasic
# Inputs    : my_video.h5 -- created by the test harness
# [docs:start]
import numpy as np
from pyflowreg.motion_correction import compensate_arr, OFOptions
from pyflowreg.util.io import get_video_file_reader

# Load video data (T, H, W, C)
reader = get_video_file_reader("my_video.h5")
video = reader[:]
reader.close()

# Create reference from stable frames
reference = np.mean(video[10:21], axis=0)

# Configure motion correction
options = OFOptions(quality_setting="balanced")

# Run compensation
registered, flow = compensate_arr(video, reference, options)
# [docs:end]
