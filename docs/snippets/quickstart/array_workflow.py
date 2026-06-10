# Docs page : docs/quickstart.md ("Basic Array-Based Workflow")
# Test      : tests/docs/test_quickstart.py::TestQuickstartArrayWorkflow
# Inputs    : my_video.tif -- created by the test harness
# [docs:start]
import numpy as np
from pyflowreg.motion_correction import compensate_arr, OFOptions
from pyflowreg.util.io import get_video_file_reader

# Load video using PyFlowReg's video readers
reader = get_video_file_reader("my_video.tif")
video = reader[:]  # Read all frames (T, H, W, C)
reader.close()

# Create reference from frames 10-20
reference = np.mean(video[10:21], axis=0)

# Configure motion correction
options = OFOptions(
    alpha=4,
    quality_setting="balanced",
)

# Run compensation - returns registered video and flow fields
registered, flow = compensate_arr(video, reference, options)
# [docs:end]
