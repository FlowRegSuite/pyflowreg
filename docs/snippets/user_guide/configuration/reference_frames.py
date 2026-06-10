# Docs page : docs/user_guide/configuration.md ("Fixed Reference Frames")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationReferenceFrames
# Inputs    : raw_video.h5 -- created by the test harness
# [docs:start]
import numpy as np

from pyflowreg.motion_correction import OFOptions
from pyflowreg.util.io import get_video_file_reader

# Single frame
options = OFOptions(reference_frames=[0])

# Frame indices: the frames are preregistered and averaged
options = OFOptions(reference_frames=list(range(100, 200)))

# Load from an image file (TIFF)
options = OFOptions(reference_frames="reference.tif")

# Provide a precomputed reference directly as a numpy array
reader = get_video_file_reader("raw_video.h5")
reference_array = np.mean(reader[10:21], axis=0)  # (H, W, C) mean image
reader.close()
options = OFOptions(reference_frames=reference_array)
# [docs:end]
