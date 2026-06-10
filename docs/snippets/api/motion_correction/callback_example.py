# Docs page : docs/api/motion_correction.md ("Example with Callbacks")
# Test      : tests/docs/api/test_motion_correction.py::TestMotionCorrectionCallbackExample
# Inputs    : recording.tif -- created by the test harness
# [docs:start]
import numpy as np

from pyflowreg.motion_correction import compensate_arr
from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.util.io import get_video_file_reader


def track_motion(w_batch, start_idx, end_idx):
    """Process displacement fields as they're computed."""
    for i in range(w_batch.shape[0]):
        magnitude = np.sqrt(w_batch[i, :, :, 0] ** 2 + w_batch[i, :, :, 1] ** 2)
        print(f"Frame {start_idx + i}: mean motion = {np.mean(magnitude):.2f}")


# Read the input video and average its first frames as reference
reader = get_video_file_reader("recording.tif")
video = reader[:]  # (T, H, W, C)
reader.close()
reference = np.mean(video[:10], axis=0)

# Configure array workflow
options = OFOptions(
    quality_setting="balanced",
    buffer_size=20,  # Process 20 frames at a time
)

# Run with callbacks
registered, w = compensate_arr(
    video,
    reference,
    options,
    w_callback=track_motion,
)
# [docs:end]
