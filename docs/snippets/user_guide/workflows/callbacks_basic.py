# Docs page : docs/user_guide/workflows.md ("Basic Callback Usage")
# Test      : tests/docs/user_guide/test_workflows.py::TestWorkflowsCallbacksBasic
# Inputs    : my_video.h5 -- created by the test harness
#
# Page context (not rendered): video, reference and options come from the
# "Array-Based Workflow" example earlier on the page.
import numpy as np
from pyflowreg.motion_correction import compensate_arr, OFOptions
from pyflowreg.util.io import get_video_file_reader

reader = get_video_file_reader("my_video.h5")
video = reader[:]
reader.close()
reference = np.mean(video[10:21], axis=0)
options = OFOptions(quality_setting="balanced")
# [docs:start]
motion_per_frame = []


def monitor_motion(w_batch, start_idx, end_idx):
    """Monitor displacement fields during processing."""
    for t in range(w_batch.shape[0]):
        magnitude = np.sqrt(w_batch[t, :, :, 0] ** 2 + w_batch[t, :, :, 1] ** 2)
        motion_per_frame.append(np.mean(magnitude))
        print(f"Frame {start_idx + t}: mean motion = {motion_per_frame[-1]:.2f}")


corrected_batches = []


def collect_frames(batch, start_idx, end_idx):
    """Collect corrected frames as they're computed, e.g. for live display."""
    corrected_batches.append(batch.copy())


# Use callbacks during processing
registered, w = compensate_arr(
    video,
    reference,
    options=options,
    w_callback=monitor_motion,
    registered_callback=collect_frames,
)
# [docs:end]
