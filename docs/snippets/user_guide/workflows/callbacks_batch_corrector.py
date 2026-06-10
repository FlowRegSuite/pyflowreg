# Docs page : docs/user_guide/workflows.md ("Callbacks in the File-Based Workflow")
# Test      : tests/docs/user_guide/test_workflows.py::TestWorkflowsCallbacksBatchCorrector
# Inputs    : raw_video.h5 -- created by the test harness
#
# Page context (not rendered): the monitor_motion / collect_frames callbacks
# are defined as in "Basic Callback Usage"; for the file-based workflow,
# options points the pipeline at an input file as in "File-Based Workflow".
# [docs:start]
import numpy as np
from pyflowreg.motion_correction import BatchMotionCorrector, OFOptions

options = OFOptions(
    input_file="raw_video.h5",
    output_path="results/",
    output_format="HDF5",
    quality_setting="balanced",
    reference_frames=list(range(10, 21)),
)

motion_per_frame = []


def monitor_motion(w_batch, start_idx, end_idx):
    """Monitor displacement fields during processing."""
    for t in range(w_batch.shape[0]):
        magnitude = np.sqrt(w_batch[t, :, :, 0] ** 2 + w_batch[t, :, :, 1] ** 2)
        motion_per_frame.append(np.mean(magnitude))


corrected_batches = []


def collect_frames(batch, start_idx, end_idx):
    """Collect corrected frames as they're computed, e.g. for live display."""
    corrected_batches.append(batch.copy())


corrector = BatchMotionCorrector(options)
corrector.register_progress_callback(lambda current, total: print(f"{current}/{total}"))
corrector.register_w_callback(monitor_motion)
corrector.register_registered_callback(collect_frames)
corrector.run()
# [docs:end]
