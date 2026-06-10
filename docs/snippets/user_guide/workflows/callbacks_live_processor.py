# Docs page : docs/user_guide/workflows.md ("Using Callbacks")
# Test      : tests/docs/user_guide/test_workflows.py::TestWorkflowsCallbacksLiveProcessor
# Inputs    : my_video.h5 -- created by the test harness
#
# Page context (not rendered): video and reference come from the
# "Array-Based Workflow" example earlier on the page.
import numpy as np
from pyflowreg.motion_correction import compensate_arr, OFOptions
from pyflowreg.util.io import get_video_file_reader

reader = get_video_file_reader("my_video.h5")
video = reader[:]
reader.close()
reference = np.mean(video[10:21], axis=0)


# [docs:start]
class LiveProcessor:
    def __init__(self):
        self.statistics = []

    def process_batch(self, w_batch, start_idx, end_idx):
        # Process displacement fields during registration
        batch_stats = {
            "start": start_idx,
            "end": end_idx,
            "mean_motion": np.mean(
                np.sqrt(w_batch[..., 0] ** 2 + w_batch[..., 1] ** 2)
            ),
        }
        self.statistics.append(batch_stats)


processor = LiveProcessor()

# compensate_arr returns arrays but also calls callbacks
registered, w = compensate_arr(
    video,
    reference,
    options=OFOptions(quality_setting="balanced"),
    w_callback=processor.process_batch,
)

# Access collected statistics
print(f"Processed {len(processor.statistics)} batches")
# [docs:end]
