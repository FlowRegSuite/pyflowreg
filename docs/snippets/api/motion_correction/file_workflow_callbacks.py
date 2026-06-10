# Docs page : docs/api/motion_correction.md ("File-Based Workflow")
# Test      : tests/docs/api/test_motion_correction.py::TestMotionCorrectionFileWorkflow
# Inputs    : recording.h5 -- created by the test harness
# [docs:start]
from pyflowreg.motion_correction.compensate_recording import BatchMotionCorrector
from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat


class ProcessingMonitor:
    def __init__(self):
        self.batch_count = 0

    def on_batch_complete(self, batch, start_idx, end_idx):
        self.batch_count += 1
        print(f"Batch {self.batch_count} complete: frames {start_idx}-{end_idx}")


monitor = ProcessingMonitor()

options = OFOptions(
    input_file="recording.h5",
    output_format=OutputFormat.HDF5,
    output_path="results/",
    save_w=True,
)

compensator = BatchMotionCorrector(options)
compensator.register_registered_callback(monitor.on_batch_complete)
compensator.run()
# [docs:end]
