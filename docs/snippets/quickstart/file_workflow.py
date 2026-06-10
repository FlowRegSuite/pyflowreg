# Docs page : docs/quickstart.md ("File-Based Workflow")
# Test      : tests/docs/test_quickstart.py::TestQuickstartFileWorkflow
# Inputs    : my_video.h5 -- created by the test harness
# [docs:start]
from pyflowreg.motion_correction import compensate_recording, OFOptions

# Configure with input/output paths
options = OFOptions(
    input_file="my_video.h5",
    output_path="results/",
    output_format="HDF5",
    quality_setting="balanced",
    reference_frames=list(range(10, 21)),  # Frames 10-20 as reference
    save_w=True,  # Save displacement fields
)

# Run compensation (auto-selects parallelization executor)
compensate_recording(options)
# [docs:end]
