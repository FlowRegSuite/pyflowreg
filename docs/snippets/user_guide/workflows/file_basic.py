# Docs page : docs/user_guide/workflows.md ("File-Based Workflow")
# Test      : tests/docs/user_guide/test_workflows.py::TestWorkflowsFileBasic
# Inputs    : raw_video.h5 -- created by the test harness
# [docs:start]
from pyflowreg.motion_correction import compensate_recording, OFOptions

# Configure with file paths
options = OFOptions(
    input_file="raw_video.h5",
    output_path="results/",
    output_format="HDF5",
    quality_setting="balanced",
    reference_frames=list(range(10, 21)),
    save_w=True,  # Save displacement fields
)

# Run compensation (auto-selects parallelization)
compensate_recording(options)
# [docs:end]
