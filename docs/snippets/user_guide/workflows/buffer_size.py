# Docs page : docs/user_guide/workflows.md ("Buffer Size Selection")
# Test      : tests/docs/user_guide/test_workflows.py::TestWorkflowsBufferSize
# Inputs    : none
# [docs:start]
from pyflowreg.motion_correction import OFOptions, OutputFormat

# Live display - smaller buffers for more frequent callback updates
options_display = OFOptions(
    buffer_size=10,  # Callbacks fire every 10 frames
    output_format=OutputFormat.NULL,  # Callback-only processing with BatchMotionCorrector
)

# Batch processing - larger buffers
options_batch = OFOptions(
    buffer_size=100,  # Process 100 frames per batch
    output_format=OutputFormat.HDF5,
)
# [docs:end]
