# Docs page : docs/user_guide/parallelization.md ("Buffer Size")
# Test      : tests/docs/user_guide/test_parallelization.py::TestParallelizationBufferSize
# Inputs    : none
# [docs:start]
from pyflowreg.motion_correction import OFOptions

options = OFOptions(
    buffer_size=400,  # Frames per batch (default: 400)
    # ... other options
)
# [docs:end]
