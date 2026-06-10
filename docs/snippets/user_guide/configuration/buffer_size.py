# Docs page : docs/user_guide/configuration.md ("Buffer Size")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationBufferSize
# Inputs    : none
# Setup (hidden from the docs page; the page imports OFOptions in an earlier block)
from pyflowreg.motion_correction import OFOptions

# [docs:start]
options = OFOptions(
    buffer_size=400,  # Number of frames per batch (default 400)
)
# [docs:end]
