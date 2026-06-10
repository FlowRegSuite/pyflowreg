# Docs page : docs/user_guide/configuration.md ("Temporal Binning")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationTemporalBinning
# Inputs    : none
# Setup (hidden from the docs page; the page imports OFOptions in an earlier block)
from pyflowreg.motion_correction import OFOptions

# [docs:start]
options = OFOptions(
    bin_size=2,  # Average every 2 consecutive frames together (default 1, no binning)
)
# [docs:end]
