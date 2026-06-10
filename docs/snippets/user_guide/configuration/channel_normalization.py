# Docs page : docs/user_guide/configuration.md ("Channel Normalization")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationChannelNormalization
# Inputs    : none
# Setup (hidden from the docs page; the page imports OFOptions in an earlier block)
from pyflowreg.motion_correction import OFOptions

# [docs:start]
# Default: normalize all channels together
options = OFOptions(channel_normalization="joint")

# Or normalize each channel independently
options = OFOptions(channel_normalization="separate")
# [docs:end]
