# Docs page : docs/user_guide/configuration.md ("Gaussian Filtering")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationGaussianFiltering
# Inputs    : none
# Setup (hidden from the docs page; the page imports OFOptions in an earlier block)
from pyflowreg.motion_correction import OFOptions

# [docs:start]
# Sigma values: [sx, sy, st] (spatial x, spatial y, temporal),
# one triple applied to all channels
options = OFOptions(sigma=[1.0, 1.0, 0.1])

# Or per-channel, here for 2 channels
options = OFOptions(sigma=[[1.0, 1.0, 0.1], [2.0, 2.0, 0.2]])
# [docs:end]
