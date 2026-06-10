# Docs page : docs/user_guide/configuration.md ("Quality Settings")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationQualitySettings
# Inputs    : none
# [docs:start]
from pyflowreg.motion_correction import OFOptions

# Fast preview (stops at pyramid level 6)
options = OFOptions(quality_setting="fast")

# Balanced quality (stops at pyramid level 4)
options = OFOptions(quality_setting="balanced")

# Maximum quality (solves down to level 0, full resolution; the default)
options = OFOptions(quality_setting="quality")
# [docs:end]
