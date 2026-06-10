# Docs page : docs/user_guide/configuration.md ("Updating the Reference Frame")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationUpdateReference
# Inputs    : none
# Setup (hidden from the docs page; the page imports OFOptions in an earlier block)
from pyflowreg.motion_correction import OFOptions

# [docs:start]
options = OFOptions(
    update_reference=True,  # default False
)
# [docs:end]
