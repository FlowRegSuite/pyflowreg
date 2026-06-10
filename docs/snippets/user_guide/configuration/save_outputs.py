# Docs page : docs/user_guide/configuration.md ("Saving Displacement Fields")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationSaveOutputs
# Inputs    : none
# Setup (hidden from the docs page; the page imports OFOptions in an earlier block)
from pyflowreg.motion_correction import OFOptions

# [docs:start]
options = OFOptions(
    save_w=True,  # Save displacement fields (default False)
    output_typename="single",  # Output dtype tag (default "double")
)
# [docs:end]
