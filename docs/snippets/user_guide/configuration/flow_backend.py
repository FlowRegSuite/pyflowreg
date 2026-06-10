# Docs page : docs/user_guide/configuration.md ("Available Flow Backends")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationFlowBackend
# Inputs    : none
# Setup (hidden from the docs page; the page imports OFOptions in an earlier block)
from pyflowreg.motion_correction import OFOptions

# [docs:start]
options = OFOptions(
    flow_backend="flowreg",  # Choose flow backend
    backend_params={},  # Backend-specific parameters
)
# [docs:end]
