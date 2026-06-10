# Docs page : docs/user_guide/configuration.md ("Saving and Loading Configuration")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationSaveLoad
# Inputs    : none -- config.json is written by the snippet itself
# [docs:start]
from pyflowreg.motion_correction import OFOptions

options = OFOptions(quality_setting="balanced", alpha=2.0)

# Save configuration to JSON
options.save_options("config.json")

# Load configuration from JSON
options = OFOptions.load_options("config.json")
# [docs:end]
