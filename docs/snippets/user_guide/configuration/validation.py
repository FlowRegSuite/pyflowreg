# Docs page : docs/user_guide/configuration.md ("Validation")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationValidation
# Inputs    : none
# [docs:start]
from pydantic import ValidationError

from pyflowreg.motion_correction import OFOptions

# Scalar alpha is expanded to a 2-tuple
options = OFOptions(alpha=2.0)
print(options.alpha)  # (2.0, 2.0)

# Invalid parameter values raise a ValidationError ("Alpha must be positive")
try:
    OFOptions(alpha=-1)
except ValidationError as e:
    print(f"Rejected: {e}")

# Unknown field names are rejected (extra="forbid")
try:
    OFOptions(alhpa=2.0)
except ValidationError as e:
    print(f"Rejected: {e}")
# [docs:end]
