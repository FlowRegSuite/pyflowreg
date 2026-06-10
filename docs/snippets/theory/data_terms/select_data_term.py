# Docs page : docs/theory/data_terms.md ("Selecting a data term")
# Test      : tests/docs/theory/test_data_terms.py::TestDataTermsSelectDataTerm
# Inputs    : none
# [docs:start]
from pyflowreg.motion_correction import OFOptions

options = OFOptions(constancy_assumption="gc")  # default; "gradient" is an alias
# [docs:end]
