# Docs page : docs/user_guide/configuration.md ("Input/Output Paths")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationIoPaths
# Inputs    : none -- the input file is only referenced, not opened
# Setup (hidden from the docs page; the page imports OFOptions in an earlier block)
from pyflowreg.motion_correction import OFOptions

# [docs:start]
options = OFOptions(
    input_file="raw_video.h5",
    output_path="results/",  # default "results"
    output_format="HDF5",  # default MAT
)
# [docs:end]
