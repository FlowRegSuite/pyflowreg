# Docs page : docs/user_guide/configuration.md ("Cross-Correlation Pre-Alignment")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationCcInitialization
# Inputs    : none
# Setup (hidden from the docs page; the page imports OFOptions in an earlier block)
from pyflowreg.motion_correction import OFOptions

# [docs:start]
options = OFOptions(
    cc_initialization=True,  # default False
    cc_hw=256,  # target height/width of the correlation images, int or (H, W) tuple (default 256)
    cc_up=1,  # upsampling factor for subpixel accuracy; 1 = integer-pixel (default 1)
)
# [docs:end]
