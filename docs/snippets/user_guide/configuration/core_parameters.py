# Docs page : docs/user_guide/configuration.md ("Core Optical Flow Parameters")
# Test      : tests/docs/user_guide/test_configuration.py::TestConfigurationCoreParameters
# Inputs    : none
# Setup (hidden from the docs page; the page imports OFOptions in an earlier block)
from pyflowreg.motion_correction import OFOptions

# [docs:start]
options = OFOptions(
    # Smoothness regularization weight (alpha_x, alpha_y), default (1.5, 1.5)
    alpha=(1.5, 1.5),  # A scalar is expanded to (alpha, alpha)
    # Solver iterations per pyramid level (default 50)
    iterations=50,
    # Pyramid configuration
    levels=100,  # Upper bound on pyramid levels; actual depth is limited by image size (default 100)
    eta=0.8,  # Downsampling factor between levels, in (0, 1] (default 0.8)
    min_level=-1,  # -1 (default) derives the finest level from quality_setting
    # Nonlinear diffusion parameters
    a_smooth=1.0,  # Smoothness diffusion parameter (default 1.0)
    a_data=0.45,  # Data-term diffusion parameter, in (0, 1] (default 0.45)
    update_lag=5,  # Update lag for the non-linear diffusion weights (default 5)
)
# [docs:end]
