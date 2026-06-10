# Docs page : docs/user_guide/parallelization.md ("Threading")
# Test      : tests/docs/user_guide/test_parallelization.py::TestParallelizationThreadingConfig
# Inputs    : none
# [docs:start]
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

config = RegistrationConfig(
    parallelization="threading",
    n_jobs=8,  # Number of worker threads
)
# [docs:end]
