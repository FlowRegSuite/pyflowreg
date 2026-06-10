# Docs page : docs/user_guide/parallelization.md ("Multiprocessing")
# Test      : tests/docs/user_guide/test_parallelization.py::TestParallelizationMultiprocessingConfig
# Inputs    : none
# [docs:start]
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

config = RegistrationConfig(
    parallelization="multiprocessing",
    n_jobs=-1,  # Use all available CPUs
)
# [docs:end]
