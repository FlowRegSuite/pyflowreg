# Docs page : docs/user_guide/parallelization.md ("Sequential")
# Test      : tests/docs/user_guide/test_parallelization.py::TestParallelizationSequentialRun
# Inputs    : video.h5 -- created by the test harness
# [docs:start]
from pyflowreg.motion_correction import compensate_recording, OFOptions
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

options = OFOptions(
    input_file="video.h5",
    output_path="results",
    output_format="HDF5",
    quality_setting="fast",
    reference_frames=list(range(5)),
)
config = RegistrationConfig(parallelization="sequential")
compensate_recording(options, config=config)
# [docs:end]
