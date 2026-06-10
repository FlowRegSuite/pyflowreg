# Docs page : docs/quickstart.md ("Parallel Processing")
# Test      : tests/docs/test_quickstart.py::TestQuickstartParallelProcessing
# Inputs    : my_video.h5 -- created by the test harness
# [docs:start]
from pyflowreg.motion_correction import compensate_recording, OFOptions
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

options = OFOptions(
    input_file="my_video.h5",
    output_path="results/",
    quality_setting="balanced",
)

# Manual executor selection
config = RegistrationConfig(
    n_jobs=-1,  # Use all CPU cores (-1, default) or specify a count
    parallelization="threading",  # "sequential", "threading", or "multiprocessing"
)

compensate_recording(options, config=config)
# [docs:end]
