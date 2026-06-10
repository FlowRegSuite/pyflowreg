# Docs page : docs/user_guide/prealignment.md ("Enabling Pre-Alignment")
# Test      : tests/docs/user_guide/test_prealignment.py::TestPrealignmentEnablePrealignment
# Inputs    : recording.h5 -- created by the test harness
# [docs:start]
from pyflowreg.motion_correction import OFOptions, compensate_recording

options = OFOptions(
    input_file="recording.h5",
    output_path="results",
    output_format="HDF5",
    reference_frames=list(range(10)),
    cc_initialization=True,  # enable pre-alignment
    cc_hw=256,  # correlation size (default)
    cc_up=4,  # subpixel upsampling
)
compensate_recording(options)
# [docs:end]
