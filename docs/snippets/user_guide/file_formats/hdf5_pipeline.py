# Docs page : docs/user_guide/file_formats.md ("HDF5 (.h5, .hdf5, .hdf)")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsHdf5Pipeline
# Inputs    : video.h5 -- created by the test harness
# [docs:start]
from pyflowreg.motion_correction import compensate_recording, OFOptions

options = OFOptions(
    input_file="video.h5",
    output_path="results/",
    output_format="HDF5",  # Creates separate 3D datasets per channel
    reference_frames=list(range(10)),  # frames averaged into the reference
)
compensate_recording(options)
# [docs:end]
