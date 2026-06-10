# Docs page : docs/user_guide/file_formats.md ("Multi-Channel from Separate Files")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsMultichannelReader
# Inputs    : ch1.tif, ch2.tif -- created by the test harness
# [docs:start]
from pyflowreg.util.io.multifile_wrappers import MULTICHANNELFileReader

# List of single-channel files
reader = MULTICHANNELFileReader(["ch1.tif", "ch2.tif"])
frames = reader[:]  # Shape: (T, H, W, 2)
# [docs:end]
