# Docs page : docs/user_guide/file_formats.md ("Simple Conversion")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsTiffToHdf5
# Inputs    : input.tif -- created by the test harness
# [docs:start]
from pyflowreg.util.io import get_video_file_reader, get_video_file_writer

# Read TIFF
reader = get_video_file_reader("input.tif")

# Write as HDF5
with get_video_file_writer("output.h5", "HDF5") as writer:
    for batch in reader:
        writer.write_frames(batch)

reader.close()
# [docs:end]
