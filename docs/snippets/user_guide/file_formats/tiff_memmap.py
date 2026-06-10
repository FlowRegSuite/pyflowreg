# Docs page : docs/user_guide/file_formats.md ("Memory-Mapped TIFF")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsTiffMemmap
# Inputs    : large.tif -- created by the test harness
# [docs:start]
from pyflowreg.util.io import get_video_file_reader

# Memory mapping is enabled by default for the tifffile series
# read path (use_memmap=True); disable it if needed:
reader = get_video_file_reader("large.tif", use_memmap=False)
frames = reader[:]
reader.close()
# [docs:end]
