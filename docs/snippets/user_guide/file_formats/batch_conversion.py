# Docs page : docs/user_guide/file_formats.md ("Batch Conversion")
# Test      : tests/docs/user_guide/test_file_formats.py::TestFileFormatsBatchConversion
# Inputs    : tiff_files/*.tif -- created by the test harness
# [docs:start]
from pathlib import Path

from pyflowreg.util.io import get_video_file_reader, get_video_file_writer

input_dir = Path("tiff_files/")
output_dir = Path("hdf5_files/")
output_dir.mkdir(exist_ok=True)

for tiff_file in input_dir.glob("*.tif"):
    reader = get_video_file_reader(str(tiff_file))
    output_file = output_dir / f"{tiff_file.stem}.h5"

    with get_video_file_writer(str(output_file), "HDF5") as writer:
        for batch in reader:
            writer.write_frames(batch)

    reader.close()
    print(f"Converted: {tiff_file.name}")
# [docs:end]
