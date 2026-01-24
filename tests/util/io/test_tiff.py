import numpy as np
import tifffile

from pyflowreg.util.io.tiff import TIFFFileReader


def test_read_batch_overreported_frame_count(tmp_path):
    """
    Simulate metadata reporting more frames than pages exist.
    Current reader will index past the last page and raise IndexError.
    """
    tif_path = tmp_path / "single_page.tif"
    with tifffile.TiffWriter(tif_path) as tw:
        tw.write(np.zeros((4, 4), dtype=np.uint16))

    reader = TIFFFileReader(str(tif_path), buffer_size=2)
    reader._ensure_initialized()

    # Overreport frame_count to mimic bad metadata (e.g., SizeT too large)
    reader.frame_count = len(reader._tiff_file.pages) + 1

    # Expected (post-fix): should not raise; currently will IndexError on page 1
    frames = reader.read_batch()
    assert frames.shape[0] == 1
