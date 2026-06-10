import numpy as np
import pytest
import tifffile

from pyflowreg.util.io.tiff import TIFFFileReader


def _write_interleaved_pages_tiff(path, data):
    """
    Write (T, H, W, C) data as interleaved single-channel pages:
    t0c0, t0c1, ..., t1c0, t1c1, ...
    """
    t_count, _, _, c_count = data.shape
    with tifffile.TiffWriter(path) as tw:
        for t_idx in range(t_count):
            for c_idx in range(c_count):
                tw.write(data[t_idx, :, :, c_idx])


def test_read_batch_overreported_frame_count(tmp_path):
    """
    Guard existing behavior:
    metadata overreporting should be clamped to available pages.
    """
    tif_path = tmp_path / "single_page.tif"
    with tifffile.TiffWriter(tif_path) as tw:
        tw.write(np.zeros((4, 4), dtype=np.uint16))

    reader = TIFFFileReader(str(tif_path), buffer_size=2)
    reader._ensure_initialized()

    # Overreport frame_count to mimic bad metadata (e.g., SizeT too large).
    reader.frame_count = len(reader._tiff_file.pages) + 1

    # Should not raise: clamp protects page-based reads from out-of-range access.
    frames = reader.read_batch()
    assert frames.shape[0] == 1


def test_deinterleave_reads_interleaved_pages_correctly(tmp_path):
    """
    Guard existing deinterleave compatibility used by ScanImage/Suite2p-style files.
    """
    t_count, height, width, c_count = 4, 8, 6, 2
    data = np.zeros((t_count, height, width, c_count), dtype=np.uint16)
    data[:, :, :, 0] = 100
    data[:, :, :, 1] = 1000

    tif_path = tmp_path / "interleaved_pages.tif"
    _write_interleaved_pages_tiff(tif_path, data)

    reader = TIFFFileReader(str(tif_path), buffer_size=2, deinterleave=2)
    try:
        frames = reader[:]
        assert reader.frame_count == t_count
        assert reader.n_channels == c_count
        assert frames.shape == (t_count, height, width, c_count)
        np.testing.assert_array_equal(frames, data)
    finally:
        reader.close()


def test_scanimage_zstack_metadata_overrides_generic_axis_inference(
    tmp_path, monkeypatch
):
    """
    Guard ScanImage compatibility:
    if ScanImage metadata provides flattened z-stack frame count, the reader should use it.
    """
    tif_path = tmp_path / "scanimage_like_stack.tif"
    with tifffile.TiffWriter(tif_path) as tw:
        for i in range(8):
            tw.write(np.full((5, 7), i, dtype=np.uint16))

    def _fake_check_scanimage_metadata(self):
        self._is_scanimage = True
        self._scanimage_metadata = {"is_scanimage": True, "version": "test"}
        self._z_stack_info = {
            "total_frames_flattened": 6,
            "slices_per_volume": 3,
            "frames_per_slice": 2,
            "volumes": 1,
        }

    monkeypatch.setattr(
        TIFFFileReader, "_check_scanimage_metadata", _fake_check_scanimage_metadata
    )

    reader = TIFFFileReader(str(tif_path), buffer_size=3)
    try:
        reader._ensure_initialized()
        assert reader._is_scanimage is True
        assert reader.frame_count == 6

        frames = reader[:]
        assert frames.shape[0] == 6
    finally:
        reader.close()


def test_scanimage_overreported_zstack_count_still_clamps_to_pages(
    tmp_path, monkeypatch
):
    """
    Guard ScanImage compatibility:
    overreported ScanImage flattened count must still clamp to actual page count.
    """
    tif_path = tmp_path / "scanimage_like_overreported.tif"
    with tifffile.TiffWriter(tif_path) as tw:
        for i in range(5):
            tw.write(np.full((4, 4), i, dtype=np.uint16))

    def _fake_check_scanimage_metadata(self):
        self._is_scanimage = True
        self._scanimage_metadata = {"is_scanimage": True, "version": "test"}
        self._z_stack_info = {
            "total_frames_flattened": 50,  # intentionally too large
            "slices_per_volume": 5,
            "frames_per_slice": 10,
            "volumes": 1,
        }

    monkeypatch.setattr(
        TIFFFileReader, "_check_scanimage_metadata", _fake_check_scanimage_metadata
    )

    reader = TIFFFileReader(str(tif_path), buffer_size=4)
    try:
        reader._ensure_initialized()
        assert reader.frame_count == 5

        frames = reader[:]
        assert frames.shape[0] == 5
    finally:
        reader.close()


def test_imagej_contiguous_single_page_stack_uses_series_frame_count(tmp_path):
    """
    Specification test for currently unsupported input:
    ImageJ TYX stack stored contiguously as a single TIFF page.
    """
    t_count, height, width = 9, 12, 10
    data = np.arange(t_count * height * width, dtype=np.uint16).reshape(
        t_count, height, width
    )

    tif_path = tmp_path / "imagej_contiguous_single_page.tif"
    tifffile.imwrite(
        tif_path,
        data,
        imagej=True,
        contiguous=True,
        metadata={"axes": "TYX"},
    )

    with tifffile.TiffFile(tif_path) as tif:
        if len(tif.series) == 0:
            pytest.skip("No series found in generated ImageJ TIFF")
        if len(tif.pages) != 1:
            pytest.skip(
                "Generated TIFF is not single-page contiguous on this tifffile build"
            )
        if tif.series[0].axes != "TYX":
            pytest.skip(f"Expected TYX axes, got {tif.series[0].axes!r}")

    reader = TIFFFileReader(str(tif_path), buffer_size=4)
    try:
        reader._ensure_initialized()
        assert reader.frame_count == t_count
        assert len(reader) == t_count
        frames = reader[:]
        assert frames.shape == (t_count, height, width, 1)
        np.testing.assert_array_equal(frames[:, :, :, 0], data)
    finally:
        reader.close()


def test_single_page_series_time_stack_reads_via_series_mode(tmp_path, monkeypatch):
    """
    Deterministic guard for contiguous stack behavior:
    one TIFF page, but series exposes multiple TYX frames.
    """
    t_count, height, width = 6, 5, 4
    data = np.arange(t_count * height * width, dtype=np.uint16).reshape(
        t_count, height, width
    )

    class _FakeSeries:
        def __init__(self, arr):
            self._arr = arr
            self.shape = arr.shape
            self.axes = "TYX"
            self.dtype = arr.dtype

        def asarray(self, out=None):
            return self._arr

    class _FakePage:
        def __init__(self, first_frame):
            self._frame = first_frame
            self.shape = first_frame.shape
            self.dtype = first_frame.dtype
            self.samplesperpixel = 1
            self.imagelength = first_frame.shape[0]
            self.imagewidth = first_frame.shape[1]
            self.tags = {}

        def asarray(self):
            return self._frame

    class _FakeTiffFile:
        def __init__(self, arr):
            self.is_scanimage = False
            self.series = [_FakeSeries(arr)]
            self.pages = [_FakePage(arr[0])]

        def close(self):
            return None

    dummy_path = tmp_path / "single_page_series.tif"
    dummy_path.write_bytes(b"not-a-real-tiff")

    monkeypatch.setattr(
        "pyflowreg.util.io.tiff.tifffile.TiffFile",
        lambda _: _FakeTiffFile(data),
    )

    reader = TIFFFileReader(str(dummy_path), buffer_size=3, bin_size=1)
    try:
        reader._ensure_initialized()
        assert reader.frame_count == t_count
        assert len(reader) == t_count
        assert reader.n_channels == 1

        frames = reader[:]
        assert frames.shape == (t_count, height, width, 1)
        np.testing.assert_array_equal(frames[:, :, :, 0], data)
    finally:
        reader.close()
