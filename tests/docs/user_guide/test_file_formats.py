"""
Tests for the code examples on docs/user_guide/file_formats.md.

Each test materializes the exact input files the published example
references (synthetic data written through pyflowreg's own writers),
executes the corresponding snippet from
``docs/snippets/user_guide/file_formats/`` via ``snippet_runner``, and
asserts on the resulting namespace and on the files the snippet writes
(read back through ``get_video_file_reader``).
"""

import numpy as np
import pytest

from pyflowreg._runtime import RuntimeContext
from pyflowreg.util.io import ArrayReader, get_video_file_reader

pytestmark = pytest.mark.docs_example

SNIPPET_DIR = "user_guide/file_formats"


def _read_video(path):
    """Read a whole video file back through the pyflowreg reader factory."""
    reader = get_video_file_reader(str(path))
    video = reader[:]
    reader.close()
    return video


class TestFileFormatsHdf5Pipeline:
    """HDF5 section: compensate_recording writing per-channel datasets."""

    def test_hdf5_pipeline_executes(self, materialize_input, snippet_runner, tmp_path):
        materialize_input("video.h5")

        # Keep the docs example fast in the test harness: restrict executor
        # auto-selection to in-process executors so no worker processes are
        # spawned for this tiny synthetic recording. The published example
        # itself is unchanged and uses the default auto-selection.
        with RuntimeContext.use(available_parallelization={"sequential", "threading"}):
            ns = snippet_runner(f"{SNIPPET_DIR}/hdf5_pipeline.py")

        assert ns["options"].output_format.value == "HDF5"

        output_file = tmp_path / "results" / "compensated.HDF5"
        assert output_file.exists()

        registered = _read_video(output_file)
        assert registered.shape == (12, 32, 48, 2)
        assert np.all(np.isfinite(registered))


class TestFileFormatsMemoryWriters:
    """In-Memory Output section: ARRAY and NULL writers via the factory."""

    def test_memory_writers_executes(self, snippet_runner):
        ns = snippet_runner(f"{SNIPPET_DIR}/memory_writers.py")

        assert ns["video"].shape == (5, 64, 64, 2)
        assert np.array_equal(ns["video"], ns["frames"])

        null_writer = ns["null_writer"]
        assert null_writer.frames_written == 5
        assert null_writer.batches_written == 1


class TestFileFormatsMultifileWriter:
    """Multi-File Formats section: MULTIFILE_HDF5 writes one file per channel."""

    def test_multifile_writer_executes(self, snippet_runner, tmp_path):
        ns = snippet_runner(f"{SNIPPET_DIR}/multifile_writer.py")

        frames = ns["frames"]
        for ch in (1, 2):
            ch_file = tmp_path / "multifile" / f"compensated_ch{ch}.HDF5"
            assert ch_file.exists()

            channel_video = _read_video(ch_file)
            assert channel_video.shape == (8, 32, 32, 1)
            assert np.array_equal(channel_video, frames[:, :, :, ch - 1 : ch])


class TestFileFormatsFactoryReaders:
    """Creating Readers section: file, array, and multi-channel inputs."""

    def test_factory_readers_executes(self, materialize_input, snippet_runner):
        materialize_input("video.h5")
        materialize_input("ch1.tif", shape=(8, 32, 48, 1), seed=1)
        materialize_input("ch2.tif", shape=(8, 32, 48, 1), seed=2)

        ns = snippet_runner(f"{SNIPPET_DIR}/factory_readers.py")

        assert ns["video_array"].shape == (12, 32, 48, 2)

        array_reader = ns["array_reader"]
        assert isinstance(array_reader, ArrayReader)
        assert array_reader.shape == (12, 32, 48, 2)
        array_reader.close()

        multichannel_reader = ns["multichannel_reader"]
        multichannel_frames = multichannel_reader[:]
        multichannel_reader.close()
        assert multichannel_frames.shape == (8, 32, 48, 2)


class TestFileFormatsFactoryWriter:
    """Creating Writers section: write frames through the writer factory."""

    def test_factory_writer_executes(self, snippet_runner, tmp_path):
        ns = snippet_runner(f"{SNIPPET_DIR}/factory_writer.py")

        written = _read_video(tmp_path / "output.h5")
        assert written.shape == (10, 64, 64, 2)
        assert written.dtype == np.float32
        assert np.array_equal(written, ns["frames"])


class TestFileFormatsArrayIndexing:
    """Array-Like Indexing section: int, slice, list, and tuple indexing."""

    def test_array_indexing_executes(self, materialize_input, snippet_runner):
        materialize_input("video.h5", shape=(32, 220, 220, 2))

        ns = snippet_runner(f"{SNIPPET_DIR}/array_indexing.py")

        # Single frame: (H, W, C)
        assert ns["frame"].shape == (220, 220, 2)
        # Final assignment is the spatial subset: (T, H, W, C) cropped
        assert ns["frames"].shape == (10, 100, 100, 2)
        ns["reader"].close()


class TestFileFormatsBatchIteration:
    """Batch Iteration section: iterating a reader yields all frames."""

    def test_batch_iteration_executes(self, materialize_input, snippet_runner):
        # 120 frames with buffer_size=100 forces a full and a partial batch.
        materialize_input("video.h5", shape=(120, 24, 32, 1))

        ns = snippet_runner(f"{SNIPPET_DIR}/batch_iteration.py")

        assert ns["n_frames"] == 120


class TestFileFormatsMultichannelReader:
    """Multi-Channel from Separate Files section: MULTICHANNELFileReader."""

    def test_multichannel_reader_executes(self, materialize_input, snippet_runner):
        materialize_input("ch1.tif", shape=(10, 32, 48, 1), seed=1)
        materialize_input("ch2.tif", shape=(10, 32, 48, 1), seed=2)

        ns = snippet_runner(f"{SNIPPET_DIR}/multichannel_reader.py")

        assert ns["frames"].shape == (10, 32, 48, 2)

        # Channels come from the per-channel input files in list order.
        ch1 = _read_video("ch1.tif")
        ch2 = _read_video("ch2.tif")
        assert np.array_equal(ns["frames"][..., 0:1], ch1)
        assert np.array_equal(ns["frames"][..., 1:2], ch2)
        ns["reader"].close()


class TestFileFormatsBatchWriting:
    """Batch Writing section: appending several batches to one file."""

    def test_batch_writing_executes(self, snippet_runner, tmp_path):
        ns = snippet_runner(f"{SNIPPET_DIR}/batch_writing.py")

        written = _read_video(tmp_path / "output.h5")
        assert written.shape == (30, 64, 64, 2)

        expected = np.concatenate(ns["video_batches"], axis=0)
        assert np.array_equal(written, expected)


class TestFileFormatsWriterContextManager:
    """Context Manager section: writer closes automatically."""

    def test_writer_context_manager_executes(self, snippet_runner, tmp_path):
        ns = snippet_runner(f"{SNIPPET_DIR}/writer_context_manager.py")

        written = _read_video(tmp_path / "output.h5")
        assert written.shape == (10, 64, 64, 2)
        assert np.array_equal(written, ns["frames"])


class TestFileFormatsTiffToHdf5:
    """Simple Conversion section: TIFF to HDF5 round-trip."""

    def test_tiff_to_hdf5_executes(self, materialize_input, snippet_runner, tmp_path):
        materialize_input("input.tif")

        snippet_runner(f"{SNIPPET_DIR}/tiff_to_hdf5.py")

        original = _read_video(tmp_path / "input.tif")
        converted = _read_video(tmp_path / "output.h5")
        assert original.shape == (12, 32, 48, 2)
        assert converted.shape == original.shape
        assert np.array_equal(converted, original)


class TestFileFormatsBatchConversion:
    """Batch Conversion section: converting a folder of TIFF files."""

    def test_batch_conversion_executes(
        self, materialize_input, snippet_runner, tmp_path
    ):
        materialize_input("tiff_files/recording_000.tif", shape=(6, 24, 32, 1), seed=1)
        materialize_input("tiff_files/recording_001.tif", shape=(6, 24, 32, 1), seed=2)

        snippet_runner(f"{SNIPPET_DIR}/batch_conversion.py")

        for name in ("recording_000", "recording_001"):
            converted_file = tmp_path / "hdf5_files" / f"{name}.h5"
            assert converted_file.exists()

            converted = _read_video(converted_file)
            original = _read_video(tmp_path / "tiff_files" / f"{name}.tif")
            assert converted.shape == (6, 24, 32, 1)
            assert np.array_equal(converted, original)


class TestFileFormatsHdf5WriterOptions:
    """HDF5 Optimization section: compression and chunk_size options."""

    @pytest.mark.parametrize("filename", ["compressed.h5", "chunked.h5"])
    def test_hdf5_writer_options_executes(self, snippet_runner, tmp_path, filename):
        ns = snippet_runner(f"{SNIPPET_DIR}/hdf5_writer_options.py")

        written = _read_video(tmp_path / filename)
        assert written.shape == (10, 64, 64, 1)
        assert np.array_equal(written, ns["frames"])


class TestFileFormatsTiffMemmap:
    """Memory-Mapped TIFF section: reading with use_memmap=False."""

    def test_tiff_memmap_executes(self, materialize_input, snippet_runner):
        materialize_input("large.tif")

        ns = snippet_runner(f"{SNIPPET_DIR}/tiff_memmap.py")

        assert ns["frames"].shape == (12, 32, 48, 2)
