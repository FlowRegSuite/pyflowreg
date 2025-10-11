"""
Tests for NullVideoWriter.

Tests the Null Object Pattern implementation for VideoWriter that discards
all frames without storage, useful for callback-only processing.
"""

import pytest
import numpy as np

from pyflowreg.util.io import NullVideoWriter
from pyflowreg.util.io.factory import get_video_file_writer
from pyflowreg.motion_correction.OF_options import OutputFormat


class TestNullWriterBasics:
    """Test basic NullVideoWriter functionality."""

    def test_initialization(self):
        """Test NullVideoWriter can be initialized."""
        writer = NullVideoWriter()
        assert not writer.initialized
        assert writer.frames_written == 0
        assert writer.batches_written == 0

    def test_write_single_frame_2d(self):
        """Test writing single 2D frame (H,W)."""
        writer = NullVideoWriter()
        frame = np.random.rand(16, 32).astype(np.float32)

        writer.write_frames(frame)

        assert writer.initialized
        assert writer.frames_written == 1
        assert writer.batches_written == 1
        assert writer.height == 16
        assert writer.width == 32
        assert writer.n_channels == 1

    def test_write_single_frame_3d(self):
        """Test writing single 3D frame (H,W,C)."""
        writer = NullVideoWriter()
        frame = np.random.rand(16, 32, 2).astype(np.float32)

        writer.write_frames(frame)

        assert writer.initialized
        assert writer.frames_written == 1
        assert writer.batches_written == 1
        assert writer.height == 16
        assert writer.width == 32
        assert writer.n_channels == 2

    def test_write_batch_4d(self):
        """Test writing batch of frames (T,H,W,C)."""
        writer = NullVideoWriter()
        batch = np.random.rand(10, 16, 32, 2).astype(np.float32)

        writer.write_frames(batch)

        assert writer.initialized
        assert writer.frames_written == 10
        assert writer.batches_written == 1
        assert writer.height == 16
        assert writer.width == 32
        assert writer.n_channels == 2

    def test_multiple_batches(self):
        """Test writing multiple batches accumulates counts correctly."""
        writer = NullVideoWriter()

        # First batch
        batch1 = np.random.rand(5, 16, 32, 2).astype(np.float32)
        writer.write_frames(batch1)
        assert writer.frames_written == 5
        assert writer.batches_written == 1

        # Second batch
        batch2 = np.random.rand(7, 16, 32, 2).astype(np.float32)
        writer.write_frames(batch2)
        assert writer.frames_written == 12
        assert writer.batches_written == 2

        # Third batch
        batch3 = np.random.rand(3, 16, 32, 2).astype(np.float32)
        writer.write_frames(batch3)
        assert writer.frames_written == 15
        assert writer.batches_written == 3

    def test_close_no_op(self):
        """Test close is a no-op (no errors)."""
        writer = NullVideoWriter()
        batch = np.random.rand(5, 16, 32, 2).astype(np.float32)
        writer.write_frames(batch)

        # Should not raise
        writer.close()

        # Counts preserved after close
        assert writer.frames_written == 5
        assert writer.batches_written == 1

    def test_context_manager(self):
        """Test NullVideoWriter works as context manager."""
        batch = np.random.rand(5, 16, 32, 2).astype(np.float32)

        with NullVideoWriter() as writer:
            writer.write_frames(batch)
            assert writer.frames_written == 5

    def test_repr(self):
        """Test string representation."""
        writer = NullVideoWriter()
        batch = np.random.rand(10, 16, 32, 2).astype(np.float32)
        writer.write_frames(batch)

        repr_str = repr(writer)
        assert "NullVideoWriter" in repr_str
        assert "frames_written=10" in repr_str
        assert "batches=1" in repr_str


class TestNullWriterDataTypes:
    """Test NullVideoWriter with different data types."""

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
    def test_different_dtypes(self, dtype):
        """Test NullVideoWriter accepts different data types."""
        writer = NullVideoWriter()
        batch = np.random.rand(5, 16, 32, 2).astype(dtype)

        writer.write_frames(batch)

        assert writer.frames_written == 5
        assert writer.dtype == dtype

    @pytest.mark.parametrize("n_channels", [1, 2, 3, 4])
    def test_different_channel_counts(self, n_channels):
        """Test NullVideoWriter with different channel counts."""
        writer = NullVideoWriter()
        batch = np.random.rand(5, 16, 32, n_channels).astype(np.float32)

        writer.write_frames(batch)

        assert writer.n_channels == n_channels


class TestNullWriterFactory:
    """Test NullVideoWriter integration with factory."""

    def test_factory_creates_null_writer(self):
        """Test factory creates NullVideoWriter for NULL format."""
        writer = get_video_file_writer(None, "NULL")

        assert isinstance(writer, NullVideoWriter)

    def test_output_format_enum(self):
        """Test OutputFormat.NULL enum value."""
        assert OutputFormat.NULL == "NULL"
        assert OutputFormat.NULL.value == "NULL"

    def test_factory_with_output_format_enum(self):
        """Test factory with OutputFormat enum."""
        writer = get_video_file_writer(None, OutputFormat.NULL.value)

        assert isinstance(writer, NullVideoWriter)


class TestNullWriterWithCompensation:
    """Test NullVideoWriter in motion compensation pipeline."""

    def test_null_writer_with_compensate_arr(self):
        """Test using NULL output format with compensate_arr."""
        from pyflowreg.motion_correction.compensate_arr import compensate_arr
        from pyflowreg.motion_correction.OF_options import OFOptions

        # Create simple test data
        video = np.random.rand(10, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        # Configure to use NULL writer
        options = OFOptions(
            output_format=OutputFormat.NULL,
            levels=1,
            iterations=2,
            save_w=True,
            buffer_size=5,
        )

        # Should still return registered array and flow fields
        registered, w = compensate_arr(video, reference, options=options)

        # Verify output
        assert registered.shape == video.shape
        assert w.shape == (10, 16, 32, 2)

        # The internal writer should be NullVideoWriter (discarded, but we can't directly check)
        # The key test is that it works without errors

    def test_null_writer_no_disk_io(self, tmp_path):
        """Test NULL writer doesn't create output files."""
        from pyflowreg.motion_correction.compensate_arr import compensate_arr
        from pyflowreg.motion_correction.OF_options import OFOptions

        video = np.random.rand(10, 16, 32, 2).astype(np.float32)
        reference = np.mean(video[:5], axis=0)

        options = OFOptions(
            output_format=OutputFormat.NULL,
            output_path=tmp_path,
            levels=1,
            iterations=2,
            save_meta_info=True,  # Even with this enabled
        )

        registered, w = compensate_arr(video, reference, options=options)

        # Check no video output files were created
        output_files = list(tmp_path.glob("compensated.*"))
        assert len(output_files) == 0, "NULL writer should not create output files"


class TestNullWriterEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_dimensions(self):
        """Test that invalid array dimensions raise appropriate errors."""
        writer = NullVideoWriter()

        # 1D array should fail
        with pytest.raises(ValueError, match="Expected 2D, 3D or 4D"):
            writer.write_frames(np.random.rand(100))

        # 5D array should fail
        with pytest.raises(ValueError, match="Expected 2D, 3D or 4D"):
            writer.write_frames(np.random.rand(2, 2, 16, 32, 2))

    def test_write_before_init(self):
        """Test that writing initializes the writer."""
        writer = NullVideoWriter()
        assert not writer.initialized

        writer.write_frames(np.random.rand(5, 16, 32, 2))
        assert writer.initialized

    def test_multiple_writes_preserve_init(self):
        """Test that subsequent writes don't re-initialize."""
        writer = NullVideoWriter()

        # First write initializes
        writer.write_frames(np.random.rand(5, 16, 32, 2).astype(np.float32))
        first_init_state = writer.initialized
        first_dtype = writer.dtype

        # Second write should preserve initialization
        writer.write_frames(np.random.rand(3, 16, 32, 2).astype(np.float32))

        assert writer.initialized == first_init_state
        assert writer.dtype == first_dtype
        assert writer.frames_written == 8  # 5 + 3
