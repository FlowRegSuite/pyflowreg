"""
Wrapper classes for video file I/O operations.
Provides multi-file, multi-channel, and subset reading/writing capabilities.
"""

from typing import Union, List
from pathlib import Path

import numpy as np

from pyflowreg.util.io._base import VideoReader, VideoWriter
from pyflowreg.util.io.factory import get_video_file_reader, get_video_file_writer


class MULTIFILEFileWriter(VideoWriter):
    """
    File writer that writes one file per channel.
    Each channel is saved to a separate file with _ch{N} suffix.
    """

    def __init__(self, filename: str, file_type: str = "TIFF", **kwargs):
        """
        Initialize multi-file writer.

        Args:
            filename: Base output filename or directory
            file_type: Output format for each channel file
            **kwargs: Additional parameters passed to individual writers
        """
        super().__init__()

        # Parse filename
        path = Path(filename)
        if path.suffix:
            self.folder = path.parent
            self.file_name = path.stem
        else:
            self.folder = path
            self.file_name = "compensated"

        # Create output directory if needed
        self.folder.mkdir(parents=True, exist_ok=True)

        self.file_type = file_type
        self.writer_parameters = kwargs
        self.file_writers = []

    def write_frames(self, frames: np.ndarray):
        """
        Write frames to multiple files (one per channel).

        Args:
            frames: Array with shape (T, H, W, C) or compatible
        """
        # Normalize input to 4D
        if frames.ndim == 2:  # Single frame, single channel
            frames = frames[np.newaxis, :, :, np.newaxis]
        elif frames.ndim == 3:
            if len(self.file_writers) > 0:  # Already initialized
                if frames.shape[0] == self.height and frames.shape[1] == self.width:
                    frames = frames[np.newaxis, :, :, :]
                else:
                    frames = frames[:, :, :, np.newaxis]
            else:  # First write - guess format
                # Assume (T, H, W) for single channel
                frames = frames[:, :, :, np.newaxis]

        # Initialize on first write
        if not self.initialized:
            T, H, W, C = frames.shape
            self.height = H
            self.width = W
            self.n_channels = C
            self.dtype = frames.dtype
            self.bit_depth = frames.dtype.itemsize * 8
            self.initialized = True

            # Create a writer for each channel
            for ch_idx in range(self.n_channels):
                ch_filename = (
                    self.folder / f"{self.file_name}_ch{ch_idx + 1}.{self.file_type}"
                )
                writer = get_video_file_writer(
                    str(ch_filename), self.file_type, **self.writer_parameters
                )
                self.file_writers.append(writer)

        # Write each channel to its file
        for ch_idx in range(self.n_channels):
            channel_frames = frames[:, :, :, ch_idx : ch_idx + 1]  # Keep 4D shape
            self.file_writers[ch_idx].write_frames(channel_frames)

    def close(self):
        """Close all channel writers."""
        for writer in self.file_writers:
            writer.close()
        self.file_writers = []


class MULTICHANNELFileReader(VideoReader):
    """
    Generic multichannel reader that reads from multiple video files
    and combines them into a single multichannel output.
    """

    def __init__(
        self,
        input_files: List[str],
        buffer_size: int = 500,
        bin_size: int = 1,
        **kwargs,
    ):
        """
        Initialize multichannel reader.

        Args:
            input_files: List of input file paths
            buffer_size: Buffer size for batch reading
            bin_size: Temporal binning factor
            **kwargs: Additional parameters passed to individual readers
        """
        super().__init__()

        self.buffer_size = buffer_size
        self.bin_size = bin_size
        self.filereaders = []
        self.reader_kwargs = kwargs

        # Store file list for initialization
        self.input_files = input_files

    def _initialize(self):
        """Initialize all file readers and set properties."""
        # Create readers for all input files
        different_bits = False
        max_dtype = None

        for i, file_path in enumerate(self.input_files):
            reader = get_video_file_reader(
                file_path, self.buffer_size, self.bin_size, **self.reader_kwargs
            )

            # Ensure the reader is initialized
            if hasattr(reader, "_ensure_initialized"):
                reader._ensure_initialized()

            self.filereaders.append(reader)

            if i == 0:
                # Set properties from first reader
                self.height = reader.height
                self.width = reader.width
                self.frame_count = reader.frame_count
                self.dtype = reader.dtype
                self.n_channels = reader.n_channels
                max_dtype = self.dtype
            else:
                # Validate consistency
                if self.height != reader.height or self.width != reader.width:
                    raise ValueError(f"Resolution mismatch in file {file_path}")
                if self.frame_count != reader.frame_count:
                    raise ValueError(f"Frame count mismatch in file {file_path}")

                # Accumulate channels
                self.n_channels += reader.n_channels

                # Handle different data types
                if reader.dtype != self.dtype:
                    # Use highest precision dtype
                    if np.can_cast(self.dtype, reader.dtype):
                        max_dtype = reader.dtype
                    elif not np.can_cast(reader.dtype, self.dtype):
                        max_dtype = np.float64
                    different_bits = True

        if different_bits:
            print(f"Warning: Different data types in channels, using {max_dtype}")
            self.dtype = max_dtype

        # Create combined name
        self.input_file_name = "_".join([Path(f).stem for f in self.input_files])

    def _read_raw_frames(self, frame_indices: Union[slice, List[int]]) -> np.ndarray:
        """
        Read frames from all files and combine channels.

        Returns:
            Array with shape (T, H, W, C_total)
        """
        # Convert indices to list for consistent handling
        if isinstance(frame_indices, slice):
            start, stop, step = frame_indices.indices(self.frame_count)
            indices = list(range(start, stop, step))
        else:
            indices = list(frame_indices)

        if len(indices) == 0:
            return np.empty(
                (0, self.height, self.width, self.n_channels), dtype=self.dtype
            )

        # Allocate output array
        n_frames = len(indices)
        output = np.zeros(
            (n_frames, self.height, self.width, self.n_channels), dtype=self.dtype
        )

        # Read from each file and combine
        ch_offset = 0
        for reader in self.filereaders:
            # Use reader's indexing directly
            frames = (
                reader[indices]
                if len(indices) > 1
                else reader[indices[0] : indices[0] + 1]
            )

            # Ensure 4D
            if frames.ndim == 3:
                frames = frames[np.newaxis, ...]

            # Copy to output
            n_ch = reader.n_channels
            output[:, :, :, ch_offset : ch_offset + n_ch] = frames.astype(self.dtype)
            ch_offset += n_ch

        return output

    def close(self):
        """Close all file readers."""
        for reader in self.filereaders:
            reader.close()
        self.filereaders = []


class SUBSETFileReader(VideoReader):
    """
    Reader that provides a subset of frames from another video reader.
    Useful for reading non-contiguous frame indices or reordering frames.
    """

    def __init__(
        self, video_file_reader: VideoReader, indices: Union[List[int], np.ndarray]
    ):
        """
        Initialize subset reader.

        Args:
            video_file_reader: Source video reader
            indices: Frame indices to include in subset (0-based)
        """
        super().__init__()

        self.video_file_reader = video_file_reader
        self.indices = np.array(indices, dtype=np.int64)

        # Inherit buffer settings
        self.buffer_size = video_file_reader.buffer_size
        self.bin_size = 1  # Disable binning for subset reading initially

        # Will be set in _initialize
        self._original_bin_size = video_file_reader.bin_size

    def _initialize(self):
        """Initialize properties from source reader."""
        # Ensure source is initialized
        if hasattr(self.video_file_reader, "_ensure_initialized"):
            self.video_file_reader._ensure_initialized()

        # Validate indices
        max_idx = np.max(self.indices)
        if max_idx >= self.video_file_reader.frame_count:
            raise ValueError(
                f"Index {max_idx} exceeds source frame count {self.video_file_reader.frame_count}"
            )

        # Copy properties from source
        self.height = self.video_file_reader.height
        self.width = self.video_file_reader.width
        self.n_channels = self.video_file_reader.n_channels
        self.dtype = self.video_file_reader.dtype

        # Set our frame count to the subset size
        self.frame_count = len(self.indices)

        # Store original bin size and temporarily disable binning in source
        self._original_bin_size = self.video_file_reader.bin_size

    def _read_raw_frames(self, frame_indices: Union[slice, List[int]]) -> np.ndarray:
        """
        Read frames from subset.

        Args:
            frame_indices: Indices into the subset (not the original video)

        Returns:
            Array with shape (T, H, W, C)
        """
        # Convert subset indices to original indices
        if isinstance(frame_indices, slice):
            start, stop, step = frame_indices.indices(self.frame_count)
            subset_indices = list(range(start, stop, step))
        else:
            subset_indices = list(frame_indices)

        if len(subset_indices) == 0:
            return np.empty(
                (0, self.height, self.width, self.n_channels), dtype=self.dtype
            )

        # Map subset indices to original video indices
        original_indices = self.indices[subset_indices]

        # Temporarily disable binning in source reader
        old_bin = self.video_file_reader.bin_size
        self.video_file_reader.bin_size = 1

        try:
            # Read frames from source using mapped indices
            frames = []
            for idx in original_indices:
                frame = self.video_file_reader[int(idx)]
                if frame.ndim == 3:  # Single frame
                    frames.append(frame[np.newaxis, ...])
                else:
                    frames.append(frame)

            result = (
                np.concatenate(frames, axis=0)
                if frames
                else np.empty(
                    (0, self.height, self.width, self.n_channels), dtype=self.dtype
                )
            )
        finally:
            # Restore original binning
            self.video_file_reader.bin_size = old_bin

        return result

    def close(self):
        """No-op as we don't own the source reader."""
        pass  # Don't close the source reader as we don't own it


def main():
    """Test wrapper implementations."""
    import tempfile

    # Create test data
    test_frames = np.random.randint(0, 255, (20, 64, 64, 2), dtype=np.uint8)

    # Test MULTIFILE writer
    print("Testing MULTIFILE writer...")
    with tempfile.TemporaryDirectory() as tmpdir:
        multifile_path = Path(tmpdir) / "test_multi"

        # Use HDF5 format since we have that implemented
        with MULTIFILEFileWriter(str(multifile_path), "HDF5") as writer:
            writer.write_frames(test_frames[:10])
            writer.write_frames(test_frames[10:])

        # Check files were created
        # When path has no extension, it's treated as folder with 'compensated' as default name
        ch1_file = multifile_path / "compensated_ch1.HDF5"
        ch2_file = multifile_path / "compensated_ch2.HDF5"

        assert ch1_file.exists(), "Channel 1 file not created"
        assert ch2_file.exists(), "Channel 2 file not created"
        print("✓ MULTIFILE writer test passed")

        # Test MULTICHANNEL reader
        print("\nTesting MULTICHANNEL reader...")
        reader = MULTICHANNELFileReader([str(ch1_file), str(ch2_file)])

        print(f"Shape: {reader.shape}")
        print(f"Channels: {reader.n_channels}")

        # Read all frames
        all_frames = reader[:]
        assert all_frames.shape == (
            20,
            64,
            64,
            2,
        ), f"Shape mismatch: {all_frames.shape}"
        print("✓ MULTICHANNEL reader test passed")

        # Test SUBSET reader
        print("\nTesting SUBSET reader...")
        subset_indices = [0, 5, 10, 15, 19]
        subset_reader = SUBSETFileReader(reader, subset_indices)

        print(f"Subset shape: {subset_reader.shape}")
        assert subset_reader.frame_count == 5, "Subset frame count incorrect"

        subset_frames = subset_reader[:]
        assert subset_frames.shape == (
            5,
            64,
            64,
            2,
        ), f"Subset shape mismatch: {subset_frames.shape}"

        # Verify correct frames were selected
        for i, orig_idx in enumerate(subset_indices):
            np.testing.assert_array_equal(
                subset_frames[i],
                all_frames[orig_idx],
                err_msg=f"Frame {i} (original {orig_idx}) mismatch",
            )

        print("✓ SUBSET reader test passed")

        reader.close()

    # Test that factory functions are properly imported
    print("\nTesting factory function imports...")
    try:
        from pyflowreg.util.io.factory import get_video_file_reader as factory_reader
        from pyflowreg.util.io.factory import get_video_file_writer as factory_writer

        assert factory_reader == get_video_file_reader
        assert factory_writer == get_video_file_writer
        print("✓ Factory functions properly imported")
    except ImportError as e:
        print(f"✗ Factory import failed: {e}")

    print("\n✓ All wrapper tests passed!")


if __name__ == "__main__":
    main()
