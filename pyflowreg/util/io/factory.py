import os
from typing import Union, List, Optional, Dict, Any
from pathlib import Path

import numpy as np

from pyflowreg.util.io._base import VideoReader, VideoWriter


def get_video_file_reader(file_path: str, buffer_size: int = 500,
                          bin_size: int = 1, **kwargs) -> VideoReader:
    """
    Factory function to create appropriate reader based on file extension.

    Args:
        file_path: Path to video file
        buffer_size: Buffer size for reading
        bin_size: Temporal binning factor
        **kwargs: Additional reader-specific arguments

    Returns:
        Appropriate VideoReader subclass instance
    """
    from pathlib import Path

    # Import readers here to avoid circular imports
    from pyflowreg.util.io.tiff import TIFFFileReader
    from pyflowreg.util.io.hdf5 import HDF5FileReader
    from pyflowreg.util.io.mat import MATFileReader
    from pyflowreg.util.io.mdf import MDFFileReader

    ext = Path(file_path).suffix.lower()

    readers = {
        '.tif': TIFFFileReader,
        '.tiff': TIFFFileReader,
        '.h5': HDF5FileReader,
        '.hdf5': HDF5FileReader,
        '.hdf': HDF5FileReader,
        '.mat': MATFileReader,
        '.mdf': MDFFileReader,
    }

    reader_class = readers.get(ext)
    if reader_class:
        return reader_class(file_path, buffer_size, bin_size, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def get_video_file_writer(file_path: str, file_type: Optional[str] = None, **kwargs) -> VideoWriter:
    """
    Factory function to create appropriate writer based on file extension.

    Args:
        file_path: Output file path
        file_type: Optional explicit file type (overrides extension)
        **kwargs: Additional writer-specific arguments

    Returns:
        Appropriate VideoWriter subclass instance
    """
    from pathlib import Path

    # Import writers here to avoid circular imports
    from pyflowreg.util.io.tiff import TIFFFileWriter
    from pyflowreg.util.io.hdf5 import HDF5FileWriter
    from pyflowreg.util.io.mat import MATFileWriter

    if file_type:
        ext = '.' + file_type.lower()
    else:
        ext = Path(file_path).suffix.lower()

    writers = {
        '.tif': TIFFFileWriter,
        '.tiff': TIFFFileWriter,
        '.h5': HDF5FileWriter,
        '.hdf5': HDF5FileWriter,
        '.hdf': HDF5FileWriter,
        '.mat': MATFileWriter,
    }

    writer_class = writers.get(ext)
    if writer_class:
        return writer_class(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def main():
    """Test wrapper implementations."""
    import tempfile
    import shutil
    from multifile_wrappers import MULTICHANNELFileReader, SUBSETFileReader, MULTIFILEFileWriter

    # Create test data
    test_frames = np.random.randint(0, 255, (20, 64, 64, 2), dtype=np.uint8)

    # Test MULTIFILE writer
    print("Testing MULTIFILE writer...")
    with tempfile.TemporaryDirectory() as tmpdir:
        multifile_path = Path(tmpdir) / "test_multi"

        with MULTIFILEFileWriter(str(multifile_path), 'TIFF') as writer:
            writer.write_frames(test_frames[:10])
            writer.write_frames(test_frames[10:])

        # Check files were created
        ch1_file = multifile_path / "compensated_ch1.TIFF"
        ch2_file = multifile_path / "compensated_ch2.TIFF"

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
        assert all_frames.shape == (20, 64, 64, 2), f"Shape mismatch: {all_frames.shape}"
        print("✓ MULTICHANNEL reader test passed")

        # Test SUBSET reader
        print("\nTesting SUBSET reader...")
        subset_indices = [0, 5, 10, 15, 19]
        subset_reader = SUBSETFileReader(reader, subset_indices)

        print(f"Subset shape: {subset_reader.shape}")
        assert subset_reader.frame_count == 5, "Subset frame count incorrect"

        subset_frames = subset_reader[:]
        assert subset_frames.shape == (5, 64, 64, 2), f"Subset shape mismatch: {subset_frames.shape}"

        # Verify correct frames were selected
        for i, orig_idx in enumerate(subset_indices):
            np.testing.assert_array_equal(
                subset_frames[i], all_frames[orig_idx],
                err_msg=f"Frame {i} (original {orig_idx}) mismatch"
            )

        print("✓ SUBSET reader test passed")

        reader.close()

    print("\n✓ All wrapper tests passed!")


if __name__ == "__main__":
    main()
