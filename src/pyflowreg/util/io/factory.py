from typing import Union, List
from pathlib import Path
import warnings

import numpy as np

from pyflowreg.util.io._base import VideoReader, VideoWriter


def get_video_file_reader(
    input_source: Union[str, Path, np.ndarray, VideoReader, List[str]],
    buffer_size: int = 500,
    bin_size: int = 1,
    **kwargs,
) -> VideoReader:
    """
    Factory function to create appropriate reader based on input type.
    Mirrors MATLAB get_video_file_reader functionality.

    Args:
        input_source: Path to video file, numpy array, VideoReader instance,
                     list of paths for multichannel, or folder for images
        buffer_size: Buffer size for reading
        bin_size: Temporal binning factor
        **kwargs: Additional reader-specific arguments

    Returns:
        Appropriate VideoReader subclass instance
    """
    from pathlib import Path

    # Handle numpy arrays
    if isinstance(input_source, np.ndarray):
        from pyflowreg.util.io._arr import ArrayReader

        return ArrayReader(input_source, buffer_size, bin_size)

    # Handle VideoReader instances (already initialized)
    if isinstance(input_source, VideoReader):
        return input_source

    # Import readers here to avoid circular imports
    from pyflowreg.util.io.tiff import TIFFFileReader
    from pyflowreg.util.io.hdf5 import HDF5FileReader
    from pyflowreg.util.io.mat import MATFileReader
    from pyflowreg.util.io.mdf import MDFFileReader
    from pyflowreg.util.io.multifile_wrappers import MULTICHANNELFileReader

    # Handle multichannel input (list of files)
    if isinstance(input_source, list):
        return MULTICHANNELFileReader(input_source, buffer_size, bin_size, **kwargs)

    # From here on, treat as file path
    file_path = input_source
    path = Path(file_path)

    # Handle folder input (image sequence) - TODO: implement IMGFileReader
    if path.is_dir():
        # Check if folder contains images
        image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        has_images = any(
            f.suffix.lower() in image_exts for f in path.iterdir() if f.is_file()
        )
        if has_images:
            # TODO: Implement IMGFileReader for image folders
            raise NotImplementedError(
                "Image folder reading not yet implemented. Use TIFF stacks instead."
            )
        else:
            raise ValueError(f"Folder {file_path} does not contain images")

    # Handle file input
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    readers = {
        ".tif": TIFFFileReader,
        ".tiff": TIFFFileReader,
        ".h5": HDF5FileReader,
        ".hdf5": HDF5FileReader,
        ".hdf": HDF5FileReader,
        ".mat": MATFileReader,
        ".mdf": MDFFileReader,
    }

    reader_class = readers.get(ext)
    if reader_class:
        return reader_class(str(file_path), buffer_size, bin_size, **kwargs)
    else:
        # Try to check if it's HDF5 without extension
        try:
            import h5py

            with h5py.File(str(file_path), "r"):
                return HDF5FileReader(str(file_path), buffer_size, bin_size, **kwargs)
        except Exception as e:
            # Not an HDF5 file - warn before raising error
            warnings.warn(f"File format detection failed: could not open as HDF5: {e}")

        # Try video formats as last resort
        # TODO: Implement AVIFileReader for video files
        raise ValueError(f"Unsupported file format: {ext}")


def get_video_file_writer(file_path: str, output_format: str, **kwargs) -> VideoWriter:
    """
    Factory function to create appropriate writer based on output format.
    Mirrors MATLAB get_video_file_writer functionality.

    Args:
        file_path: Output file path
        output_format: Output format string (e.g., 'TIFF', 'HDF5', 'MAT', 'MULTIFILE_TIFF', 'ARRAY', 'NULL', etc.)
                       Special formats:
                       - 'ARRAY': Returns ArrayWriter for in-memory accumulation
                       - 'NULL': Returns NullVideoWriter that discards all frames (useful for callbacks only)
        **kwargs: Additional writer-specific arguments

    Returns:
        Appropriate VideoWriter subclass instance
    """

    # Import writers here to avoid circular imports
    from pyflowreg.util.io.tiff import TIFFFileWriter
    from pyflowreg.util.io.hdf5 import HDF5FileWriter
    from pyflowreg.util.io.mat import MATFileWriter
    from pyflowreg.util.io.multifile_wrappers import MULTIFILEFileWriter

    # Special handling for memory formats
    if output_format == "ARRAY":
        from pyflowreg.util.io._arr import ArrayWriter

        return ArrayWriter()

    # NULL writer - discards output without storage
    if output_format == "NULL":
        from pyflowreg.util.io._null import NullVideoWriter

        return NullVideoWriter()

    # Handle different output formats (matches MATLAB switch statement)
    if output_format == "TIFF":
        return TIFFFileWriter(file_path, **kwargs)
    elif output_format == "SUITE2P_TIFF":
        # TODO: Add suite2p-specific formatting
        return TIFFFileWriter(file_path, format="suite2p", **kwargs)
    elif output_format == "MAT":
        return MATFileWriter(file_path, **kwargs)
    elif output_format == "HDF5":
        return HDF5FileWriter(file_path, **kwargs)
    elif output_format == "MULTIFILE_TIFF":
        return MULTIFILEFileWriter(file_path, "TIFF", **kwargs)
    elif output_format == "MULTIFILE_MAT":
        return MULTIFILEFileWriter(file_path, "MAT", **kwargs)
    elif output_format == "MULTIFILE_HDF5":
        return MULTIFILEFileWriter(file_path, "HDF5", **kwargs)
    elif output_format == "CAIMAN_HDF5":
        # Multifile HDF5 with /mov dataset for CaImAn compatibility
        return MULTIFILEFileWriter(file_path, "HDF5", dataset_names="/mov", **kwargs)
    elif output_format == "BEGONIA":
        # TODO: Implement TSERIESH5_file_writer
        raise NotImplementedError("BEGONIA format not yet implemented")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def main():
    """Test wrapper implementations."""
    import tempfile
    from multifile_wrappers import (
        MULTICHANNELFileReader,
        SUBSETFileReader,
        MULTIFILEFileWriter,
    )

    # Create test data
    test_frames = np.random.randint(0, 255, (20, 64, 64, 2), dtype=np.uint8)

    # Test MULTIFILE writer
    print("Testing MULTIFILE writer...")
    with tempfile.TemporaryDirectory() as tmpdir:
        multifile_path = Path(tmpdir) / "test_multi"

        with MULTIFILEFileWriter(str(multifile_path), "TIFF") as writer:
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

    print("\n✓ All wrapper tests passed!")


if __name__ == "__main__":
    main()
