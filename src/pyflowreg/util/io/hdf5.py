import os
from typing import Union, List

import h5py
import numpy as np

from pyflowreg.util.io._base import VideoReader, VideoWriter
from pyflowreg.util.io._ds_io import DSFileReader, DSFileWriter


class HDF5FileReader(DSFileReader, VideoReader):
    """HDF5 video file reader with dataset discovery."""

    def __init__(
        self, file_path: str, buffer_size: int = 500, bin_size: int = 1, **kwargs
    ):
        # Initialize parent classes
        DSFileReader.__init__(self)
        VideoReader.__init__(self)

        self.file_path = file_path
        self.buffer_size = buffer_size
        self.bin_size = bin_size
        self.h5file = None

        # Dataset-specific options
        self.dataset_names = kwargs.get("dataset_names")
        self.dimension_ordering = kwargs.get("dimension_ordering")

    def _initialize(self):
        """Open file and set up properties."""
        try:
            self.h5file = h5py.File(self.file_path, "r")
        except Exception as e:
            raise IOError(f"Cannot open HDF5 file: {e}")

        # Use DSFileReader mixin to find datasets
        if not self.dataset_names:
            datasets_info = []

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets_info.append((name, obj.shape))

            self.h5file.visititems(visitor)
            self.dataset_names = self._find_datasets(datasets_info)

        if not self.dataset_names:
            raise ValueError("No suitable datasets found")

        # Set properties from first dataset
        first_ds = self.h5file[self.dataset_names[0]]
        shape = first_ds.shape

        # Detect dimension ordering (implementation specific)
        # For now assume it's already (T, H, W) or needs transposing
        if len(shape) == 3:
            self.frame_count, self.height, self.width = shape
            self.n_channels = len(self.dataset_names)
        elif len(shape) == 4:
            self.frame_count, self.height, self.width, self.n_channels = shape

        self.dtype = first_ds.dtype

    def _read_raw_frames(self, frame_indices: Union[slice, List[int]]) -> np.ndarray:
        """Read raw frames from HDF5 file."""
        # Convert list to slice if contiguous
        if isinstance(frame_indices, list):
            if len(frame_indices) == 0:
                return np.empty(
                    (0, self.height, self.width, self.n_channels), dtype=self.dtype
                )

            # Check if contiguous
            if len(frame_indices) > 1:
                diffs = np.diff(frame_indices)
                if np.all(diffs == 1):
                    frame_indices = slice(frame_indices[0], frame_indices[-1] + 1)

        # Allocate output
        if isinstance(frame_indices, slice):
            start, stop, step = frame_indices.indices(self.frame_count)
            indices = range(start, stop, step)
        else:
            indices = frame_indices

        n_frames = len(indices)
        output = np.zeros(
            (n_frames, self.height, self.width, self.n_channels), dtype=self.dtype
        )

        # Read from each dataset/channel
        for ch_idx, ds_name in enumerate(self.dataset_names):
            dataset = self.h5file[ds_name]

            if isinstance(frame_indices, slice):
                # Efficient slicing for contiguous frames
                data = dataset[frame_indices, :, :]
            else:
                # Fancy indexing for non-contiguous
                data = dataset[indices, :, :]

            output[:, :, :, ch_idx] = data

        return output

    def close(self):
        """Close HDF5 file."""
        if self.h5file:
            self.h5file.close()
            self.h5file = None


class HDF5FileWriter(DSFileWriter, VideoWriter):
    """
    HDF5 video file writer with MATLAB compatibility.

    Accepts frames in Python format (T, H, W, C) but stores them
    in MATLAB-compatible format as separate 3D datasets per channel
    with configurable dimension ordering.
    """

    def __init__(self, file_path: str, **kwargs):
        """
        Initialize HDF5 writer.

        Args:
            file_path: Output file path
            dataset_names: Optional dataset naming pattern or list
                          Default: 'ch*' (produces ch1, ch2, etc.)
            dimension_ordering: Storage order for MATLAB compatibility
                               Default: (0, 1, 2) for (H, W, T) storage
            compression: HDF5 compression ('gzip', 'lzf', or None)
            compression_level: Compression level for gzip (1-9)
            chunk_size: Chunk size for temporal dimension (default: 1)
        """
        # Initialize parent classes
        DSFileWriter.__init__(self, **kwargs)
        VideoWriter.__init__(self)

        self.file_path = file_path
        self._h5file = None
        self._datasets = {}
        self._frame_counter = 0

        # MATLAB compatibility options
        # Default (1, 2, 0) means store as (T, H, W) which MATLAB reads as (H, W, T)
        self.dimension_ordering = kwargs.get("dimension_ordering", (1, 2, 0))

        # Compression options
        self.compression = kwargs.get("compression", None)
        self.compression_level = kwargs.get("compression_level", 4)
        self.chunk_temporal = kwargs.get("chunk_size", 1)

        # Dataset naming - default to MATLAB convention
        if not self.dataset_names:
            self.dataset_names = "ch*"  # Will produce ch1, ch2, etc.

    def _create_datasets(self):
        """Create HDF5 datasets for each channel."""
        # Remove existing file if it exists
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        self._h5file = h5py.File(self.file_path, "w")

        # Define initial shape and max shape for expandable datasets
        # We store in MATLAB format: separate 3D datasets per channel
        initial_shape = [None, None, None]
        initial_shape[self.dimension_ordering[0]] = self.height
        initial_shape[self.dimension_ordering[1]] = self.width
        initial_shape[self.dimension_ordering[2]] = 0  # Start with 0 frames
        initial_shape = tuple(initial_shape)

        max_shape = [None, None, None]
        max_shape[self.dimension_ordering[0]] = self.height
        max_shape[self.dimension_ordering[1]] = self.width
        max_shape[self.dimension_ordering[2]] = None  # Unlimited frames
        max_shape = tuple(max_shape)

        # Chunking for efficient I/O
        chunk_shape = [None, None, None]
        chunk_shape[self.dimension_ordering[0]] = self.height
        chunk_shape[self.dimension_ordering[1]] = self.width
        chunk_shape[self.dimension_ordering[2]] = self.chunk_temporal
        chunk_shape = tuple(chunk_shape)

        # Create a dataset for each channel
        for ch_idx in range(self.n_channels):
            ds_name = self.get_ds_name(ch_idx + 1, self.n_channels)

            # Create expandable dataset
            if self.compression:
                if self.compression == "gzip":
                    ds = self._h5file.create_dataset(
                        name=ds_name,
                        shape=initial_shape,
                        maxshape=max_shape,
                        dtype=self.dtype,
                        chunks=chunk_shape,
                        compression="gzip",
                        compression_opts=self.compression_level,
                    )
                else:
                    ds = self._h5file.create_dataset(
                        name=ds_name,
                        shape=initial_shape,
                        maxshape=max_shape,
                        dtype=self.dtype,
                        chunks=chunk_shape,
                        compression=self.compression,
                    )
            else:
                ds = self._h5file.create_dataset(
                    name=ds_name,
                    shape=initial_shape,
                    maxshape=max_shape,
                    dtype=self.dtype,
                    chunks=chunk_shape,
                )

            self._datasets[ds_name] = ds

            # Add MATLAB-friendly attributes
            ds.attrs["dimension_ordering"] = self.dimension_ordering
            ds.attrs["original_shape_THWC"] = (
                0,
                self.height,
                self.width,
                self.n_channels,
            )

    def write_frames(self, frames: np.ndarray):
        """
        Write frames to HDF5 file.

        Args:
            frames: Array with shape (T, H, W, C) or (T, H, W) or (H, W)
        """
        # Normalize input to 4D (T, H, W, C)
        if frames.ndim == 2:  # Single frame, single channel (H, W)
            frames = frames[np.newaxis, :, :, np.newaxis]
        elif frames.ndim == 3:
            if frames.shape[0] == self.height and frames.shape[1] == self.width:
                # Single frame, multiple channels (H, W, C)
                frames = frames[np.newaxis, :, :, :]
            else:
                # Multiple frames, single channel (T, H, W)
                frames = frames[:, :, :, np.newaxis]
        elif frames.ndim != 4:
            raise ValueError(f"Expected 2D, 3D or 4D input, got {frames.ndim}D")

        # Initialize on first write
        if not self.initialized:
            T, H, W, C = frames.shape
            self.height = H
            self.width = W
            self.n_channels = C
            self.dtype = frames.dtype
            self.initialized = True
            self._create_datasets()

        # Validate shape
        T, H, W, C = frames.shape
        if H != self.height or W != self.width:
            raise ValueError(
                f"Frame size mismatch. Expected ({self.height}, {self.width}), "
                f"got ({H}, {W})"
            )
        if C != self.n_channels:
            raise ValueError(
                f"Channel count mismatch. Expected {self.n_channels}, got {C}"
            )

        # Write each channel separately for MATLAB compatibility
        for ch_idx in range(self.n_channels):
            ds_name = self.get_ds_name(ch_idx + 1, self.n_channels)
            dataset = self._datasets[ds_name]

            # Extract channel data: (T, H, W)
            channel_data = frames[:, :, :, ch_idx]

            # Transpose to MATLAB storage order
            # From (T, H, W) to dimension_ordering
            if self.dimension_ordering != (2, 0, 1):
                # Create mapping from current (T=0, H=1, W=2) to target ordering
                # Default MATLAB is (H=0, W=1, T=2)
                perm = [None, None, None]
                perm[self.dimension_ordering[0]] = 1  # H position
                perm[self.dimension_ordering[1]] = 2  # W position
                perm[self.dimension_ordering[2]] = 0  # T position
                channel_data = np.transpose(channel_data, perm)

            # Determine where to write in the dataset
            current_frames = dataset.shape[self.dimension_ordering[2]]
            new_total_frames = current_frames + T

            # Resize dataset along time dimension
            new_shape = list(dataset.shape)
            new_shape[self.dimension_ordering[2]] = new_total_frames
            dataset.resize(new_shape)

            # Create slice objects for writing
            slices = [slice(None), slice(None), slice(None)]
            slices[self.dimension_ordering[2]] = slice(current_frames, new_total_frames)

            # Write the data
            dataset[tuple(slices)] = channel_data

            # Update attributes
            dataset.attrs["original_shape_THWC"] = (
                new_total_frames,
                H,
                W,
                self.n_channels,
            )

        self._frame_counter = new_total_frames

        # Flush to ensure data is written
        if self._h5file:
            self._h5file.flush()

    def close(self):
        """Close the HDF5 file."""
        if self._h5file:
            # Write final metadata for MATLAB compatibility
            if self._datasets:
                # Add file-level attributes
                self._h5file.attrs["n_channels"] = self.n_channels
                self._h5file.attrs["frame_count"] = self._frame_counter
                self._h5file.attrs["height"] = self.height
                self._h5file.attrs["width"] = self.width
                self._h5file.attrs["dimension_ordering"] = self.dimension_ordering
                self._h5file.attrs["format"] = "pyflowreg_hdf5_v1"

                # Store dataset names as attribute for easy discovery
                dataset_names_list = list(self._datasets.keys())
                self._h5file.attrs["dataset_names"] = dataset_names_list

            self._h5file.close()
            self._h5file = None
            self._datasets = {}
            print(f"HDF5 file written: {self.file_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    import numpy as np
    from pathlib import Path
    from mdf import MDFFileReader
    import cv2

    filename = r"D:\2025_OIST\Shinobu\RFPonly\190403_001.MDF"
    out_path = Path(filename + ".hdf")

    mdf = MDFFileReader(filename, buffer_size=500, bin_size=1)

    with HDF5FileWriter(str(out_path)) as w:
        # for i in range(5 * 8200, 5 * 9200):
        for i in range(5 * 8200, 5 * 8300):
            frame = mdf[i]
            w.write_frames(frame[np.newaxis])

    h5 = HDF5FileReader(str(out_path), buffer_size=500, bin_size=5)
    h5_b5 = h5[0:20]
    h5.close()
    mdf.close()

    mdf2 = MDFFileReader(filename, buffer_size=500, bin_size=5)
    mdf_b5 = mdf2[8200 : 8200 + 20]
    mdf2.close()

    counter = 0
    while True:
        frame = np.concatenate([h5_b5[counter], mdf_b5[counter]], axis=0)
        counter = (counter + 1) % h5_b5.shape[0]
        cv2.imshow(
            "Frame",
            cv2.normalize(
                frame[..., 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            ),
        )
        key = cv2.waitKey(1)
        if key == 27:
            break

    if not np.array_equal(h5_b5, mdf_b5):
        d = h5_b5.astype(np.int64) - mdf_b5.astype(np.int64)
        print(int(np.abs(d).max()))
        print("Frames are not equal!")
    else:
        print(f"OK {out_path}")


def reader_main():
    import cv2

    filename = r"D:\2025_OIST\Shinobu\RFPonly\test.hdf"
    reader = HDF5FileReader(filename, buffer_size=500, bin_size=1)
    print(f"Number of frames: {len(reader)}")
    for i in range(len(reader)):
        frame = reader[i]
        cv2.imshow(
            "Frame",
            cv2.normalize(
                frame[:, :, 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            ),
        )
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
