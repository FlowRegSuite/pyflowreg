import os
from typing import Union, List
import warnings

import numpy as np
import scipy.io as sio
import h5py
import hdf5storage as h5s

from pyflowreg.util.io._base import VideoReader, VideoWriter
from pyflowreg.util.io._ds_io import DSFileReader, DSFileWriter


class MATFileReader(DSFileReader, VideoReader):
    """
    MAT video file reader with dataset discovery.
    Supports both traditional MAT files (v5, v7) and v7.3 (HDF5-based).
    """

    def __init__(
        self, file_path: str, buffer_size: int = 500, bin_size: int = 1, **kwargs
    ):
        # Initialize parent classes
        DSFileReader.__init__(self)
        VideoReader.__init__(self)

        self.file_path = file_path
        self.buffer_size = buffer_size
        self.bin_size = bin_size

        # MAT-specific
        self.mat_data = None
        self.is_v73 = False
        self.h5file = None  # For v7.3 files

        # Dataset options from kwargs
        self.dataset_names = kwargs.get("dataset_names")
        self.dimension_ordering = kwargs.get(
            "dimension_ordering", [0, 1, 2]
        )  # MATLAB default

        # Known dataset patterns from MATLAB version
        self.known_patterns = ["ch*_reg", "ch*", "buffer*", "mov", "data"]

    @staticmethod
    def _is_v73(path: str) -> bool:
        try:
            with open(path, "rb") as f:
                head = f.read(128)
            if b"MATLAB 7.3 MAT-file" in head:
                return True
        except Exception:
            pass
        return h5py.is_hdf5(path)

    def _initialize(self):
        """Open MAT file and set up properties."""
        if self._is_v73(self.file_path):
            self.is_v73 = True
            try:
                self.h5file = h5py.File(self.file_path, "r")
            except Exception as e:
                raise IOError(f"Cannot open MAT v7.3 (HDF5) file: {e}")
        else:
            try:
                self.mat_data = sio.loadmat(
                    self.file_path, verify_compressed_data_integrity=False
                )
                self.is_v73 = False
            except NotImplementedError:
                # Unexpected v7.3 despite header test
                self.is_v73 = True
                self.h5file = h5py.File(self.file_path, "r")
            except ValueError as e:
                raise IOError(f"Cannot open MAT v7.3 file: {e}")

        # Find datasets
        if not self.dataset_names:
            if self.is_v73:
                datasets_info = self._find_datasets_v73()
            else:
                datasets_info = self._find_datasets_regular()

            self.dataset_names = self._find_datasets(datasets_info)

        if not self.dataset_names:
            raise ValueError("No suitable datasets found in MAT file")

        # Verify and setup properties from first dataset
        self._setup_properties()

    def _find_datasets_regular(self):
        """Find datasets in regular MAT files."""
        datasets_info = []

        for key in self.mat_data.keys():
            # Skip metadata keys
            if key.startswith("__"):
                continue

            data = self.mat_data[key]
            if isinstance(data, np.ndarray) and data.ndim == 3:
                datasets_info.append((key, data.shape))

        return datasets_info

    def _find_datasets_v73(self):
        """Find datasets in v7.3 MAT files."""
        datasets_info = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
                # Skip MATLAB metadata
                if not name.startswith("#"):
                    datasets_info.append((name, obj.shape))

        self.h5file.visititems(visitor)
        return datasets_info

    def _setup_properties(self):
        """Setup reader properties from discovered datasets."""
        if not self.dataset_names:
            raise ValueError("No datasets to setup properties from")

        # Get first dataset to determine properties
        if self.is_v73:
            first_ds = self.h5file[self.dataset_names[0]]
            shape = first_ds.shape
            self.dtype = first_ds.dtype
        else:
            first_ds = self.mat_data[self.dataset_names[0]]
            shape = first_ds.shape
            self.dtype = first_ds.dtype

        # Map from MATLAB dimension ordering to properties
        # MATLAB: [height, width, time] by default
        # Python: expecting (T, H, W, C)
        self.height = shape[self.dimension_ordering[0]]
        self.width = shape[self.dimension_ordering[1]]
        self.frame_count = shape[self.dimension_ordering[2]]
        self.n_channels = len(self.dataset_names)

        # Legacy compatibility
        self.m = self.height
        self.n = self.width
        self.mat_data_type = str(self.dtype)

        # Verify all datasets have same shape
        for ds_name in self.dataset_names[1:]:
            if self.is_v73:
                ds_shape = self.h5file[ds_name].shape
            else:
                ds_shape = self.mat_data[ds_name].shape

            if ds_shape != shape:
                raise ValueError(
                    f"Dataset {ds_name} has different shape: {ds_shape} vs {shape}"
                )

    def _read_raw_frames(self, frame_indices: Union[slice, List[int]]) -> np.ndarray:
        """
        Read raw frames from MAT file.

        Returns:
            Array with shape (T, H, W, C)
        """
        # Convert list to array for indexing
        if isinstance(frame_indices, list):
            if len(frame_indices) == 0:
                return np.empty(
                    (0, self.height, self.width, self.n_channels), dtype=self.dtype
                )
            indices = np.array(frame_indices)
        else:
            # Convert slice to indices
            start, stop, step = frame_indices.indices(self.frame_count)
            indices = np.arange(start, stop, step)

        n_frames = len(indices)
        output = np.zeros(
            (n_frames, self.height, self.width, self.n_channels), dtype=self.dtype
        )

        # Read from each dataset/channel
        for ch_idx, ds_name in enumerate(self.dataset_names):
            if self.is_v73:
                data = self._read_v73_dataset(ds_name, indices)
            else:
                data = self._read_regular_dataset(ds_name, indices)

            # Store in output array
            output[:, :, :, ch_idx] = data

        return output

    def _read_regular_dataset(self, ds_name: str, indices: np.ndarray) -> np.ndarray:
        """Read from regular MAT file dataset."""
        dataset = self.mat_data[ds_name]

        # Create index arrays for each dimension
        idx = [slice(None), slice(None), slice(None)]
        idx[self.dimension_ordering[2]] = indices

        # Read data with proper ordering
        data = dataset[tuple(idx)]

        # Permute to (T, H, W) format
        if self.dimension_ordering != [2, 0, 1]:
            # Create inverse permutation - find where each output dim comes from
            perm = [None, None, None]
            perm[0] = self.dimension_ordering[2]  # T comes from stored time position
            perm[1] = self.dimension_ordering[0]  # H comes from stored height position
            perm[2] = self.dimension_ordering[1]  # W comes from stored width position
            data = np.transpose(data, perm)

        return data

    def _read_v73_dataset(self, ds_name: str, indices: np.ndarray) -> np.ndarray:
        """Read from v7.3 MAT file dataset."""
        dataset = self.h5file[ds_name]

        # Check if indices are contiguous for efficient reading
        if len(indices) > 1 and np.all(np.diff(indices) == 1):
            # Contiguous - use slicing
            idx = [slice(None), slice(None), slice(None)]
            idx[self.dimension_ordering[2]] = slice(indices[0], indices[-1] + 1)
            data = dataset[tuple(idx)]
        else:
            # Non-contiguous - read individually
            n_frames = len(indices)
            shape = [dataset.shape[0], dataset.shape[1], dataset.shape[2]]
            shape[self.dimension_ordering[2]] = n_frames
            data = np.zeros(shape, dtype=self.dtype)

            for i, frame_idx in enumerate(indices):
                idx_src = [slice(None), slice(None), slice(None)]
                idx_src[self.dimension_ordering[2]] = frame_idx

                idx_dst = [slice(None), slice(None), slice(None)]
                idx_dst[self.dimension_ordering[2]] = i

                data[tuple(idx_dst)] = dataset[tuple(idx_src)]

        # Permute to (T, H, W) format
        if self.dimension_ordering != [2, 0, 1]:
            # Create inverse permutation - find where each output dim comes from
            perm = [None, None, None]
            perm[0] = self.dimension_ordering[2]  # T comes from stored time position
            perm[1] = self.dimension_ordering[0]  # H comes from stored height position
            perm[2] = self.dimension_ordering[1]  # W comes from stored width position
            data = np.transpose(data, perm)

        return data

    def close(self):
        """Close MAT file."""
        if self.h5file:
            self.h5file.close()
            self.h5file = None
        self.mat_data = None


class MATFileWriter(DSFileWriter, VideoWriter):
    """
    MAT video file writer with MATLAB compatibility.

    Creates MAT files with separate 3D datasets per channel,
    stored in MATLAB-compatible dimension ordering.
    """

    def __init__(self, file_path: str, **kwargs):
        """
        Initialize MAT writer.

        Args:
            file_path: Output file path
            dataset_names: Optional dataset naming pattern or list
                          Default: 'ch*' (produces ch1, ch2, etc.)
            dimension_ordering: Storage order for MATLAB compatibility
                               Default: [0, 1, 2] for (H, W, T) in MATLAB
            use_v73: Force v7.3 format (HDF5-based) for large files
        """
        # Initialize parent classes
        DSFileWriter.__init__(self, **kwargs)
        VideoWriter.__init__(self)

        self.file_path = file_path
        self.use_v73 = kwargs.get("use_v73", False)
        self._data_dict = {}
        self._frame_counter = 0

        # MATLAB compatibility options
        self.dimension_ordering = kwargs.get("dimension_ordering", [0, 1, 2])

        # Dataset naming
        if not self.dataset_names:
            self.dataset_names = "ch*"

    def write_frames(self, frames: np.ndarray):
        """
        Write frames to MAT file buffers.

        Args:
            frames: Array with shape (T, H, W, C) or (T, H, W) or (H, W)
        """
        # Normalize input to 4D (T, H, W, C)
        if frames.ndim == 2:  # Single frame, single channel
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

            # Initialize data buffers for each channel
            for ch_idx in range(self.n_channels):
                ds_name = self.get_ds_name(ch_idx + 1, self.n_channels)
                self._data_dict[ds_name] = []

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

        # Accumulate frames for each channel
        for ch_idx in range(self.n_channels):
            ds_name = self.get_ds_name(ch_idx + 1, self.n_channels)
            channel_data = frames[:, :, :, ch_idx]  # (T, H, W)

            # Convert to MATLAB dimension ordering
            if self.dimension_ordering != [2, 0, 1]:
                perm = [None, None, None]
                perm[self.dimension_ordering[0]] = 1  # H position
                perm[self.dimension_ordering[1]] = 2  # W position
                perm[self.dimension_ordering[2]] = 0  # T position
                channel_data = np.transpose(channel_data, perm)

            self._data_dict[ds_name].append(channel_data)

        self._frame_counter += T

    def close(self):
        """Close and write the MAT file."""
        if not self._data_dict:
            return

        # Concatenate accumulated frames for each channel
        final_dict = {}
        for ds_name, frame_list in self._data_dict.items():
            if frame_list:
                # Concatenate along time dimension
                concat_axis = self.dimension_ordering[2]
                final_dict[ds_name] = np.concatenate(frame_list, axis=concat_axis)

        # Add metadata
        final_dict["__pyflowreg_metadata__"] = {
            "n_channels": self.n_channels,
            "frame_count": self._frame_counter,
            "height": self.height,
            "width": self.width,
            "dimension_ordering": self.dimension_ordering,
            "format": "pyflowreg_mat_v1",
        }

        # Write MAT file
        # if self.use_v73:
        #    # Use v7.3 format for large files
        #    sio.savemat(self.file_path, final_dict, do_compression=True, format='7.3')
        # else:
        # Use default format
        #    try:
        #        sio.savemat(self.file_path, final_dict, do_compression=True)
        #    except ValueError:
        # File too large for v5/v7, switch to v7.3
        #        warnings.warn("File too large for MAT v5/v7, switching to v7.3 format")
        #        sio.savemat(self.file_path, final_dict, do_compression=True, format='7.3')

        try:
            if self.use_v73:
                h5s.savemat(self.file_path, final_dict)
            else:
                sio.savemat(self.file_path, final_dict, do_compression=True, format="5")
        except ValueError:
            warnings.warn("Switching to v7.3 (file too large for v5).")
            h5s.savemat(self.file_path, final_dict)

        print(f"MAT file written: {self.file_path}")
        self._data_dict = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Test MAT file I/O."""
    import tempfile

    # Create test data
    test_frames = np.random.randint(0, 255, (100, 128, 128, 2), dtype=np.uint8)

    # Test writing
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
        mat_path = f.name

    print(f"Writing test MAT file: {mat_path}")
    with MATFileWriter(mat_path, use_v73=True) as writer:
        writer.write_frames(test_frames[:50])
        writer.write_frames(test_frames[50:])

    # Test reading
    print(f"Reading test MAT file: {mat_path}")
    reader = MATFileReader(mat_path, buffer_size=10, bin_size=1)

    print(f"Shape: {reader.shape}")
    print(f"Channels: {reader.n_channels}")
    print(f"Frame count: {reader.frame_count}")

    # Test different access patterns
    single_frame = reader[0]
    print(f"Single frame shape: {single_frame.shape}")

    frame_slice = reader[10:20]
    print(f"Slice shape: {frame_slice.shape}")

    # Test batch reading
    reader.reset()
    batch = reader.read_batch()
    print(f"Batch shape: {batch.shape}")

    # Verify data integrity
    all_frames = reader[:]
    if np.array_equal(all_frames, test_frames):
        print("✓ Data integrity verified")
    else:
        print("✗ Data mismatch!")

    reader.close()

    # Cleanup
    os.unlink(mat_path)
    print("Test complete")


if __name__ == "__main__":
    main()
