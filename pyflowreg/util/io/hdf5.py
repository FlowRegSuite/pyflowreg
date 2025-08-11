from pyflowreg.util.io._base import VideoReader, VideoWriter
from pyflowreg.util.io._ds_io import DSFileReader, DSFileWriter
import h5py
import numpy as np
from typing import Optional
import os


class HDF5FileReader(DSFileReader, VideoReader):
    """
    Reads video data from an HDF5 file. It inherits the dataset finding
    heuristic from DSFileReader and the core reader interface from VideoReader.
    """

    def __init__(self, input_file: str, buffer_size: int = 500, bin_size: int = 1, **kwargs):
        # Call initializers of parent classes
        DSFileReader.__init__(self)
        VideoReader.__init__(self)

        self.file_path = input_file
        self.buffer_size = buffer_size
        self.bin_size = bin_size
        self.dimension_ordering = kwargs.get('dimension_ordering', (0, 1, 2))  # H, W, T

        try:
            self.h5file = h5py.File(self.file_path, 'r')
        except Exception as e:
            raise IOError(f"Could not open HDF5 file: {self.file_path}. Error: {e}")

        user_specified_datasets = kwargs.get('dataset_names')
        if user_specified_datasets:
            self.dataset_names = [user_specified_datasets] if isinstance(user_specified_datasets,
                                                                         str) else user_specified_datasets
        else:
            datasets_with_info = []

            def visitor_func(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets_with_info.append((name, obj.shape))

            self.h5file.visititems(visitor_func)

            self.dataset_names = self._find_datasets(datasets_with_info)

        if not self.dataset_names:
            self.close()
            raise IOError(f"Could not find any suitable video datasets in '{self.file_path}'.")

        self._configure_properties()
        self.reset()

    def _configure_properties(self):
        first_ds = self.h5file[self.dataset_names[0]]
        shape = first_ds.shape

        if len(shape) < 2:
            raise ValueError("Dataset must have at least 2 dimensions.")

        if len(shape) > 2:
            time_dim_index = np.argmax(shape)
            self.frame_count = shape[time_dim_index]
            hw_dims = [i for i in range(len(shape)) if i != time_dim_index]
            self.height, self.width = shape[hw_dims[0]], shape[hw_dims[1]]
            self._internal_dim_order = (hw_dims[0], hw_dims[1], time_dim_index)
        else:
            self.height, self.width = shape
            self.frame_count = 1
            self._internal_dim_order = (0, 1, -1)  # -1 indicates no time dimension

        self.n_channels = len(self.dataset_names)
        self.dtype = first_ds.dtype
        self.bit_depth = self.dtype.itemsize * 8

    def read_batch(self) -> Optional[np.ndarray]:
        if not self.has_batch():
            return None

        start_frame = self.current_frame
        end_frame = min(self.current_frame + self.buffer_size, self.frame_count)
        num_frames = end_frame - start_frame

        batch_data = np.zeros((self.height, self.width, self.n_channels, num_frames), dtype=self.dtype)

        h_idx, w_idx, t_idx = self._internal_dim_order

        for i, ds_name in enumerate(self.dataset_names):
            dataset = self.h5file[ds_name]

            # Construct the slice tuple for reading from disk
            slices = [slice(None)] * dataset.ndim
            if t_idx != -1:
                slices[t_idx] = slice(start_frame, end_frame)

            data_slice = dataset[tuple(slices)]

            # Transpose from on-disk order to the standard (H, W, T) memory order
            # The destination order is always (H, W, T)
            transpose_order = [h_idx, w_idx]
            if t_idx != -1:
                transpose_order.append(t_idx)

            # We need to find the inverse permutation to bring it to H, W, T
            dest_order = np.argsort(transpose_order)

            batch_data[:, :, i, :] = np.transpose(data_slice, dest_order)

        self.current_frame = end_frame
        return batch_data

    def has_batch(self) -> bool:
        return self.current_frame < self.frame_count

    def reset(self):
        self.current_frame = 0

    def close(self):
        if hasattr(self, 'h5file') and self.h5file:
            self.h5file.close()
            self.h5file = None
            print("HDF5 file closed.")


class HDF5FileWriter(DSFileWriter, VideoWriter):
    """
    Writes video data to an HDF5 file, supporting chunking and multiple channels.
    Inherits dataset naming from DSFileWriter and the interface from VideoWriter.
    """

    def __init__(self, file_name: str, **kwargs):
        # Call initializers of parent classes
        DSFileWriter.__init__(self, **kwargs)
        VideoWriter.__init__(self)

        self.file_name = file_name
        self._h5file = None
        self._frame_counter = 0

    def write_frames(self, frames: np.ndarray):
        """
        Writes a batch of frames. The first call will create and configure the file.
        """
        # Ensure frames are at least 4D for consistent processing
        if frames.ndim == 2:
            frames = frames[:, :, np.newaxis, np.newaxis]  # H, W, C=1, T=1
        elif frames.ndim == 3:
            frames = frames[:, :, :, np.newaxis]  # H, W, C, T=1

        if not self.initialized:
            self.init(frames)
            self._create_file_and_datasets()

        num_frames = frames.shape[3]

        for i in range(self.n_channels):
            ds_name = self.get_ds_name(i + 1, self.n_channels)
            dataset = self._h5file[ds_name]

            # Resize the dataset to accommodate the new frames
            dataset.resize(self._frame_counter + num_frames, axis=2)

            # Get data for the current channel and permute it
            channel_data = frames[:, :, i, :]
            permuted_data = np.transpose(channel_data, self.dimension_ordering)

            # Write the new frames to the end of the dataset
            dataset[:, :, self._frame_counter:] = permuted_data

        self._frame_counter += num_frames

    def _create_file_and_datasets(self):
        """Internal method to create the HDF5 file and datasets on first write."""
        if os.path.exists(self.file_name):
            os.remove(self.file_name)

        self._h5file = h5py.File(self.file_name, 'w')

        # Define shape and chunking for optimal performance
        # Dataspace is infinitely expandable along the time axis (last dimension)
        dataspace = (self.height, self.width, None)
        chunksize = (self.height, self.width, 1)

        # Permute shape and chunking according to the specified dimension ordering
        permuted_dataspace = tuple(dataspace[i] for i in self.dimension_ordering)
        permuted_chunksize = tuple(chunksize[i] for i in self.dimension_ordering)

        for i in range(self.n_channels):
            ds_name = self.get_ds_name(i + 1, self.n_channels)
            self._h5file.create_dataset(
                name=ds_name,
                shape=permuted_dataspace,
                maxshape=permuted_dataspace,
                dtype=self.dtype,
                chunks=permuted_chunksize
            )

    def close(self):
        """Closes the HDF5 file handle."""
        if self._h5file:
            self._h5file.close()
            self._h5file = None
            print("HDF5 file writer closed.")


if __name__ == "__main__":
    pass