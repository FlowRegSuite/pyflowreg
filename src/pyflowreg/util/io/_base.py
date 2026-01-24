from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple

import numpy as np


class VideoReader(ABC):
    """
    Abstract base class for all video file readers.

    Data is returned in (T, H, W, C) format:
    - T: Time/frames
    - H: Height
    - W: Width
    - C: Channels

    This format is optimal for OpenCV operations and can be easily
    converted to PyTorch format (T, C, H, W) when needed.
    """

    def __init__(self):
        # Core properties - set by _initialize()
        self.height: int = 0
        self.width: int = 0
        self.frame_count: int = 0
        self.n_channels: int = 0
        self.dtype: Optional[np.dtype] = None

        # Reader configuration
        self.buffer_size: int = 500
        self.bin_size: int = 1

        # State tracking
        self.current_frame: int = 0
        self._initialized: bool = False

    @abstractmethod
    def _initialize(self):
        """
        Initialize file-specific properties.
        Must set: height, width, frame_count, n_channels, dtype
        """
        pass

    @abstractmethod
    def _read_raw_frames(self, frame_indices: Union[slice, List[int]]) -> np.ndarray:
        """
        Read raw frames from the underlying file.

        Args:
            frame_indices: Either a slice object or list of 0-based indices

        Returns:
            Array with shape (T, H, W, C) containing raw frames
        """
        pass

    @abstractmethod
    def close(self):
        """Close file handles and clean up resources."""
        pass

    def _ensure_initialized(self):
        """Ensure the reader is initialized before operations."""
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def bin_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply temporal binning to reduce frame count.

        Args:
            frames: Input array with shape (T, H, W, C)

        Returns:
            Binned array with shape (T//bin_size, H, W, C)
        """
        if self.bin_size == 1:
            return frames

        input_dtype = frames.dtype

        if frames.ndim != 4:
            raise ValueError(f"Expected 4D array (T, H, W, C), got {frames.ndim}D")

        T, H, W, C = frames.shape

        # Pad to make divisible by bin_size
        pad = (-T) % self.bin_size
        if pad:
            frames = np.pad(frames, [(0, pad), (0, 0), (0, 0), (0, 0)], mode="edge")
            T = frames.shape[0]

        # Reshape and average
        frames = frames.reshape(T // self.bin_size, self.bin_size, H, W, C)
        frames = frames.mean(axis=1).astype(input_dtype)

        return frames

    def __getitem__(self, key: Union[int, slice, Tuple]) -> np.ndarray:
        """
        Array-like indexing with automatic binning.

        With bin_size > 1, indices refer to binned frames:
        - reader[0] returns average of first bin_size frames
        - reader[1] returns average of next bin_size frames

        Returns:
            Single frame: (H, W, C)
            Multiple frames: (T, H, W, C)
        """
        self._ensure_initialized()

        # Calculate binned frame count
        binned_count = (self.frame_count + self.bin_size - 1) // self.bin_size

        # Handle single integer
        if isinstance(key, int):
            if key < 0:
                key = binned_count + key
            if key < 0 or key >= binned_count:
                raise IndexError(
                    f"Index {key} out of range for {binned_count} binned frames"
                )

            # Get raw frame range for this bin
            start = key * self.bin_size
            end = min((key + 1) * self.bin_size, self.frame_count)

            # Read and average frames
            raw_frames = self._read_raw_frames(slice(start, end))
            binned_frame = raw_frames.mean(axis=0).astype(raw_frames.dtype)
            return binned_frame

        # Handle slice
        elif isinstance(key, slice):
            start, stop, step = key.indices(binned_count)

            if start >= stop:
                return np.empty(
                    (0, self.height, self.width, self.n_channels), dtype=self.dtype
                )

            # Collect all requested bins
            binned_frames = []
            for bin_idx in range(start, stop, step):
                frame_start = bin_idx * self.bin_size
                frame_end = min((bin_idx + 1) * self.bin_size, self.frame_count)

                raw_frames = self._read_raw_frames(slice(frame_start, frame_end))
                binned_frame = raw_frames.mean(axis=0, keepdims=True).astype(
                    raw_frames.dtype
                )
                binned_frames.append(binned_frame)

            return np.concatenate(binned_frames, axis=0)

        # Handle list or numpy array (fancy indexing)
        elif isinstance(key, (list, np.ndarray)):
            # Convert to numpy array if it's a list
            indices = np.asarray(key, dtype=np.int64)

            # Handle negative indices
            indices = np.where(indices < 0, binned_count + indices, indices)

            # Check bounds
            if np.any(indices < 0) or np.any(indices >= binned_count):
                raise IndexError(f"Index out of range for {binned_count} binned frames")

            # Collect frames at specified indices
            frames_list = []
            for idx in indices:
                idx = int(idx)  # Ensure it's a Python int
                frame_start = idx * self.bin_size
                frame_end = min((idx + 1) * self.bin_size, self.frame_count)

                raw_frames = self._read_raw_frames(slice(frame_start, frame_end))
                binned_frame = raw_frames.mean(axis=0, keepdims=True).astype(
                    raw_frames.dtype
                )
                frames_list.append(binned_frame)

            # Return (T, H, W, C) for consistency with slice indexing
            return np.concatenate(frames_list, axis=0)

        # Handle tuple for advanced indexing
        elif isinstance(key, tuple):
            frame_key, *rest = key

            # Get frames first
            if isinstance(frame_key, int):
                frames = self[frame_key]  # Returns (H, W, C)
                frames = frames[np.newaxis, ...]  # Add T dimension back
            else:
                frames = self[frame_key]  # Returns (T, H, W, C)

            # Apply additional slicing
            if rest:
                # Convert to handle both (T, ...) and direct (...) slicing
                if frames.ndim == 4:  # Has time dimension
                    full_key = (slice(None),) + tuple(rest)
                else:  # Single frame, no time dimension
                    full_key = tuple(rest)
                frames = frames[full_key]

            return frames

        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def read_batch(self) -> Optional[np.ndarray]:
        """
        Read next batch of frames with binning.

        Returns:
            Array with shape (T, H, W, C) or None if no more frames
        """
        self._ensure_initialized()

        if not self.has_batch():
            return None

        # Calculate frames to read
        frames_to_read = self.buffer_size * self.bin_size
        end_frame = min(self.current_frame + frames_to_read, self.frame_count)

        # Read raw frames
        raw_frames = self._read_raw_frames(slice(self.current_frame, end_frame))
        self.current_frame = end_frame

        # Apply binning
        return self.bin_frames(raw_frames)

    def has_batch(self) -> bool:
        """Check if more frames are available."""
        return self.current_frame < self.frame_count

    def reset(self):
        """Reset to beginning of file."""
        self.current_frame = 0

    def __len__(self) -> int:
        """Number of frames after binning."""
        self._ensure_initialized()
        return (self.frame_count + self.bin_size - 1) // self.bin_size

    def __iter__(self):
        """Make reader iterable."""
        self.reset()
        return self

    def __next__(self) -> np.ndarray:
        """Iterator protocol."""
        if not self.has_batch():
            raise StopIteration
        return self.read_batch()

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """
        Shape after binning.

        Returns
        -------
        tuple of int
            Shape as (T_binned, H, W, C)
        """
        self._ensure_initialized()
        return (len(self), self.height, self.width, self.n_channels)

    @property
    def unbinned_shape(self) -> Tuple[int, int, int, int]:
        """
        Original shape before binning.

        Returns
        -------
        tuple of int
            Shape as (T_original, H, W, C)
        """
        self._ensure_initialized()
        return (self.frame_count, self.height, self.width, self.n_channels)

    def to_pytorch(self, frames: np.ndarray) -> np.ndarray:
        """
        Convert from OpenCV (T, H, W, C) to PyTorch (T, C, H, W) format.
        """
        if frames.ndim == 3:  # Single frame (H, W, C)
            return np.transpose(frames, (2, 0, 1))
        elif frames.ndim == 4:  # Multiple frames (T, H, W, C)
            return np.transpose(frames, (0, 3, 1, 2))
        else:
            raise ValueError(f"Expected 3D or 4D array, got {frames.ndim}D")

    def __repr__(self):
        self._ensure_initialized()
        return (
            f"{self.__class__.__name__}(shape={self.shape}, "
            f"dtype={self.dtype}, bin_size={self.bin_size})"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoWriter(ABC):
    """
    Abstract base class for all video file writers.
    Defines a common interface for writing frames.
    """

    def __init__(self):
        self.initialized = False
        self.height = 0
        self.width = 0
        self.n_channels = 0
        self.bit_depth = 0
        self.dtype = None

    def init(self, first_frame_batch: np.ndarray):
        """Initializes writer properties based on the first batch of frames."""
        shape = first_frame_batch.shape
        self.height = shape[0]
        self.width = shape[1]
        self.n_channels = shape[2] if len(shape) > 2 else 1
        self.dtype = first_frame_batch.dtype
        self.bit_depth = self.dtype.itemsize * 8
        self.initialized = True

    @abstractmethod
    def write_frames(self, frames: np.ndarray):
        """Writes a batch of frames to the file."""
        pass

    @abstractmethod
    def close(self):
        """Closes the writer and finalizes the file."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
