"""
Private module for in-memory array I/O wrappers.
Allows numpy arrays to be processed through the same pipeline as video files.
"""

from typing import Union, List, Optional
import numpy as np

from ._base import VideoReader, VideoWriter


class ArrayReader(VideoReader):
    """
    Wraps numpy arrays to provide VideoReader interface.
    Enables batch processing and binning for in-memory arrays.
    """

    def __init__(
        self,
        array: np.ndarray,
        buffer_size: int = 100,
        bin_size: int = 1,
        inplace: bool = False,
    ):
        """
        Initialize array reader.

        Args:
            array: Input array with shape (T,H,W,C) or (H,W,C) or (T,H,W)
            buffer_size: Number of frames per batch
            bin_size: Temporal binning factor
            inplace: If True, return views for memory efficiency (no copy).
                     If False (default), return copies for safety with multiprocessing.
        """
        super().__init__()

        # Handle different input shapes - store reference, don't copy the whole array
        if array.ndim == 2:  # (H,W)
            self._array = array[np.newaxis, :, :, np.newaxis]  # -> (1,H,W,1)
        elif array.ndim == 3:
            # Could be (T,H,W) or (H,W,C)
            # Assume (H,W,C) if last dimension is small (<=4 channels typical)
            if array.shape[-1] <= 4:
                self._array = array[np.newaxis, ...]  # (H,W,C) -> (1,H,W,C)
            else:
                self._array = array[..., np.newaxis]  # (T,H,W) -> (T,H,W,1)
        elif array.ndim == 4:
            self._array = array  # Already (T,H,W,C)
        else:
            raise ValueError(f"Array must be 2D, 3D or 4D, got {array.ndim}D")

        self.buffer_size = buffer_size
        self.bin_size = bin_size
        self._inplace = inplace

        # Initialize immediately
        self._initialize()

    def _initialize(self):
        """Set VideoReader properties from array shape."""
        self.frame_count, self.height, self.width, self.n_channels = self._array.shape
        self.dtype = self._array.dtype
        self._initialized = True

    def _read_raw_frames(self, frame_indices: Union[slice, List[int]]) -> np.ndarray:
        """
        Read frames from array.

        Args:
            frame_indices: Slice or list of frame indices

        Returns:
            Array with shape (T,H,W,C), copy by default unless inplace=True
        """
        if isinstance(frame_indices, list):
            if len(frame_indices) == 0:
                return np.empty(
                    (0, self.height, self.width, self.n_channels), dtype=self.dtype
                )
            result = self._array[frame_indices]
        else:
            # slice
            result = self._array[frame_indices]

        # Return copy by default for safety with multiprocessing
        # Only return view if explicitly requested with inplace=True
        return result if self._inplace else result.copy()

    def close(self):
        """No-op for array reader."""
        pass


class ArrayWriter(VideoWriter):
    """
    Accumulates frames in memory instead of writing to file.
    Provides VideoWriter interface for array output.
    """

    def __init__(self):
        """Initialize array writer."""
        super().__init__()
        self._vid = []  # Accumulated video frames

    def init(self, first_frame_batch: np.ndarray):
        """
        Initialize writer from first batch following base class pattern.

        Args:
            first_frame_batch: First batch with shape (T,H,W,C) or (H,W,C) or even (H,W)
        """
        shape = first_frame_batch.shape

        if len(shape) == 2:
            # Single channel single frame (H,W)
            self.height = shape[0]
            self.width = shape[1]
            self.n_channels = 1
        elif len(shape) == 3:
            # Single frame (H,W,C)
            self.height = shape[0]
            self.width = shape[1]
            self.n_channels = shape[2] if len(shape) > 2 else 1
        elif len(shape) == 4:
            # Batch (T,H,W,C) - use first frame dimensions
            self.height = shape[1]
            self.width = shape[2]
            self.n_channels = shape[3]
        else:
            raise ValueError(
                f"Expected 2D, 3D or 4D array, got {first_frame_batch.ndim}D"
            )

        self.dtype = first_frame_batch.dtype
        self.bit_depth = self.dtype.itemsize * 8
        self.initialized = True

    def write_frames(self, frames: np.ndarray):
        """
        Accumulate frames in memory.

        Args:
            frames: Array with shape (T,H,W,C), (H,W,C), or (H,W)
        """
        if not self.initialized:
            self.init(frames)

        # Handle different input dimensions
        if frames.ndim == 2:
            # Single channel single frame (H,W) -> (1,H,W,1)
            frames = frames[np.newaxis, :, :, np.newaxis]
        elif frames.ndim == 3:
            # Single frame (H,W,C) -> (1,H,W,C)
            frames = frames[np.newaxis, ...]
        elif frames.ndim == 4:
            # Already batched (T,H,W,C)
            pass
        else:
            raise ValueError(f"Expected 2D, 3D or 4D array, got {frames.ndim}D")

        # Always copy to prevent external modifications
        self._vid.append(frames.copy())

    def get_array(self) -> Optional[np.ndarray]:
        """
        Fetch accumulated frames as single array.

        Returns:
            Concatenated frames or None if empty
        """
        if not self._vid:
            return None
        return np.concatenate(self._vid, axis=0)

    def close(self):
        """No-op for array writer."""
        pass

    def __repr__(self):
        n_frames = sum(f.shape[0] for f in self._vid) if self._vid else 0
        return f"ArrayWriter(frames={n_frames}, shape=({self.height},{self.width},{self.n_channels}))"
