"""
Private module for in-memory array I/O wrappers.
Allows numpy arrays to be processed through the same pipeline as video files.
"""

from typing import Union, List, Optional
import numpy as np

from ._base import VideoReader, VideoWriter


class ArrayReader(VideoReader):
    """
    In-memory array reader providing the VideoReader interface.

    Wraps a numpy array so it can be processed through the same pipeline
    as video files, enabling batch processing and temporal binning for
    in-memory data. Frames are returned in (T, H, W, C) format.
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

        Parameters
        ----------
        array : ndarray
            Input array with shape (T, H, W, C), (H, W), (H, W, C) or
            (T, H, W). 3D arrays are interpreted as (H, W, C) when the
            last dimension is at most 4, otherwise as (T, H, W).
        buffer_size : int, optional
            Number of frames per batch (default 100).
        bin_size : int, optional
            Temporal binning factor (default 1).
        inplace : bool, optional
            If True, return views for memory efficiency (no copy).
            If False (default), return copies for safety with
            multiprocessing.

        Raises
        ------
        ValueError
            If ``array`` is not 2D, 3D or 4D.
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
        Read frames from the wrapped array.

        Parameters
        ----------
        frame_indices : slice or list of int
            0-based raw frame indices.

        Returns
        -------
        ndarray
            Array with shape (T, H, W, C); a copy by default, or a view
            if the reader was created with ``inplace=True``.
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
        """Close the reader (no-op; there are no resources to release)."""
        pass


class ArrayWriter(VideoWriter):
    """
    In-memory writer that accumulates frames instead of writing to file.

    Provides the VideoWriter interface for array output; the accumulated
    frames can be retrieved as a single (T, H, W, C) array with
    ``get_array()``.
    """

    def __init__(self):
        """Initialize array writer."""
        super().__init__()
        self._vid = []  # Accumulated video frames

    def init(self, first_frame_batch: np.ndarray):
        """
        Initialize writer properties from the first batch.

        Parameters
        ----------
        first_frame_batch : ndarray
            First batch with shape (T, H, W, C), (H, W, C) or (H, W).

        Raises
        ------
        ValueError
            If the input is not a 2D, 3D or 4D array.
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

        A copy of the input is stored to prevent external modifications.

        Parameters
        ----------
        frames : ndarray
            Array with shape (T, H, W, C), (H, W, C), or (H, W).

        Raises
        ------
        ValueError
            If the input is not a 2D, 3D or 4D array.
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
        Fetch accumulated frames as a single array.

        Returns
        -------
        ndarray or None
            Frames concatenated along the time axis with shape
            (T, H, W, C), or None if no frames have been written.
        """
        if not self._vid:
            return None
        return np.concatenate(self._vid, axis=0)

    def close(self):
        """Close the writer (no-op; frames remain available in memory)."""
        pass

    def __repr__(self):
        n_frames = sum(f.shape[0] for f in self._vid) if self._vid else 0
        return f"ArrayWriter(frames={n_frames}, shape=({self.height},{self.width},{self.n_channels}))"
