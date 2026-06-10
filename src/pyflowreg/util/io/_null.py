"""
Null video writer that discards frames without storage.
Implements the Null Object Pattern for the VideoWriter interface.
"""

import numpy as np
from ._base import VideoWriter


class NullVideoWriter(VideoWriter):
    """
    Writer that discards all frames without storing or writing them.

    Useful for running the motion correction pipeline when only intermediate
    computations (callbacks, displacement fields) are needed, without the
    overhead of actual I/O operations.

    This implements the Null Object Pattern, allowing the pipeline to run
    normally without special case handling for "no output" scenarios.

    Attributes
    ----------
    frames_written : int
        Counter tracking total frames processed.
    batches_written : int
        Counter tracking total batches processed.

    Examples
    --------
    >>> import numpy as np
    >>> from pyflowreg.util.io import NullVideoWriter
    >>> writer = NullVideoWriter()
    >>> frames = np.random.rand(10, 256, 256, 2)  # 10 frames
    >>> writer.write_frames(frames)
    >>> print(writer)
    NullVideoWriter(frames_written=10, batches=1)
    """

    def __init__(self):
        """Initialize the null writer with counters."""
        super().__init__()
        self.frames_written = 0
        self.batches_written = 0

    def init(self, first_frame_batch: np.ndarray):
        """
        Initialize writer properties from the first batch.

        Parameters
        ----------
        first_frame_batch : ndarray
            First batch with shape (T, H, W, C), (H, W, C), or (H, W).

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
            self.n_channels = shape[2]
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
        Discard frames while tracking counts for debugging/monitoring.

        Parameters
        ----------
        frames : ndarray
            Array with shape (T, H, W, C), (H, W, C), or (H, W). The
            frames are not stored, only counted.

        Raises
        ------
        ValueError
            If the input is not a 2D, 3D or 4D array.
        """
        if not self.initialized:
            self.init(frames)

        # Count frames for debugging/logging
        if frames.ndim == 2:
            # Single channel single frame (H,W)
            self.frames_written += 1
        elif frames.ndim == 3:
            # Single frame (H,W,C)
            self.frames_written += 1
        elif frames.ndim == 4:
            # Batch (T,H,W,C)
            self.frames_written += frames.shape[0]
        else:
            raise ValueError(f"Expected 2D, 3D or 4D array, got {frames.ndim}D")

        self.batches_written += 1

    def close(self):
        """Close the writer (no-op; there are no resources to clean up)."""
        pass

    def __repr__(self):
        """Return a string representation showing processing statistics."""
        return (
            f"NullVideoWriter(frames_written={self.frames_written}, "
            f"batches={self.batches_written})"
        )
