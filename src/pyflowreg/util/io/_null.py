"""
Null video writer that discards frames without storage.
Implements the Null Object Pattern for the VideoWriter interface.
"""

import numpy as np
from ._base import VideoWriter


class NullVideoWriter(VideoWriter):
    """
    A writer that discards all frames without storing or writing them.

    Useful for running the motion correction pipeline when only intermediate
    computations (callbacks, displacement fields) are needed, without the
    overhead of actual I/O operations.

    This implements the Null Object Pattern, allowing the pipeline to run
    normally without special case handling for "no output" scenarios.

    Attributes:
        frames_written: Counter tracking total frames processed
        batches_written: Counter tracking total batches processed

    Example:
        >>> from pyflowreg.util.io import NullVideoWriter
        >>> writer = NullVideoWriter()
        >>> frames = np.random.rand(10, 256, 256, 2)  # 10 frames
        >>> writer.write_frames(frames)
        >>> print(writer)  # NullVideoWriter(frames_written=10, batches=1)
    """

    def __init__(self):
        """Initialize the null writer with counters."""
        super().__init__()
        self.frames_written = 0
        self.batches_written = 0

    def init(self, first_frame_batch: np.ndarray):
        """
        Initialize writer properties from first batch.

        Args:
            first_frame_batch: First batch with shape (T,H,W,C), (H,W,C), or (H,W)
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
        Discard frames but track count for debugging/monitoring.

        Args:
            frames: Array with shape (T,H,W,C), (H,W,C), or (H,W)
                    These frames are not stored, only counted.
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
        """No-op for null writer - no resources to clean up."""
        pass

    def __repr__(self):
        """String representation showing processing statistics."""
        return (
            f"NullVideoWriter(frames_written={self.frames_written}, "
            f"batches={self.batches_written})"
        )
