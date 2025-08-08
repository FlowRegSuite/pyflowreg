from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class VideoReader(ABC):
    """
    Abstract base class for all video file readers.
    Defines the common interface for reading frames and batches from a video source.
    """
    def __init__(self):
        self.height: int = 0
        self.width: int = 0
        self.frame_count: int = 0
        self.bit_depth: int = 0
        self.n_channels: int = 0
        self.dtype: Optional[np.dtype] = None
        self.current_frame: int = 0

    @abstractmethod
    def read_batch(self) -> Optional[np.ndarray]:
        """Reads the next batch of frames from the file."""
        pass

    @abstractmethod
    def read_frames(self, frame_indices: list[int]) -> Optional[np.ndarray]:
        """Reads specific frames from the file by their indices."""
        pass

    @abstractmethod
    def has_batch(self) -> bool:
        """Returns True if there are more frames to read."""
        pass

    @abstractmethod
    def reset(self):
        """Resets the reader to the beginning of the file."""
        pass

    @abstractmethod
    def close(self):
        """Closes the file and releases any resources."""
        pass

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
        self.n_channels = 0
        self.bit_depth = 0

    @abstractmethod
    def write_frames(self, frames: np.ndarray):
        """Writes frames to the file."""
        pass

    @abstractmethod
    def close(self):
        """Closes the writer and finalizes the file."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()