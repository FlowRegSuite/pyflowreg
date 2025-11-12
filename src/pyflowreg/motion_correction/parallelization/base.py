"""
Base executor abstract class for parallelization strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
from pyflowreg._runtime import RuntimeContext


class BaseExecutor(ABC):
    """
    Abstract base class for parallelization executors.

    All executors must implement the process_batch method which takes:
    - Batch of frames to process
    - Preprocessed batch
    - Reference frames (raw and preprocessed)
    - Initial flow field
    - Options and parameters

    And returns:
    - Registered frames
    - Computed flow fields
    """

    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize the executor.

        Args:
            n_workers: Number of workers to use. If None, uses RuntimeContext default.
        """
        self.n_workers = n_workers or RuntimeContext.get("max_workers", 1)
        self.name = self.__class__.__name__.replace("Executor", "").lower()

    @staticmethod
    def _as2d(x):
        """Convert single-channel 3D array to 2D."""
        return x[..., 0] if x.ndim == 3 and x.shape[2] == 1 else x

    @staticmethod
    def _as3d(x):
        """Ensure array has channel dimension (H,W,C)."""
        return x[..., None] if x.ndim == 2 else x

    @staticmethod
    def _as4d(x):
        """Ensure array has time and channel dimensions (T,H,W,C)."""
        return x[..., None] if x.ndim == 3 else x

    def _normalize_inputs(
        self, batch, batch_proc, reference_raw, reference_proc, w_init
    ):
        """
        Normalize input dimensions to ensure consistency across executors.

        Ensures:
        - batch and batch_proc are 4D (T,H,W,C)
        - reference_raw and reference_proc are 3D (H,W,C)
        - All arrays are contiguous float32
        """
        batch = np.ascontiguousarray(self._as4d(batch))
        batch_proc = np.ascontiguousarray(
            self._as4d(batch_proc).astype(np.float32, copy=False)
        )
        reference_raw = np.ascontiguousarray(
            self._as3d(reference_raw).astype(np.float32, copy=False)
        )
        reference_proc = np.ascontiguousarray(
            self._as3d(reference_proc).astype(np.float32, copy=False)
        )
        w_init = np.ascontiguousarray(w_init.astype(np.float32, copy=False))
        return batch, batch_proc, reference_raw, reference_proc, w_init

    @abstractmethod
    def process_batch(
        self,
        batch: np.ndarray,
        batch_proc: np.ndarray,
        reference_raw: np.ndarray,
        reference_proc: np.ndarray,
        w_init: np.ndarray,
        get_displacement_func: Callable,
        imregister_func: Callable,
        interpolation_method: str = "cubic",
        progress_callback: Optional[Callable[[int], None]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of frames for motion correction.

        Args:
            batch: Raw frames to register, shape (T, H, W, C)
            batch_proc: Preprocessed frames for flow computation, shape (T, H, W, C)
            reference_raw: Raw reference frame, shape (H, W, C)
            reference_proc: Preprocessed reference frame, shape (H, W, C)
            w_init: Initial flow field, shape (H, W, 2)
            get_displacement_func: Function to compute optical flow
            imregister_func: Function to apply flow field for registration
            interpolation_method: Interpolation method for registration
            progress_callback: Optional callback for per-frame progress (frames_completed)
            **kwargs: Additional parameters

        Returns:
            Tuple of (registered_frames, flow_fields) where:
                registered_frames: shape (T, H, W, C)
                flow_fields: shape (T, H, W, 2)
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False

    def setup(self):
        """
        Setup method called before processing.
        Override in subclasses if needed.
        """
        pass

    def cleanup(self):
        """
        Cleanup method called after processing.
        Override in subclasses if needed.
        """
        pass

    @classmethod
    def register(cls):
        """Register this executor with the RuntimeContext."""
        instance_name = cls.__name__.replace("Executor", "").lower()
        RuntimeContext.register_parallelization_executor(instance_name, cls)

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this executor.

        Returns:
            Dictionary with executor information
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "n_workers": self.n_workers,
        }
