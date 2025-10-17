"""
Threading executor - processes frames in parallel using threads.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Tuple, Optional
import numpy as np
from .base import BaseExecutor


class ThreadingExecutor(BaseExecutor):
    """
    Threading executor that processes frames in parallel using threads.

    Good for I/O-bound operations or when the GIL is released (e.g., NumPy operations).
    Less efficient than multiprocessing for pure Python CPU-bound operations.
    """

    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize threading executor.

        Args:
            n_workers: Number of worker threads. If None, uses RuntimeContext default.
        """
        super().__init__(n_workers)
        self.executor = None

    def setup(self):
        """Create the thread pool executor."""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.n_workers)

    def cleanup(self):
        """Shutdown the thread pool executor."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def _process_frame(
        self,
        t: int,
        frame: np.ndarray,
        frame_proc: np.ndarray,
        reference_raw: np.ndarray,
        reference_proc: np.ndarray,
        w_init: np.ndarray,
        get_displacement_func: Callable,
        imregister_func: Callable,
        interpolation_method: str,
        flow_params: dict,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Process a single frame.

        Args:
            t: Frame index
            frame: Raw frame to register
            frame_proc: Preprocessed frame
            reference_raw: Raw reference frame
            reference_proc: Preprocessed reference frame
            w_init: Initial flow field
            get_displacement_func: Function to compute optical flow
            imregister_func: Function to apply flow field
            interpolation_method: Interpolation method
            flow_params: Dictionary of flow computation parameters

        Returns:
            Tuple of (frame_index, registered_frame, flow_field)
        """
        # Import prealignment functions if needed
        from pyflowreg.util.xcorr_prealignment import estimate_rigid_xcorr_2d

        # Extract CC parameters and remove them from flow_params
        use_cc = bool(flow_params.get("cc_initialization", False))
        cc_hw = flow_params.get("cc_hw", 256)
        cc_up = int(flow_params.get("cc_up", 1))

        # Create flow_params without CC parameters
        flow_params_clean = {
            k: v
            for k, v in flow_params.items()
            if k not in ["cc_initialization", "cc_hw", "cc_up"]
        }

        if use_cc:
            target_hw = cc_hw
            if isinstance(target_hw, int):
                target_hw = (target_hw, target_hw)
            up = cc_up
            weight = flow_params_clean.get("weight", None)

            # Step 1: Backward warp mov by w_init to get partially aligned
            mov_partial = imregister_func(
                frame_proc,
                w_init[..., 0],  # dx
                w_init[..., 1],  # dy
                reference_proc,
                interpolation_method="linear",
            )

            # Use 2D views for CC
            ref_for_cc = self._as2d(reference_proc)
            mov_for_cc = self._as2d(mov_partial)

            # Step 2: Estimate rigid residual between ref and partially aligned mov
            w_cross = estimate_rigid_xcorr_2d(
                ref_for_cc, mov_for_cc, target_hw=target_hw, up=up, weight=weight
            )

            # Step 3: Combine w_init + w_cross
            w_combined = w_init.copy()
            w_combined[..., 0] += w_cross[0]
            w_combined[..., 1] += w_cross[1]

            # Step 4: Backward warp original mov by combined field
            mov_aligned = imregister_func(
                frame_proc,
                w_combined[..., 0],
                w_combined[..., 1],
                reference_proc,
                interpolation_method="linear",
            )

            # Ensure mov_aligned has channel dimension (imregister_wrapper strips it for single channel)
            mov_aligned = self._as3d(mov_aligned)

            # Step 5: Get residual non-rigid displacement
            w_residual = get_displacement_func(
                reference_proc,
                mov_aligned,
                uv=np.zeros_like(w_init),
                **flow_params_clean,
            )

            # Step 6: Total flow is w_init + w_cross + w_residual
            flow = (w_combined + w_residual).astype(np.float32, copy=False)
        else:
            # Compute optical flow without prealignment
            flow = get_displacement_func(
                reference_proc, frame_proc, uv=w_init.copy(), **flow_params
            ).astype(np.float32, copy=False)

        # Apply flow field to register the frame
        reg_frame = imregister_func(
            frame,
            flow[..., 0],
            flow[..., 1],
            reference_raw,
            interpolation_method=interpolation_method,
        )

        return t, reg_frame, flow

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
        Process frames in parallel using threads.

        Args:
            batch: Raw frames to register, shape (T, H, W, C)
            batch_proc: Preprocessed frames for flow computation, shape (T, H, W, C)
            reference_raw: Raw reference frame, shape (H, W, C)
            reference_proc: Preprocessed reference frame, shape (H, W, C)
            w_init: Initial flow field, shape (H, W, 2)
            get_displacement_func: Function to compute optical flow
            imregister_func: Function to apply flow field for registration
            interpolation_method: Interpolation method for registration
            **kwargs: Additional parameters including 'flow_params' dict

        Returns:
            Tuple of (registered_frames, flow_fields)
        """
        # Normalize inputs to ensure consistent dimensions
        batch, batch_proc, reference_raw, reference_proc, w_init = (
            self._normalize_inputs(
                batch, batch_proc, reference_raw, reference_proc, w_init
            )
        )

        T, H, W, C = batch.shape

        # Get flow parameters from kwargs
        flow_params = kwargs.get("flow_params", {})

        # Initialize output arrays (use empty instead of zeros for performance)
        registered = np.empty_like(batch)
        flow_fields = np.empty((T, H, W, 2), dtype=np.float32)

        # Ensure executor is created
        if self.executor is None:
            self.setup()

        # Submit all frames for processing
        futures = []
        for t in range(T):
            future = self.executor.submit(
                self._process_frame,
                t,
                batch[t],
                batch_proc[t],
                reference_raw,
                reference_proc,
                w_init,
                get_displacement_func,
                imregister_func,
                interpolation_method,
                flow_params,
            )
            futures.append(future)

        # Collect results as they complete
        for future in as_completed(futures):
            t, reg_frame, flow = future.result()

            # Store results
            flow_fields[t] = flow

            # Handle case where registered frame might have fewer channels
            if reg_frame.ndim < registered.ndim - 1:
                registered[t, ..., 0] = reg_frame
            else:
                registered[t] = reg_frame

            # Call progress callback for this frame
            if progress_callback is not None:
                progress_callback(1)

        return registered, flow_fields

    def get_info(self) -> dict:
        """Get information about this executor."""
        info = super().get_info()
        info.update(
            {
                "parallel": True,
                "description": f"Threaded parallel processing with {self.n_workers} workers",
            }
        )
        return info


# Register this executor with RuntimeContext on import
ThreadingExecutor.register()
