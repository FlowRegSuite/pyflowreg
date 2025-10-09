"""
Sequential executor - processes frames one by one without parallelization.
"""

from typing import Callable, Tuple
import numpy as np
from .base import BaseExecutor


class SequentialExecutor(BaseExecutor):
    """
    Sequential executor that processes frames one at a time.

    This is the simplest executor and serves as a reference implementation.
    It's also the most memory-efficient as it only processes one frame at a time.
    """

    def __init__(self, n_workers: int = 1):
        """
        Initialize sequential executor.

        Args:
            n_workers: Ignored for sequential executor, always uses 1.
        """
        super().__init__(n_workers=1)

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
        progress_callback: Callable = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process frames sequentially.

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
        flow_params_all = kwargs.get("flow_params", {})

        # Extract CC parameters and remove them from flow_params
        use_cc = bool(flow_params_all.get("cc_initialization", False))
        cc_hw = flow_params_all.get("cc_hw", 256)
        cc_up = int(flow_params_all.get("cc_up", 1))

        # Create flow_params without CC parameters
        flow_params = {
            k: v
            for k, v in flow_params_all.items()
            if k not in ["cc_initialization", "cc_hw", "cc_up"]
        }

        # Initialize output arrays (use empty instead of zeros for performance)
        registered = np.empty_like(batch)
        flow_fields = np.empty((T, H, W, 2), dtype=np.float32)

        # Import prealignment functions if needed
        if use_cc:
            from pyflowreg.util.xcorr_prealignment import estimate_rigid_xcorr_2d

        # Setup CC parameters
        target_hw = cc_hw
        if isinstance(target_hw, int):
            target_hw = (target_hw, target_hw)
        up = cc_up
        weight = flow_params.get("weight", None)

        # Process each frame sequentially
        for t in range(T):
            if use_cc:
                # Step 1: Backward warp mov by w_init to get partially aligned
                mov_partial = imregister_func(
                    batch_proc[t],
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
                    batch_proc[t],
                    w_combined[..., 0],
                    w_combined[..., 1],
                    reference_proc,
                    interpolation_method="linear",
                )

                # Ensure mov_aligned has channel dimension (imregister_wrapper strips it for single channel)
                mov_aligned = self._as3d(mov_aligned)

                # Step 5: Get residual non-rigid displacement
                w_residual = get_displacement_func(
                    reference_proc, mov_aligned, uv=np.zeros_like(w_init), **flow_params
                )

                # Step 6: Total flow is w_init + w_cross + w_residual
                flow = (w_combined + w_residual).astype(np.float32, copy=False)
            else:
                # Compute optical flow for this frame without prealignment
                flow = get_displacement_func(
                    reference_proc, batch_proc[t], uv=w_init.copy(), **flow_params
                ).astype(np.float32, copy=False)

            # Apply flow field to register the frame
            reg_frame = imregister_func(
                batch[t],
                flow[..., 0],
                flow[..., 1],
                reference_raw,
                interpolation_method=interpolation_method,
            )

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
            {"parallel": False, "description": "Sequential frame-by-frame processing"}
        )
        return info


# Register this executor with RuntimeContext on import
SequentialExecutor.register()
