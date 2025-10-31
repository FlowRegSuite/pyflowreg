"""
FlowRegLive - Real-time motion compensation with adaptive reference.

Implements fast online registration with robust preregistered reference
and temporal filtering using half kernels for speed.
"""

from collections import deque
from typing import Optional, Tuple
import numpy as np

from pyflowreg.motion_correction.OF_options import OFOptions, QualitySetting
from pyflowreg.motion_correction.compensate_arr import compensate_arr
from pyflowreg.core.warping import imregister_wrapper
from pyflowreg.util.image_processing import (
    normalize,
    apply_gaussian_filter,
    gaussian_filter_1d_half_kernel,
)


class FlowRegLive:
    """
    Real-time motion compensation with adaptive reference updating.

    Optimized for speed using:
    - Fast quality setting for optical flow
    - Half kernel temporal filtering
    - Circular buffer for temporal history
    - Weighted reference updates every 20 frames
    """

    def __init__(
        self,
        options: Optional[OFOptions] = None,
        reference_buffer_size: int = 50,
        reference_update_interval: int = 20,
        reference_update_weight: float = 0.2,
        truncate: float = 4.0,
        **kwargs,
    ):
        """
        Initialize FlowRegLive for real-time motion compensation.

        Args:
            options: OFOptions for configuration. If None, uses fast defaults.
            reference_buffer_size: Size of buffer for reference initialization
            reference_update_interval: Update reference every N frames
            reference_update_weight: Weight for mixing new frames into reference
            truncate: Truncate filter at this many standard deviations
            **kwargs: Additional parameters to override OFOptions
        """
        # Create options with fast defaults for real-time
        if options is None:
            options = OFOptions(
                quality_setting=QualitySetting.FAST,
                **kwargs,
            )
        else:
            # Copy and override for speed
            options = options.model_copy(
                update={
                    "quality_setting": QualitySetting.FAST,
                    **kwargs,
                }
            )

        self.options = options
        self.truncate = truncate

        # Resolve displacement function from options
        self._get_disp = self.options.resolve_get_displacement()

        # Reference buffer
        self.reference_buffer = deque(maxlen=reference_buffer_size)

        # Get sigma values and determine temporal buffer size
        self.sigma = np.asarray(self.options.sigma)
        if self.sigma.ndim == 1:
            self.sigma_2d = self.sigma[:2]  # [sy, sx]
            self.sigma_t = self.sigma[2] if len(self.sigma) > 2 else 0.0
        else:
            # Per-channel, find max temporal sigma
            self.sigma_2d = self.sigma[0, :2]  # Use first channel for 2D
            # Get max temporal sigma across all channels
            self.sigma_t = np.max(self.sigma[:, 2]) if self.sigma.shape[1] > 2 else 0.0

        # Calculate temporal buffer size from sigma_t
        # Buffer size = kernel radius + 1, where radius = truncate * sigma_t
        temporal_buffer_size = max(1, int(self.truncate * self.sigma_t + 0.5) + 1)
        self.temporal_buffer = deque(maxlen=temporal_buffer_size)

        # Reference management
        self.reference_raw = (
            None  # Original reference (for warping unfiltered frames if needed)
        )
        self.reference_proc = (
            None  # Filtered reference (main reference for OF and normalization)
        )
        self.reference_update_interval = reference_update_interval
        self.reference_update_weight = reference_update_weight
        self.frame_count = 0

        # Flow initialization
        self.last_flow = None

        # Normalization parameters from filtered reference
        self.norm_min = None
        self.norm_max = None

    def set_reference(self, frames: Optional[np.ndarray] = None):
        """
        Initialize reference from frames or buffer.

        Args:
            frames: Optional array of frames (T,H,W,C). If None, uses buffer.
        """
        if frames is not None:
            # Use provided frames
            if frames.ndim == 2:
                # Single 2D frame (H,W) - convert to 3D
                self.reference_raw = frames[..., np.newaxis].copy()
            elif frames.ndim == 3:
                # Single frame (H,W,C)
                self.reference_raw = frames.copy()
            else:
                # Multiple frames - compensate and average
                print(f"Preregistering {frames.shape[0]} frames for reference...")

                # Use balanced quality for reference
                ref_options = self.options.model_copy(
                    update={"quality_setting": QualitySetting.BALANCED}
                )

                # Take mean as initial reference
                initial_ref = np.mean(frames, axis=0)

                # Compensate frames
                registered, _ = compensate_arr(frames, initial_ref, ref_options)

                # Average compensated frames as reference
                self.reference_raw = np.mean(registered, axis=0)
                print("Reference initialized from compensated frames")
        else:
            # Use buffer
            if len(self.reference_buffer) == 0:
                raise ValueError(
                    "Reference buffer is empty, cannot initialize reference"
                )

            frames = np.array(list(self.reference_buffer))
            self.set_reference(frames)
            return

        # Ensure 3D reference (H,W,C)
        if self.reference_raw.ndim == 2:
            self.reference_raw = self.reference_raw[..., np.newaxis]

        # Preprocess reference (normalize and filter)
        self._preprocess_reference()

        # Store normalization parameters from FILTERED reference
        if self.options.channel_normalization.value == "separate":
            self.norm_min = []
            self.norm_max = []
            for c in range(self.reference_proc.shape[2]):
                self.norm_min.append(self.reference_proc[..., c].min())
                self.norm_max.append(self.reference_proc[..., c].max())
        else:
            self.norm_min = self.reference_proc.min()
            self.norm_max = self.reference_proc.max()

        # Reset flow initialization
        self.last_flow = None

        # Clear temporal buffer
        self.temporal_buffer.clear()

    def _preprocess_reference(self):
        """Preprocess reference frame (normalize and filter)."""
        # First normalize
        ref_norm = normalize(
            self.reference_raw,
            channel_normalization=self.options.channel_normalization.value,
        )

        # Apply 2D Gaussian filter
        self.reference_proc = apply_gaussian_filter(
            ref_norm, self.sigma_2d, mode="reflect", truncate=self.truncate
        )

    def _normalize_with_reference(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame using filtered reference min/max values."""
        if self.norm_min is None or self.norm_max is None:
            # Fallback to frame's own normalization
            return normalize(
                frame, channel_normalization=self.options.channel_normalization.value
            )

        eps = 1e-8
        if self.options.channel_normalization.value == "separate":
            result = np.zeros_like(frame, dtype=np.float64)
            for c in range(frame.shape[-1]):
                if c < len(self.norm_min):
                    result[..., c] = (frame[..., c] - self.norm_min[c]) / (
                        self.norm_max[c] - self.norm_min[c] + eps
                    )
                else:
                    # Fallback for extra channels
                    ch = frame[..., c]
                    result[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + eps)
            return result
        else:
            return (frame - self.norm_min) / (self.norm_max - self.norm_min + eps)

    def __call__(
        self, frame: np.ndarray, normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process single frame with motion compensation.

        Args:
            frame: Input frame (H,W,C) or (H,W)
            normalize: Whether to normalize (unused, always normalizes)

        Returns:
            Tuple of (registered_frame, flow_field) with shapes (H,W,C) and (H,W,2)
        """
        if self.reference_raw is None:
            # Add to buffer and return original
            self.reference_buffer.append(frame.copy())
            return frame, np.zeros(frame.shape[:2] + (2,), dtype=np.float32)

        # Ensure 3D frame
        if frame.ndim == 2:
            frame = frame[..., np.newaxis]

        self.frame_count += 1

        # Normalize using filtered reference parameters
        frame_norm = self._normalize_with_reference(frame)

        # Apply 2D Gaussian filter
        frame_2d_filtered = apply_gaussian_filter(
            frame_norm, self.sigma_2d, mode="reflect", truncate=self.truncate
        )

        # Add to temporal buffer
        self.temporal_buffer.append(frame_2d_filtered)

        # Apply temporal filtering with half kernel
        if self.sigma_t > 0 and len(self.temporal_buffer) > 1:
            frame_proc = gaussian_filter_1d_half_kernel(
                self.temporal_buffer, self.sigma_t, truncate=self.truncate
            )
        else:
            frame_proc = frame_2d_filtered

        # Compute optical flow with initialization
        flow_params = self.options.to_dict()
        # Ensure weight is numpy array
        if "weight" in flow_params and isinstance(flow_params["weight"], list):
            flow_params["weight"] = np.array(flow_params["weight"])
        flow = self._get_disp(
            self.reference_proc,
            frame_proc,
            uv=self.last_flow,  # Use previous flow as initialization
            **flow_params,
        )

        # Store flow for next frame
        self.last_flow = flow.copy()

        # Apply warping to original frame
        registered = imregister_wrapper(
            frame,
            flow[..., 0],
            flow[..., 1],
            self.reference_raw,
            interpolation_method=self.options.interpolation_method.value,
        )

        # Ensure registered maintains 3D shape
        if registered.ndim == 2:
            registered = registered[..., np.newaxis]

        # Update reference every N frames
        if self.frame_count % self.reference_update_interval == 0:
            # Warp the processed frame and mix into processed reference
            registered_proc = imregister_wrapper(
                frame_proc,
                flow[..., 0],
                flow[..., 1],
                self.reference_proc,
                interpolation_method=self.options.interpolation_method.value,
            )

            # Ensure 3D shape for consistency
            if registered_proc.ndim == 2:
                registered_proc = registered_proc[..., np.newaxis]

            # Mix into processed reference
            self.reference_proc = (
                1 - self.reference_update_weight
            ) * self.reference_proc + self.reference_update_weight * registered_proc

            # Also update raw reference for consistency
            self.reference_raw = (
                1 - self.reference_update_weight
            ) * self.reference_raw + self.reference_update_weight * registered

            # Update normalization parameters from new filtered reference
            if self.options.channel_normalization.value == "separate":
                self.norm_min = []
                self.norm_max = []
                for c in range(self.reference_proc.shape[2]):
                    self.norm_min.append(self.reference_proc[..., c].min())
                    self.norm_max.append(self.reference_proc[..., c].max())
            else:
                self.norm_min = self.reference_proc.min()
                self.norm_max = self.reference_proc.max()

        return registered, flow

    def register_frames(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register multiple frames in sequence.

        Args:
            frames: Array of frames (T,H,W,C)

        Returns:
            Tuple of (registered_frames, flow_fields)
        """
        registered = np.empty_like(frames)
        flows = np.empty(frames.shape[:3] + (2,), dtype=np.float32)

        for i in range(frames.shape[0]):
            registered[i], flows[i] = self(frames[i], False)

        return registered, flows

    def reset_reference(self, new_reference: np.ndarray):
        """
        Set new reference and reset flow.

        Args:
            new_reference: New reference frame (H,W,C)
        """
        self.set_reference(new_reference)

    def get_current_flow(self) -> Optional[np.ndarray]:
        """Get current flow field state."""
        return self.last_flow.copy() if self.last_flow is not None else None

    def set_flow_init(self, w_init: np.ndarray):
        """Set flow field initialization for next frame."""
        self.last_flow = w_init.copy()
