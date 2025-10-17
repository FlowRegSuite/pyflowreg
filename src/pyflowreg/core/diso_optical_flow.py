"""
Dense Inverse Search Optical Flow (DIS) backend using OpenCV.
Provides an alternative optical flow computation method to the variational approach.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any


class DisoOF:
    """
    Dense Inverse Search Optical Flow (DIS) implementation using OpenCV.

    This class provides a callable interface compatible with get_displacement
    for computing optical flow between fixed and moving images.

    Uses lazy initialization to ensure pickle compatibility for multiprocessing.

    Parameters
    ----------
    preset : int
        DIS preset mode:
        - cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST: Ultrafast preset
        - cv2.DISOPTICAL_FLOW_PRESET_FAST: Fast preset
        - cv2.DISOPTICAL_FLOW_PRESET_MEDIUM: Medium preset (default)
    finest_scale : int
        Finest scale for the image pyramid (0 = original scale)
    gradient_descent_iterations : int
        Number of gradient descent iterations at each pyramid level
    patch_size : int
        Size of the patch for matching (default: 8)
    patch_stride : int
        Stride between neighbor patches (default: 4)
    use_mean_normalization : bool
        Whether to use mean normalization (default: True)
    use_spatial_propagation : bool
        Whether to use spatial propagation (default: True)
    """

    def __init__(
        self,
        preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        finest_scale: int = 2,
        gradient_descent_iterations: int = 12,
        patch_size: int = 8,
        patch_stride: int = 4,
        use_mean_normalization: bool = True,
        use_spatial_propagation: bool = True,
    ):
        self._cfg = dict(
            preset=preset,
            finest_scale=finest_scale,
            gradient_descent_iterations=gradient_descent_iterations,
            patch_size=patch_size,
            patch_stride=patch_stride,
            use_mean_normalization=use_mean_normalization,
            use_spatial_propagation=use_spatial_propagation,
        )
        self._dis = None

    def __getstate__(self):
        """Support pickling for multiprocessing."""
        return {"_cfg": self._cfg, "_dis": None}

    def __setstate__(self, state):
        """Support unpickling for multiprocessing."""
        self._cfg = state["_cfg"]
        self._dis = None

    def _ensure(self):
        """Lazy initialization of OpenCV DIS object."""
        if self._dis is not None:
            return

        d = cv2.DISOpticalFlow_create(self._cfg["preset"])
        d.setFinestScale(self._cfg["finest_scale"])
        d.setGradientDescentIterations(self._cfg["gradient_descent_iterations"])
        d.setPatchSize(self._cfg["patch_size"])
        d.setPatchStride(self._cfg["patch_stride"])
        d.setUseMeanNormalization(bool(self._cfg["use_mean_normalization"]))
        d.setUseSpatialPropagation(bool(self._cfg["use_spatial_propagation"]))
        self._dis = d

    def _to_gray(
        self, img: np.ndarray, weight: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert image to grayscale using weights if provided.

        Parameters
        ----------
        img : np.ndarray
            Input image of shape (H, W) or (H, W, C)
        weight : np.ndarray, optional
            Channel weights - can be:
            - 1D array of channel weights
            - 2D array (H, W) for spatial weights
            - 3D array (H, W, C) for full spatial-channel weights

        Returns
        -------
        np.ndarray
            Grayscale image of shape (H, W)
        """
        if img.ndim == 2:
            return img

        if img.ndim == 3:
            if img.shape[2] == 1:
                return img[:, :, 0]

            # Handle different weight formats
            if weight is not None:
                if weight.ndim == 1:
                    # 1D channel weights - normalize and broadcast
                    if len(weight) != img.shape[2]:
                        # Use equal weights if mismatch
                        weight = np.ones(img.shape[2]) / img.shape[2]
                    else:
                        weight = weight / weight.sum()
                    # Broadcast to spatial dimensions
                    weight = np.ones(
                        (img.shape[0], img.shape[1], img.shape[2])
                    ) * weight.reshape(1, 1, -1)
                elif weight.ndim == 2:
                    # 2D spatial weights - broadcast to all channels
                    weight = weight[:, :, np.newaxis]
                    # Apply spatial weights equally to all channels, then average
                    return np.mean(img * weight, axis=2)
                elif weight.ndim == 3:
                    # Full 3D weights - use as is
                    if weight.shape != img.shape:
                        # Fallback to equal weights if shape mismatch
                        weight = np.ones(img.shape[2]) / img.shape[2]
                        weight = np.ones(
                            (img.shape[0], img.shape[1], img.shape[2])
                        ) * weight.reshape(1, 1, -1)
                else:
                    # Fallback to equal weights
                    weight = np.ones(img.shape[2]) / img.shape[2]
                    weight = np.ones(
                        (img.shape[0], img.shape[1], img.shape[2])
                    ) * weight.reshape(1, 1, -1)
            else:
                # Equal weights for all channels
                weight = np.ones(img.shape[2]) / img.shape[2]
                weight = np.ones(
                    (img.shape[0], img.shape[1], img.shape[2])
                ) * weight.reshape(1, 1, -1)

            # Weighted average
            return np.sum(img * weight, axis=2)

        raise ValueError(f"Unexpected image shape: {img.shape}")

    def _normalize(self, a: np.ndarray, b: np.ndarray) -> tuple:
        """
        Convert images to uint8 [0,255] range for OpenCV.
        Handles both [0,1] float and [0,255] uint8 inputs.

        Parameters
        ----------
        a : np.ndarray
            First image
        b : np.ndarray
            Second image

        Returns
        -------
        tuple
            Both images converted to uint8 [0,255] range
        """
        # Check if already uint8
        if a.dtype == np.uint8 and b.dtype == np.uint8:
            return a, b

        # Handle uint8 input (convert to float for consistent processing)
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        if b.dtype == np.uint8:
            b = b.astype(np.float32) / 255.0

        # Now assume [0,1] range and convert to [0,255]
        # Clip to [0,1] range (in case of slight overflow from preprocessing)
        a_clipped = np.clip(a, 0, 1)
        b_clipped = np.clip(b, 0, 1)

        # Convert to [0,255] uint8
        A = (a_clipped * 255).astype(np.uint8)
        B = (b_clipped * 255).astype(np.uint8)

        return A, B

    def __call__(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        w: Optional[np.ndarray] = None,
        weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute optical flow between fixed and moving images.

        Parameters
        ----------
        fixed : np.ndarray
            Reference/fixed image of shape (H, W) or (H, W, C)
            Expected to be normalized to [0,1] range or uint8 [0,255]
        moving : np.ndarray
            Moving image of shape (H, W) or (H, W, C)
            Expected to be normalized to [0,1] range or uint8 [0,255]
        w : np.ndarray, optional
            Initial flow field of shape (H, W, 2) for warm start
        weight : np.ndarray, optional
            Channel weights for multi-channel images
        **kwargs : dict
            Additional parameters (for compatibility, not used)

        Returns
        -------
        np.ndarray
            Displacement field of shape (H, W, 2) as float32
        """
        self._ensure()

        # Convert to grayscale using weights
        a = self._to_gray(fixed, weight)
        b = self._to_gray(moving, weight)

        # Normalize to [0,255] uint8
        A, B = self._normalize(a, b)

        # Prepare initial flow if provided
        init = None
        if (
            w is not None
            and isinstance(w, np.ndarray)
            and w.ndim == 3
            and w.shape[2] == 2
        ):
            init = w.astype(np.float32, copy=False)

        # Compute optical flow
        flow = self._dis.calc(A, B, init)

        # Return as float32
        return flow.astype(np.float32, copy=False)

    def set_preset(self, preset: int):
        """
        Update the DIS preset configuration.

        Parameters
        ----------
        preset : int
            One of cv2.DISOPTICAL_FLOW_PRESET_*
        """
        cfg = dict(self._cfg)
        cfg["preset"] = preset
        self._cfg = cfg
        self._dis = None

    def get_params(self) -> Dict[str, Any]:
        """
        Get current DIS parameters.

        Returns
        -------
        dict
            Dictionary containing current parameter values
        """
        return dict(self._cfg)

    def set_params(self, **params):
        """
        Update DIS parameters.

        Parameters
        ----------
        **params : dict
            Parameters to update
        """
        self._cfg.update(params)
        self._dis = None


def _diso_factory(**kwargs):
    """
    Factory function for creating DisoOF instances suitable for multiprocessing.

    Returns a DisoOF instance that is pickle-safe due to lazy initialization.

    Parameters
    ----------
    **kwargs : dict
        Parameters to pass to DisoOF constructor

    Returns
    -------
    DisoOF
        DisoOF instance with specified parameters
    """
    return DisoOF(**kwargs)
