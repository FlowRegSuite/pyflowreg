"""
Dense Inverse Search (DIS) optical flow backend wrapping OpenCV's DISOpticalFlow.

Provides :class:`DisoOF`, a callable wrapper around ``cv2.DISOpticalFlow``
that follows the same ``(fixed, moving, ...) -> (H, W, 2)`` calling
convention as the variational solver
:func:`pyflowreg.core.optical_flow.get_displacement`, so it can be selected
as an alternative, patch-based flow backend in the motion correction
pipelines.

Notes
-----
Registration and constraints as implemented:

- ``pyflowreg.core`` (package ``__init__``) registers :func:`_diso_factory`
  under the backend name ``"diso"`` only when ``import cv2`` succeeds, with
  ``supported_executors={"sequential", "threading"}``. Multiprocessing is
  excluded there because the multiprocessing workers import the flowreg
  solver directly and do not reconstruct registry backends, so ``diso``
  would silently fall back to the variational solver.
- ``OFOptions.resolve_get_displacement`` raises ``ValueError`` when
  ``flow_backend="diso"`` is combined with a constancy assumption other
  than ``"gc"`` (the default; alias ``"gradient"``), or with the graduated
  non-convexity options ``gnc_schedule`` / ``warping_steps``. The
  ``"gc"`` value itself is not used by this wrapper; it is the only setting
  that passes validation.
- ``OFOptions.backend_params`` is expanded as keyword arguments into
  :func:`_diso_factory` and therefore maps one-to-one onto the
  :class:`DisoOF` constructor parameters (``preset``, ``finest_scale``,
  ``gradient_descent_iterations``, ``patch_size``, ``patch_stride``,
  ``use_mean_normalization``, ``use_spatial_propagation``).
- Variational solver keywords forwarded by the pipelines (``alpha``,
  ``iterations``, ``const_assumption``, ``gnc_schedule``, ...) are
  accepted by :meth:`DisoOF.__call__` for signature compatibility but
  ignored. The flow initialization keyword ``uv`` is the exception: it is
  honored as the DIS warm start (``w`` takes precedence when both are
  given).

References
----------
.. [Kroeger2016] T. Kroeger, R. Timofte, D. Dai, L. Van Gool,
   "Fast Optical Flow using Dense Inverse Search", ECCV 2016.
   Algorithm implemented by OpenCV's ``cv2.DISOpticalFlow``.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any


class DisoOF:
    """
    Dense Inverse Search (DIS) optical flow wrapper around OpenCV.

    Wraps ``cv2.DISOpticalFlow`` behind a callable interface compatible with
    :func:`pyflowreg.core.optical_flow.get_displacement`: instances are
    called as ``w = diso(fixed, moving, ...)`` and return an ``(H, W, 2)``
    displacement field. The OpenCV object is created lazily on first use so
    that instances remain picklable.

    Parameters
    ----------
    preset : int, optional
        OpenCV DIS preset passed to ``cv2.DISOpticalFlow_create``. One of
        ``cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST``,
        ``cv2.DISOPTICAL_FLOW_PRESET_FAST`` or
        ``cv2.DISOPTICAL_FLOW_PRESET_MEDIUM``.
        Default is ``cv2.DISOPTICAL_FLOW_PRESET_MEDIUM``.
    finest_scale : int, optional
        Finest pyramid scale on which the flow is computed; 0 corresponds to
        the original image resolution. Applied via ``setFinestScale``.
        Default is 2.
    gradient_descent_iterations : int, optional
        Number of gradient descent iterations in the patch inverse search
        stage. Applied via ``setGradientDescentIterations``. Default is 12.
    patch_size : int, optional
        Size of an image patch for matching, in pixels. Applied via
        ``setPatchSize``. Default is 8.
    patch_stride : int, optional
        Stride between neighboring patches. Applied via ``setPatchStride``.
        Default is 4.
    use_mean_normalization : bool, optional
        Whether to use mean-normalized patches when computing patch
        distances. Applied via ``setUseMeanNormalization``. Default is True.
    use_spatial_propagation : bool, optional
        Whether to use spatial propagation of flow vectors. Applied via
        ``setUseSpatialPropagation``. Default is True.

    See Also
    --------
    pyflowreg.core.optical_flow.get_displacement : Variational solver with
        the same calling convention.
    _diso_factory : Factory registered as flow backend ``"diso"``.

    Notes
    -----
    Registered as flow backend ``"diso"`` in ``pyflowreg.core`` (only when
    OpenCV is importable) with
    ``supported_executors={"sequential", "threading"}``. The constructor
    only stores the configuration; ``__getstate__`` / ``__setstate__`` drop
    the OpenCV handle so instances can be pickled, and the handle is
    re-created lazily on the next call.

    Examples
    --------
    >>> import numpy as np
    >>> from pyflowreg.core.diso_optical_flow import DisoOF
    >>> diso = DisoOF()
    >>> fixed = np.zeros((64, 64), dtype=np.float32)
    >>> fixed[24:40, 24:40] = 1.0
    >>> moving = np.roll(fixed, 2, axis=1)
    >>> w = diso(fixed, moving)  # doctest: +SKIP
    >>> w.shape  # doctest: +SKIP
    (64, 64, 2)
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
        """
        Return picklable state with the OpenCV handle dropped.

        Returns
        -------
        dict
            State with the configuration under ``"_cfg"`` and ``"_dis"`` set
            to None so unpickled instances re-create the OpenCV object
            lazily.
        """
        return {"_cfg": self._cfg, "_dis": None}

    def __setstate__(self, state):
        """
        Restore instance state after unpickling.

        Parameters
        ----------
        state : dict
            State produced by ``__getstate__``. The OpenCV object is not
            restored; it is re-created lazily on the next call.
        """
        self._cfg = state["_cfg"]
        self._dis = None

    def _ensure(self):
        """
        Create and configure the OpenCV DIS object if not yet created.

        Instantiates ``cv2.DISOpticalFlow_create`` with the configured
        preset and applies the remaining configuration values through the
        corresponding OpenCV setters. Does nothing if the object already
        exists.
        """
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
        Reduce an image to a single channel using optional weights.

        Parameters
        ----------
        img : ndarray
            Input image of shape (H, W) or (H, W, C).
        weight : ndarray, optional
            Channel and/or spatial weights:

            - 1D array of length C: normalized to sum to 1 and used for a
              weighted sum over channels. If the length does not match C,
              equal weights ``1/C`` are used instead.
            - 2D array of shape (H, W): broadcast over channels; the result
              is the mean over channels of ``img * weight[..., None]``.
            - 3D array of shape (H, W, C): used as-is (without
              normalization) for a weighted sum over channels. If the shape
              does not match ``img``, equal weights ``1/C`` are used
              instead.

            If None, or for any other dimensionality, equal weights ``1/C``
            are used.

        Returns
        -------
        ndarray
            Single-channel image of shape (H, W). 2D inputs are returned
            unchanged; (H, W, 1) inputs are returned with the channel axis
            removed, without applying weights.

        Raises
        ------
        ValueError
            If ``img`` is neither 2D nor 3D.
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
        Convert an image pair to uint8 in [0, 255] for OpenCV.

        Parameters
        ----------
        a : ndarray
            First image (the fixed image in ``__call__``).
        b : ndarray
            Second image (the moving image in ``__call__``).

        Returns
        -------
        tuple of ndarray
            ``(A, B)`` as uint8 arrays. If both inputs are already uint8,
            they are returned unchanged. Otherwise, uint8 inputs are first
            rescaled to [0, 1] float32, both images are clipped to [0, 1],
            and the result is quantized to uint8 in [0, 255].

        Notes
        -----
        Non-uint8 inputs are assumed to be normalized to [0, 1]; values
        outside this range are clipped before quantization.
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
        Compute the displacement field between fixed and moving images.

        Both images are reduced to a single channel (see ``_to_gray``),
        converted to uint8 (see ``_normalize``), and passed to
        ``cv2.DISOpticalFlow.calc(fixed, moving, init)``.

        Parameters
        ----------
        fixed : ndarray
            Reference (fixed) image of shape (H, W) or (H, W, C), either
            float values normalized to [0, 1] or uint8 in [0, 255].
        moving : ndarray
            Moving image of shape (H, W) or (H, W, C), same value
            conventions as ``fixed``.
        w : ndarray, optional
            Initial displacement field of shape (H, W, 2) used as warm
            start. Takes precedence over ``uv`` when both are given. Used
            only if it is an ndarray with exactly this layout and the same
            spatial size as the images; it is converted to contiguous
            float32.
        weight : ndarray, optional
            Channel weights for multi-channel inputs; see ``_to_gray`` for
            the accepted formats.
        **kwargs : dict
            Accepted for signature compatibility with the variational
            :func:`pyflowreg.core.optical_flow.get_displacement`. The
            ``uv`` keyword (the flow initialization passed by the batch
            pipelines) is honored as warm start when ``w`` is not given;
            all other extra keyword arguments (e.g. ``alpha``,
            ``iterations``, ``const_assumption``, ``gnc_schedule``) are
            ignored.

        Returns
        -------
        ndarray
            Displacement field of shape (H, W, 2), float32, with channel 0
            the horizontal component ``u`` (x) and channel 1 the vertical
            component ``v`` (y). Warping ``moving`` backward by ``(u, v)``
            (as done by ``pyflowreg.core.warping.imregister_wrapper``)
            aligns it to ``fixed``.

        Notes
        -----
        The batch pipelines pass the flow initialization as the keyword
        ``uv``; it is forwarded to ``cv2.DISOpticalFlow.calc`` as the
        initial flow. A warm start whose spatial size does not match the
        images is silently skipped (mirroring OpenCV's own size check).
        """
        self._ensure()

        # Convert to grayscale using weights
        a = self._to_gray(fixed, weight)
        b = self._to_gray(moving, weight)

        # Normalize to [0,255] uint8
        A, B = self._normalize(a, b)

        # Prepare initial flow if provided. Precedence: explicit ``w`` wins,
        # otherwise fall back to the pipeline keyword ``uv`` (used by the
        # executors, FlowRegLive and compensate_recording).
        init = None
        init_src = w if w is not None else kwargs.get("uv")
        if (
            isinstance(init_src, np.ndarray)
            and init_src.ndim == 3
            and init_src.shape[2] == 2
            and init_src.shape[:2] == A.shape[:2]
        ):
            # Copy: cv2.DISOpticalFlow.calc treats the init flow as an
            # InputOutputArray and writes the result into it in place; cv2
            # also requires contiguous float32 (CV_32FC2) of image size.
            init = np.array(init_src, dtype=np.float32, order="C", copy=True)

        # Compute optical flow
        flow = self._dis.calc(A, B, init)

        # Return as float32
        return flow.astype(np.float32, copy=False)

    def set_preset(self, preset: int):
        """
        Update the DIS preset and reset the OpenCV object.

        Parameters
        ----------
        preset : int
            One of the ``cv2.DISOPTICAL_FLOW_PRESET_*`` constants.

        Notes
        -----
        The cached OpenCV object is discarded and re-created with the new
        preset on the next call.
        """
        cfg = dict(self._cfg)
        cfg["preset"] = preset
        self._cfg = cfg
        self._dis = None

    def get_params(self) -> Dict[str, Any]:
        """
        Return a copy of the current configuration.

        Returns
        -------
        dict
            Copy of the configuration dictionary with the keys ``preset``,
            ``finest_scale``, ``gradient_descent_iterations``,
            ``patch_size``, ``patch_stride``, ``use_mean_normalization``
            and ``use_spatial_propagation``.
        """
        return dict(self._cfg)

    def set_params(self, **params):
        """
        Update configuration values and reset the OpenCV object.

        Parameters
        ----------
        **params : dict
            Configuration entries merged into the current configuration.
            Recognized keys are the constructor parameters; unrecognized
            keys are stored but have no effect on the OpenCV object.

        Notes
        -----
        The cached OpenCV object is discarded and re-created with the
        updated configuration on the next call.
        """
        self._cfg.update(params)
        self._dis = None


def _diso_factory(**kwargs):
    """
    Create a DisoOF instance for use as the ``"diso"`` flow backend.

    This factory is registered under the backend name ``"diso"`` in
    ``pyflowreg.core`` (only when OpenCV is importable) with
    ``supported_executors={"sequential", "threading"}``.
    ``OFOptions.backend_params`` is expanded into ``kwargs`` when the
    backend is resolved, so its keys must match the :class:`DisoOF`
    constructor parameters.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments forwarded to :class:`DisoOF` (``preset``,
        ``finest_scale``, ``gradient_descent_iterations``, ``patch_size``,
        ``patch_stride``, ``use_mean_normalization``,
        ``use_spatial_propagation``).

    Returns
    -------
    DisoOF
        Pickle-safe callable (lazy OpenCV initialization) computing DIS
        optical flow.

    Raises
    ------
    TypeError
        If ``kwargs`` contains keys that are not :class:`DisoOF`
        constructor parameters.

    See Also
    --------
    DisoOF : The wrapper class returned by this factory.
    pyflowreg.core.backend_registry.register_backend : Backend registration.
    """
    return DisoOF(**kwargs)
