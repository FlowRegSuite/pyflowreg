"""
Optical Flow Options Configuration Module
==========================================

Python port of MATLAB ``OF_options`` using Pydantic v2 for validation/IO
with full MATLAB compatibility including proper private attributes,
reference preregistration, and edge case handling.

This module provides the OFOptions class which centralizes all configuration
for optical flow motion correction, including:

- I/O settings (input file, output path/format/filename, channel selection)
- Flow parameters (alpha, levels, eta, iterations, constancy assumption,
  optional graduated non-convexity schedule)
- Preprocessing options (Gaussian filtering, spatial binning)
- Reference frame configuration (frame indices, image file, or array,
  with optional preregistration)
- Flow backend selection and backend-specific parameters

Classes
-------
OFOptions
    Main configuration class with Pydantic v2 validation.
OutputFormat
    Enum of supported output formats (TIFF, HDF5, MAT, multi-file
    variants, ARRAY, NULL).
QualitySetting
    Enum of quality presets (quality, balanced, fast, custom).
ChannelNormalization
    Enum of channel normalization modes (joint, separate).
InterpolationMethod
    Enum of warp interpolation methods (nearest, linear, cubic).
ConstancyAssumption
    Enum of optical-flow data terms (gray, gc, cs).
NamingConvention
    Enum of output filename styles (default, batch).

Examples
--------
>>> from pyflowreg.motion_correction import OFOptions
>>> options = OFOptions(quality_setting="fast")
>>> options.input_file = "data/video.h5"
>>> options.output_format = "TIFF"
"""

from __future__ import annotations

import json
import warnings
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictInt,
    field_validator,
    model_validator,
)

# Optional heavy deps
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

# Import IO backends - these are always available as part of the package
from pyflowreg.util.io._base import VideoReader, VideoWriter
from pyflowreg.core.optical_flow import imregister_wrapper


# Enums
class OutputFormat(str, Enum):
    """
    Supported output formats for motion-corrected video.

    The enum value is passed to
    ``pyflowreg.util.io.factory.get_video_file_writer`` to create the
    matching ``VideoWriter``. The memory formats ``ARRAY`` and ``NULL``
    receive special handling and ignore ``output_path``.

    Attributes
    ----------
    TIFF : str
        Single TIFF file.
    HDF5 : str
        Single HDF5 file.
    MAT : str
        MATLAB MAT-file (the ``OFOptions`` default).
    MULTIFILE_TIFF : str
        One TIFF file per channel.
    MULTIFILE_MAT : str
        One MAT-file per channel.
    MULTIFILE_HDF5 : str
        One HDF5 file per channel.
    CAIMAN_HDF5 : str
        Per-channel HDF5 files written with the ``/mov`` dataset name for
        CaImAn compatibility.
    BEGONIA : str
        Begonia format; not yet implemented in the writer factory.
    SUITE2P_TIFF : str
        TIFF writer created with ``format="suite2p"``.
    ARRAY : str
        In-memory accumulation via ``ArrayWriter``.
    NULL : str
        ``NullVideoWriter`` that discards frames without storage.
    """

    # File formats
    TIFF = "TIFF"
    HDF5 = "HDF5"
    MAT = "MAT"
    MULTIFILE_TIFF = "MULTIFILE_TIFF"
    MULTIFILE_MAT = "MULTIFILE_MAT"
    MULTIFILE_HDF5 = "MULTIFILE_HDF5"
    CAIMAN_HDF5 = "CAIMAN_HDF5"
    BEGONIA = "BEGONIA"
    SUITE2P_TIFF = "SUITE2P_TIFF"

    # Memory formats (special handling - ignores output_path)
    ARRAY = "ARRAY"  # Returns ArrayWriter for in-memory accumulation
    NULL = "NULL"  # Returns NullVideoWriter that discards frames without storage


class QualitySetting(str, Enum):
    """
    Quality presets controlling the finest pyramid level of the solver.

    When ``OFOptions.min_level`` is left at ``-1``, the preset determines
    ``OFOptions.effective_min_level``: ``QUALITY`` resolves to level 0
    (full resolution), ``BALANCED`` to level 4, and ``FAST`` to level 6.
    ``CUSTOM`` is selected automatically when ``min_level`` is set to a
    non-negative value.

    Attributes
    ----------
    QUALITY : str
        Solve down to pyramid level 0 (the ``OFOptions`` default).
    BALANCED : str
        Stop at pyramid level 4.
    FAST : str
        Stop at pyramid level 6.
    CUSTOM : str
        Use the user-supplied ``min_level``.
    """

    QUALITY = "quality"
    BALANCED = "balanced"
    FAST = "fast"
    CUSTOM = "custom"


class ChannelNormalization(str, Enum):
    """
    Normalization modes for multi-channel intensity scaling.

    Attributes
    ----------
    JOINT : str
        Normalize all channels together using a shared min/max (the
        ``OFOptions`` default).
    SEPARATE : str
        Normalize each channel independently with its own min/max.
    """

    JOINT = "joint"
    SEPARATE = "separate"


class InterpolationMethod(str, Enum):
    """
    Interpolation methods for warping frames with displacement fields.

    Attributes
    ----------
    NEAREST : str
        Nearest-neighbor interpolation.
    LINEAR : str
        Bilinear interpolation.
    CUBIC : str
        Bicubic interpolation (the ``OFOptions`` default).
    """

    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"


class ConstancyAssumption(str, Enum):
    """
    Data-term constancy assumptions for the variational optical flow solver.

    The string aliases ``"gradient"``, ``"brightness"`` and ``"census"``
    are normalized to the serialized values ``"gc"``, ``"gray"`` and
    ``"cs"`` when validating ``OFOptions.constancy_assumption``.

    Attributes
    ----------
    GRAY : str
        Gray-value (brightness) constancy, value ``"gray"``.
    GRADIENT : str
        Gradient constancy, value ``"gc"`` (the ``OFOptions`` default).
    CENSUS : str
        Census constancy, value ``"cs"``.
    """

    GRAY = "gray"
    GRADIENT = "gc"
    CENSUS = "cs"


def _normalize_constancy_assumption_value(v):
    """Normalize constancy assumption aliases to serialized option values."""
    if hasattr(v, "value"):
        v = v.value
    if isinstance(v, str):
        aliases = {
            "gradient": ConstancyAssumption.GRADIENT.value,
            "brightness": ConstancyAssumption.GRAY.value,
            "census": ConstancyAssumption.CENSUS.value,
        }
        key = v.strip().lower()
        return aliases.get(key, key)
    return v


class NamingConvention(str, Enum):
    """
    Output filename styles used by ``OFOptions.get_video_writer``.

    Attributes
    ----------
    DEFAULT : str
        Write to ``<output_path>/compensated.<ext>``.
    BATCH : str
        Write to ``<output_path>/<input_name>_compensated.<ext>``, where
        ``<input_name>`` is derived from the input file name.
    """

    DEFAULT = "default"
    BATCH = "batch"


class OFOptions(BaseModel):
    """
    Configuration model for variational optical-flow motion correction.

    Python port of the MATLAB ``OF_options`` class. Centralizes all I/O,
    solver, preprocessing, reference, and runtime settings in a single
    Pydantic v2 model with field validation and normalization (scalar
    ``alpha`` is expanded to a 2-tuple, 1D channel ``weight`` lists are
    normalized to sum to 1, ``sigma`` is reshaped to ``(n_channels, 3)``).
    Options can be saved to and restored from JSON via ``save_options`` /
    ``load_options``, and the model provides factory helpers for the video
    reader/writer (``get_video_reader`` / ``get_video_writer``), the
    reference frame (``get_reference_frame``), and the flow-backend
    callable (``resolve_get_displacement``).

    **Input/Output**

    - ``input_file`` (str, Path, ndarray or VideoReader, default ``None``):
      Input video as a file path, in-memory array, or open reader.
    - ``output_path`` (Path, default ``Path("results")``): Output directory.
    - ``output_format`` (OutputFormat, default ``OutputFormat.MAT``): Output
      format; ``ARRAY`` and ``NULL`` are in-memory formats.
    - ``output_file_name`` (str, default ``None``): Custom output filename
      overriding the naming convention.
    - ``channel_idx`` (list of int, default ``None``): Channel indices to
      process.
    - ``naming_convention`` (NamingConvention, default
      ``NamingConvention.DEFAULT``): Output filename style,
      ``compensated.<ext>`` (default) or ``<input>_compensated.<ext>``
      (batch).

    **Flow parameters**

    - ``alpha`` (float or 2-tuple of float, default ``(1.5, 1.5)``):
      Regularization strength; scalars are expanded to ``(alpha, alpha)``.
    - ``weight`` (list of float or ndarray, default ``[0.5, 0.5]``): Channel
      weights (1D, normalized to sum to 1) or spatial weight maps, 2D
      ``(H, W)`` or 3D ``(H, W, C)``.
    - ``levels`` (int, default ``100``): Maximum number of pyramid levels.
    - ``min_level`` (int, default ``-1``): Finest pyramid level to solve;
      ``-1`` derives the level from ``quality_setting``.
    - ``quality_setting`` (QualitySetting, default
      ``QualitySetting.QUALITY``): Quality preset mapped to an effective
      ``min_level`` (quality=0, balanced=4, fast=6).
    - ``eta`` (float, default ``0.8``): Downsampling factor per pyramid
      level, in (0, 1].
    - ``iterations`` (int, default ``50``): Solver iterations per level.
    - ``update_lag`` (int, default ``5``): Update lag for the non-linear
      diffusion weights.
    - ``a_data`` (float, default ``0.45``): Data-term diffusion parameter,
      in (0, 1].
    - ``a_smooth`` (float, default ``1.0``): Smoothness diffusion parameter.
    - ``constancy_assumption`` (ConstancyAssumption, default
      ``ConstancyAssumption.GRADIENT``): Data term, ``"gc"`` (gradient),
      ``"gray"`` (brightness), or ``"cs"`` (census).
    - ``gnc_schedule`` (tuple of float, default ``None``): Optional
      graduated non-convexity stage weights; must be 1D with at least two
      entries, monotone nondecreasing, starting at 0.0 and ending at 1.0.
    - ``warping_steps`` (int, default ``None``): Optional warp/relinearize
      steps per pyramid level in GNC mode.

    **Preprocessing**

    - ``sigma`` (array-like, default ``[[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]]``):
      Gaussian smoothing ``[sx, sy, st]`` per channel; a single triple is
      applied to all channels.
    - ``bin_size`` (int, default ``1``): Temporal binning factor; the
      reader averages every ``bin_size`` consecutive frames.
    - ``channel_normalization`` (ChannelNormalization, default
      ``ChannelNormalization.JOINT``): Normalize channels jointly or
      separately.
    - ``preproc_funct`` (callable, default ``None``): Optional preprocessing
      callable; excluded from serialization.

    **Reference**

    - ``reference_frames`` (list of int, str, Path or ndarray, default
      ``list(range(50, 500))``): Frame indices to preregister and average,
      an image file path, or a precomputed reference array. Indices beyond
      the recording length are clipped to the last frame and the resulting
      duplicates are removed with a printed warning, so short recordings
      preregister each available frame exactly once.
    - ``update_reference`` (bool, default ``False``): Update reference
      during processing.
    - ``n_references`` (int, default ``1``): Number of references;
      multi-reference mode is not fully implemented and repeats a single
      computed reference.
    - ``min_frames_per_reference`` (int, default ``20``): Minimum frames per
      reference cluster.

    **Pre-alignment**

    - ``cc_initialization`` (bool, default ``False``): Enable
      cross-correlation initialization. Also applied during reference
      preregistration in ``get_reference_frame`` (an improvement over the
      MATLAB reference, whose preregistration is not pre-aligned).
    - ``cc_hw`` (int or 2-tuple of int, default ``256``): Target
      height/width for cross-correlation projections.
    - ``cc_up`` (int, default ``1``): Upsampling factor for subpixel
      cross-correlation accuracy.

    **Backend/runtime**

    - ``flow_backend`` (str, default ``"flowreg"``): Name of the registered
      flow backend (see ``pyflowreg.core.backend_registry``).
    - ``backend_params`` (dict, default ``{}``): Backend-specific keyword
      arguments passed to the backend factory.
    - ``get_displacement_impl`` (callable, default ``None``): Direct
      displacement callable overriding the backend; excluded from
      serialization.
    - ``get_displacement_factory`` (callable, default ``None``): Factory
      producing a displacement callable from ``backend_params``; excluded
      from serialization.
    - ``buffer_size`` (int, default ``400``): Frame buffer size passed to
      the video reader.

    **Misc**

    - ``save_w`` (bool, default ``False``): Save displacement fields.
    - ``save_meta_info`` (bool, default ``True``): Save meta information.
    - ``save_valid_mask`` (bool, default ``False``): Save valid masks.
    - ``save_valid_idx`` (bool, default ``False``): Save valid frame
      indices.
    - ``output_typename`` (str, default ``"double"``): Output dtype tag.
    - ``interpolation_method`` (InterpolationMethod, default
      ``InterpolationMethod.CUBIC``): Interpolation used when warping
      frames.
    - ``update_initialization_w`` (bool, default ``True``): Propagate flow
      initialization across batches.
    - ``verbose`` (bool, default ``False``): Verbose logging.

    Examples
    --------
    >>> from pyflowreg.motion_correction import OFOptions
    >>> opts = OFOptions(quality_setting="balanced", alpha=2.0)
    >>> opts.alpha
    (2.0, 2.0)
    >>> opts.effective_min_level
    4

    See Also
    --------
    pyflowreg.motion_correction.compensate_arr.compensate_arr :
        In-memory motion compensation driven by these options.
    pyflowreg.motion_correction.compensate_recording.compensate_recording :
        File-based motion compensation driven by these options.

    Notes
    -----
    Setting ``min_level`` to a non-negative value switches
    ``quality_setting`` to ``custom``; resetting ``min_level`` to ``-1``
    restores the previous preset. Displacement fields follow the ``(u, v)``
    convention where ``u`` is the horizontal (x) and ``v`` the vertical (y)
    component.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,  # Default Pydantic behavior - appropriate for config objects
        extra="forbid",
    )

    # I/O
    input_file: Optional[Union[str, Path, np.ndarray, VideoReader]] = Field(
        None, description="Path/ndarray/VideoReader for input"
    )
    output_path: Path = Field(Path("results"), description="Output directory")
    output_format: OutputFormat = Field(OutputFormat.MAT, description="Output format")
    output_file_name: Optional[str] = Field(None, description="Custom output filename")
    channel_idx: Optional[List[int]] = Field(
        None, description="Channel indices to process"
    )

    # Flow parameters
    alpha: Union[float, Tuple[float, float]] = Field(
        (1.5, 1.5), description="Regularization strength"
    )
    weight: Union[List[float], np.ndarray] = Field(
        [0.5, 0.5], description="Channel weights"
    )
    levels: StrictInt = Field(100, ge=1, description="Number of pyramid levels")
    min_level: StrictInt = Field(
        -1, ge=-1, description="Min pyramid level; -1 = from preset"
    )
    quality_setting: QualitySetting = Field(
        QualitySetting.QUALITY, description="Quality preset"
    )
    eta: float = Field(0.8, gt=0, le=1, description="Downsample factor per level")
    update_lag: StrictInt = Field(
        5, ge=1, description="Update lag for non-linear diffusion"
    )
    iterations: StrictInt = Field(50, ge=1, description="Iterations per level")
    a_smooth: float = Field(1.0, ge=0, description="Smoothness diffusion parameter")
    a_data: float = Field(0.45, gt=0, le=1, description="Data-term diffusion parameter")
    gnc_schedule: Optional[Tuple[float, ...]] = Field(
        None,
        description="Optional graduated non-convexity stage weights from 0.0 to 1.0",
    )
    warping_steps: Optional[StrictInt] = Field(
        None,
        ge=1,
        description="Optional warp/relinearize steps per pyramid level in GNC mode",
    )

    # Preprocessing
    sigma: Any = Field(
        [[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]],
        description="Gaussian [sx, sy, st] per-channel",
    )
    bin_size: StrictInt = Field(
        1, ge=1, description="Temporal binning factor (frames averaged on read)"
    )
    buffer_size: StrictInt = Field(400, ge=1, description="Frame buffer size")

    # Reference
    reference_frames: Union[List[int], str, Path, np.ndarray] = Field(
        list(range(50, 500)), description="Indices, path, or ndarray for reference"
    )
    update_reference: bool = Field(
        False, description="Update reference during processing"
    )
    n_references: StrictInt = Field(1, ge=1, description="Number of references")
    min_frames_per_reference: StrictInt = Field(
        20, ge=1, description="Min frames per reference cluster"
    )

    # Processing options
    verbose: bool = Field(False, description="Verbose logging")
    save_meta_info: bool = Field(True, description="Save meta info")
    save_w: bool = Field(False, description="Save displacement fields")
    save_valid_mask: bool = Field(False, description="Save valid masks")
    save_valid_idx: bool = Field(False, description="Save valid frame indices")
    output_typename: Optional[str] = Field("double", description="Output dtype tag")
    channel_normalization: ChannelNormalization = Field(
        ChannelNormalization.JOINT, description="Normalization mode"
    )
    interpolation_method: InterpolationMethod = Field(
        InterpolationMethod.CUBIC, description="Warp interpolation"
    )
    cc_initialization: bool = Field(
        False, description="Cross-correlation initialization"
    )
    cc_hw: Union[int, Tuple[int, int]] = Field(
        256, description="Target HW size for CC projections"
    )
    cc_up: int = Field(
        1, ge=1, description="Upsampling factor for subpixel CC accuracy"
    )
    update_initialization_w: bool = Field(
        True, description="Propagate flow init across batches"
    )
    naming_convention: NamingConvention = Field(
        NamingConvention.DEFAULT, description="Output filename style"
    )
    constancy_assumption: ConstancyAssumption = Field(
        ConstancyAssumption.GRADIENT,
        description="Optical-flow data term: 'gc', 'gray', or 'cs'",
    )

    # Backend configuration
    flow_backend: str = Field("flowreg", description="Flow backend name")
    backend_params: Dict[str, Any] = Field(
        default_factory=dict, description="Backend-specific parameters"
    )

    # Non-serializable/runtime
    preproc_funct: Optional[Callable] = Field(None, exclude=True)
    get_displacement_impl: Optional[Callable] = Field(
        None, exclude=True, description="Direct displacement callable"
    )
    get_displacement_factory: Optional[Callable[..., Callable]] = Field(
        None, exclude=True, description="Factory for displacement callable"
    )

    # Private attributes (using PrivateAttr for Pydantic v2)
    _video_reader: Optional[VideoReader] = PrivateAttr(default=None)
    _video_writer: Optional[VideoWriter] = PrivateAttr(default=None)
    _quality_setting_old: QualitySetting = PrivateAttr(default=QualitySetting.QUALITY)
    _datatype: str = PrivateAttr(default="NONE")

    @field_validator("alpha", mode="before")
    @classmethod
    def normalize_alpha(cls, v):
        """Normalize alpha to always be a 2-tuple of positive floats."""
        if isinstance(v, (int, float)):
            if v <= 0:
                raise ValueError("Alpha must be positive")
            return (float(v), float(v))
        elif isinstance(v, (list, tuple)):
            if len(v) == 1:
                if v[0] <= 0:
                    raise ValueError("Alpha must be positive")
                return (float(v[0]), float(v[0]))
            elif len(v) == 2:
                if v[0] <= 0 or v[1] <= 0:
                    raise ValueError("All alpha values must be positive")
                return (float(v[0]), float(v[1]))
            else:
                raise ValueError("Alpha must be scalar or 2-element tuple")
        else:
            raise ValueError("Alpha must be scalar or 2-element tuple")

    @field_validator("weight", mode="before")
    @classmethod
    def normalize_weight(cls, v):
        """Normalize weight values to sum to 1.

        Accepts:
        - List [1, 2]: normalized to [0.33, 0.67]
        - 1D numpy array: normalized and converted to list
        - 2D numpy array (H, W): spatial weight map for single channel
        - 3D numpy array (H, W, C): spatial weight maps from preregistration
        """
        if isinstance(v, np.ndarray):
            if v.ndim == 1:
                # 1D weight array: normalize and convert to list for JSON serialization
                weight_sum = v.sum()
                if weight_sum > 0:
                    return (v / weight_sum).tolist()
                return v.tolist()
            elif v.ndim <= 3:
                # 2D/3D arrays (spatial weight maps from preregistration)
                # Keep as numpy array - don't convert to nested lists
                # Pydantic v2 with arbitrary_types_allowed=True handles this correctly
                return v
            else:
                # Weight is spatial only, not temporal
                raise ValueError(
                    f"Weight array cannot exceed 3 dimensions (got {v.ndim}D array). "
                    "Weight must be either channel weights (1D) or spatial weight maps (2D/3D)."
                )
        elif isinstance(v, (list, tuple)):
            # List or tuple: normalize if 1D
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 1:
                weight_sum = arr.sum()
                if weight_sum > 0:
                    return (arr / weight_sum).tolist()
            return v
        return v

    @field_validator("sigma", mode="before")
    @classmethod
    def normalize_sigma(cls, v):
        """Normalize sigma to correct shape."""
        sig = np.asarray(v, dtype=float)
        if sig.ndim == 1:
            if sig.size != 3:
                raise ValueError("1D sigma must be [sx, sy, st]")
            return sig.reshape(1, 3).tolist()
        elif sig.ndim == 2:
            if sig.shape[1] != 3:
                raise ValueError("2D sigma must be (n_channels, 3)")
            return sig.tolist()
        else:
            raise ValueError("Sigma must be [sx,sy,st] or (n_channels, 3)")
        return v

    @field_validator("gnc_schedule", mode="before")
    @classmethod
    def normalize_gnc_schedule(cls, v):
        """Normalize and validate an optional GNC stage schedule."""
        if v is None:
            return None

        schedule = np.asarray(v, dtype=float)
        if schedule.ndim != 1:
            raise ValueError("gnc_schedule must be a 1D sequence")
        if schedule.size < 2:
            raise ValueError("gnc_schedule must contain at least two stages")
        if np.any(schedule < 0.0) or np.any(schedule > 1.0):
            raise ValueError("gnc_schedule entries must lie in [0, 1]")
        if not np.all(np.diff(schedule) >= 0.0):
            raise ValueError("gnc_schedule must be monotone nondecreasing")
        if not np.isclose(schedule[0], 0.0):
            raise ValueError("gnc_schedule must start at 0.0")
        if not np.isclose(schedule[-1], 1.0):
            raise ValueError("gnc_schedule must end at 1.0")
        return tuple(float(x) for x in schedule.tolist())

    @field_validator("constancy_assumption", mode="before")
    @classmethod
    def normalize_constancy_assumption(cls, v):
        """Normalize constancy assumption aliases to serialized option values."""
        return _normalize_constancy_assumption_value(v)

    @model_validator(mode="after")
    def validate_and_normalize(self) -> "OFOptions":
        """Normalize fields and maintain MATLAB parity."""
        # Path conversion
        if not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)

        # Quality setting logic (MATLAB parity)
        if self.quality_setting != QualitySetting.CUSTOM:
            self._quality_setting_old = self.quality_setting

        if self.min_level >= 0:
            self.quality_setting = QualitySetting.CUSTOM
        elif self.min_level == -1 and self.quality_setting == QualitySetting.CUSTOM:
            self.quality_setting = self._quality_setting_old

        return self

    @property
    def effective_min_level(self) -> int:
        """Get effective min_level based on quality setting."""
        if self.min_level >= 0:
            return self.min_level

        mapping = {
            QualitySetting.QUALITY: 0,
            QualitySetting.BALANCED: 4,
            QualitySetting.FAST: 6,
            QualitySetting.CUSTOM: max(self.min_level, 0),
        }
        return mapping.get(self.quality_setting, 0)

    def get_sigma_at(self, i: int) -> np.ndarray:
        """Get sigma for channel i (0-indexed)."""
        sig = np.asarray(self.sigma, dtype=float)

        # If sigma is 1D, return it for all channels
        if sig.ndim == 1:
            return sig

        # If sigma is 2D, return row for channel i
        if i >= sig.shape[0]:
            if self.verbose:
                print(f"Sigma for channel {i} not specified, using channel 0")
            return sig[0]

        return sig[i]

    def get_weight_at(self, i: int, n_channels: int) -> Union[float, np.ndarray]:
        """Get weight for channel i (0-indexed)."""
        w = np.asarray(self.weight, dtype=float)

        # Handle scalar or 1D weights
        if w.ndim <= 1:
            if w.size == 1:
                return float(w.item())

            # Truncate if too many weights
            if w.size > n_channels:
                w = w[:n_channels]
                w = w / w.sum()  # Renormalize
                self.weight = w.tolist()

            if i >= w.size:
                if self.verbose:
                    print(f"Weight for channel {i} not set, using 1/n_channels")
                return 1.0 / n_channels

            return float(w[i])

        # Handle 2D or 3D weights (spatial weights)
        # 2D: (H, W) - single channel spatial weight map
        # 3D: (H, W, C) - multi-channel spatial weight map (channel-last)

        if w.ndim == 2:
            # 2D weight map - return for channel 0, otherwise uniform weight
            if i == 0:
                return w
            else:
                if self.verbose:
                    print(f"Weight for channel {i} not set, using uniform weight")
                return np.ones_like(w) / n_channels

        elif w.ndim == 3:
            # 3D weight map in channel-last format (H, W, C)
            if i >= w.shape[2]:
                if self.verbose:
                    print(f"Weight for channel {i} not set, using 1/n_channels")
                return np.ones(w.shape[:2]) / n_channels

            return w[:, :, i]

        else:
            raise ValueError(f"Unexpected weight array with {w.ndim} dimensions")

    def copy(self) -> "OFOptions":
        """Create a deep copy (MATLAB copyable interface)."""
        return self.model_copy(deep=True)

    def get_video_reader(self) -> VideoReader:
        """Get or create video reader (mirrors MATLAB get_video_file_reader)."""
        # Return cached reader if available
        if self._video_reader is not None:
            return self._video_reader

        # If input_file is already a VideoReader, use it directly
        if isinstance(self.input_file, VideoReader):
            self._video_reader = self.input_file
            return self._video_reader

        # Call factory function to create reader (matches MATLAB behavior)
        from pyflowreg.util.io.factory import get_video_file_reader

        self._video_reader = get_video_file_reader(
            self.input_file, buffer_size=self.buffer_size, bin_size=self.bin_size
        )

        # Store reader back in input_file (matches MATLAB line 247)
        self.input_file = self._video_reader

        return self._video_reader

    def get_video_writer(self) -> VideoWriter:
        """Get or create video writer (mirrors MATLAB get_video_writer)."""
        # Return cached writer if available
        if self._video_writer is not None:
            return self._video_writer

        # Determine filename (matches MATLAB lines 258-269)
        if self.output_file_name:
            filename = self.output_file_name
        else:
            if self.naming_convention == NamingConvention.DEFAULT:
                # Extension from output_format enum value
                ext = (
                    "HDF5"
                    if self.output_format == OutputFormat.HDF5
                    else self.output_format.value
                )
                filename = str(self.output_path / f"compensated.{ext}")
            else:
                reader = self.get_video_reader()
                input_name = Path(getattr(reader, "input_file_name", "output")).stem
                ext = (
                    "HDF5"
                    if self.output_format == OutputFormat.HDF5
                    else self.output_format.value
                )
                filename = str(self.output_path / f"{input_name}_compensated.{ext}")

        # Call factory function to create writer (matches MATLAB)
        from pyflowreg.util.io.factory import get_video_file_writer

        self._video_writer = get_video_file_writer(filename, self.output_format.value)

        return self._video_writer

    def get_reference_frame(
        self,
        video_reader: Optional[VideoReader] = None,
        registration_config: Optional[Any] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get reference frame(s), with optional preregistration.

        Index lists are clipped to the recording length (duplicates from
        clipping are removed with a printed warning); the preregistration
        honors ``cc_initialization``/``cc_hw``/``cc_up``.
        """
        if self.n_references > 1:
            warnings.warn(
                "Multi-reference mode not fully implemented; repeating a single computed reference"
            )
            # Create a copy with n_references=1 to avoid recursion
            single_ref_opts = self.model_copy(update={"n_references": 1})
            ref = single_ref_opts.get_reference_frame(
                video_reader, registration_config=registration_config
            )
            return [ref] * self.n_references

        # Direct ndarray
        if isinstance(self.reference_frames, np.ndarray):
            return self.reference_frames

        # Path to image file
        if isinstance(self.reference_frames, (str, Path)):
            p = Path(self.reference_frames)
            if p.suffix.lower() in (".tif", ".tiff"):
                return tifffile.imread(str(p))
            try:
                import imageio.v3 as iio

                return iio.imread(str(p))
            except ImportError as e:
                raise RuntimeError(f"Unable to read reference image: {p}") from e

        # List of frame indices - preregister
        if isinstance(self.reference_frames, list) and video_reader is not None:
            # Get actual frame count and clip reference indices to valid range
            frame_count = len(video_reader)
            valid_indices = []
            clipped = False

            for idx in self.reference_frames:
                if idx >= frame_count:
                    valid_indices.append(min(idx, frame_count - 1))
                    clipped = True
                else:
                    valid_indices.append(idx)

            if clipped:
                # Deduplicate (order-preserving): clipping maps every
                # out-of-range index to the last frame, which would
                # otherwise be read, flow-solved, and averaged hundreds of
                # times, massively over-weighting it in the reference.
                # Intentional duplicates in fully in-range user lists are
                # preserved (no dedup when nothing was clipped).
                n_requested = len(valid_indices)
                valid_indices = list(dict.fromkeys(valid_indices))
                n_removed = n_requested - len(valid_indices)
                print(
                    f"Warning: Reference frames exceed video length ({frame_count} frames). "
                    f"Clipping indices from {self.reference_frames[0]}-{self.reference_frames[-1]} "
                    f"to {valid_indices[0]}-{valid_indices[-1]} and removing "
                    f"{n_removed} duplicate indices."
                )

            frames = video_reader[valid_indices]  # (T,H,W,C) using array-like indexing

            if frames.ndim != 4:
                if frames.ndim == 3:
                    return frames  # Single frame (H,W,C)
                raise ValueError("read_frames must return (H,W,C) or (T,H,W,C)")

            # Convert from (T,H,W,C) to (H,W,C,T) for compatibility
            frames = np.transpose(frames, (1, 2, 3, 0))  # Now (H,W,C,T)

            # Single frame
            if frames.shape[3] == 1:
                return frames[:, :, :, 0]

            n_channels = frames.shape[2]

            # Build weight array
            weight_2d = np.zeros((frames.shape[0], frames.shape[1], n_channels))
            for c in range(n_channels):
                weight_2d[:, :, c] = self.get_weight_at(c, n_channels)

            if self.verbose:
                print("Preregistering reference frames...")

            # Preprocess with extra smoothing for preregistration
            if gaussian_filter is not None:
                frames_smooth = np.zeros_like(frames)
                for c in range(n_channels):
                    # sig is (sx, sy, st); the per-channel slice is
                    # (H, W, T), so reorder to scipy's axis order (sy, sx, st).
                    sig = self.get_sigma_at(c) + np.array([1, 1, 0.5])
                    frames_smooth[:, :, c, :] = gaussian_filter(
                        frames[:, :, c, :],
                        sigma=(sig[1], sig[0], sig[2]),
                        mode="reflect",
                    )
            else:
                frames_smooth = frames

            # Normalize
            if self.channel_normalization == ChannelNormalization.SEPARATE:
                frames_norm = np.zeros_like(frames_smooth)
                for c in range(n_channels):
                    ch = frames_smooth[:, :, c, :]
                    ch_min = ch.min()
                    ch_max = ch.max()
                    frames_norm[:, :, c, :] = (ch - ch_min) / (ch_max - ch_min + 1e-8)
            else:
                f_min = frames_smooth.min()
                f_max = frames_smooth.max()
                frames_norm = (frames_smooth - f_min) / (f_max - f_min + 1e-8)

            # Mean as initial reference
            ref_mean = np.mean(frames_norm, axis=3)

            # Compensate using stronger regularization for preregistration
            from pyflowreg.motion_correction.compensate_arr import compensate_arr

            # Use stronger regularization for preregistration
            alpha_prereg = (
                tuple(a + 2.0 for a in self.alpha)
                if isinstance(self.alpha, tuple)
                else self.alpha + 2.0
            )

            # Create a temporary OFOptions for preregistration. The
            # cross-correlation pre-alignment settings are forwarded so an
            # enabled cc_initialization also applies during reference
            # preregistration (the MATLAB reference does not pre-align its
            # preregistration; this is a deliberate improvement).
            prereg_options = OFOptions(
                alpha=alpha_prereg,
                levels=self.levels,
                min_level=self.effective_min_level,
                eta=self.eta,
                update_lag=self.update_lag,
                iterations=self.iterations,
                a_smooth=self.a_smooth,
                a_data=self.a_data,
                constancy_assumption=self.constancy_assumption,
                weight=weight_2d,
                buffer_size=self.buffer_size,
                cc_initialization=self.cc_initialization,
                cc_hw=self.cc_hw,
                cc_up=self.cc_up,
            )

            # Reshape frames_norm from (H,W,C,T) to (T,H,W,C) for compensate_arr
            frames_for_compensation = np.transpose(frames_norm, (3, 0, 1, 2))

            # Compensate: compute displacement fields using normalized frames
            _, w_fields = compensate_arr(
                frames_for_compensation,
                ref_mean,
                options=prereg_options,
                registration_config=registration_config,
            )

            # Warp the RAW frames using the computed displacement fields
            frames_raw_for_warp = np.transpose(frames, (3, 0, 1, 2))  # (T,H,W,C)
            ref_mean_raw = np.mean(frames_raw_for_warp, axis=0)  # (H,W,C)

            compensated_raw = np.zeros_like(frames_raw_for_warp)
            for t in range(frames_raw_for_warp.shape[0]):
                warped = imregister_wrapper(
                    frames_raw_for_warp[t],
                    w_fields[t, :, :, 0],  # u
                    w_fields[t, :, :, 1],  # v
                    ref_mean_raw,
                    interpolation_method="cubic",
                )
                if warped.ndim == 2:
                    warped = warped[:, :, np.newaxis]
                compensated_raw[t] = warped

            # Calculate mean of compensated RAW frames as the reference
            reference = np.mean(compensated_raw, axis=0)

            if self.verbose:
                print("Finished pre-registration of the reference frames.")

            return reference

        # Fallback
        return np.asarray(self.reference_frames)

    def save_options(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """Save options to JSON with MATLAB-compatible header."""
        path = Path(filepath) if filepath else self.output_path / "options.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for JSON
        data = self.model_dump(
            exclude={
                "preproc_funct",
                "_video_reader",
                "_video_writer",
                "_quality_setting_old",
                "_datatype",
            }
        )

        # Convert non-JSON types
        for k, v in list(data.items()):
            if isinstance(v, Path):
                data[k] = str(v)
            elif isinstance(v, np.ndarray):
                data[k] = v.tolist()

        # Handle reference frames if ndarray
        if isinstance(self.reference_frames, np.ndarray):
            ref_path = path.parent / "reference_frames.tif"
            tifffile.imwrite(str(ref_path), self.reference_frames)
            data["reference_frames"] = str(ref_path)

        # Write with MATLAB header
        with path.open("w", encoding="utf-8") as f:
            f.write(f"Compensation options {date.today().isoformat()}\n\n")
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"Options saved to {path}")

    @classmethod
    def load_options(cls, filepath: Union[str, Path]) -> "OFOptions":
        """Load options from JSON (MATLAB or Python format)."""
        p = Path(filepath)

        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip header lines (MATLAB compatibility)
        json_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break

        json_text = "".join(lines[json_start:])
        data = json.loads(json_text)

        # Load reference frames if file path
        ref = data.get("reference_frames")
        if isinstance(ref, str):
            ref_path = Path(ref)
            if ref_path.exists() and ref_path.suffix.lower() in (".tif", ".tiff"):
                data["reference_frames"] = tifffile.imread(str(ref_path))

        return cls(**data)

    def resolve_get_displacement(self) -> Callable:
        """
        Resolve the displacement computation callable from the configuration.

        Resolution priority:

        1. ``get_displacement_impl`` -- returned directly if set.
        2. ``get_displacement_factory`` -- called with ``**backend_params``
           if set.
        3. ``flow_backend`` -- the named backend factory is looked up in
           ``pyflowreg.core.backend_registry`` and called with
           ``**backend_params``.

        Before the registry lookup, the ``'diso'`` backend is guarded: it
        only supports the gradient-constancy data term and no graduated
        non-convexity, so a ``ValueError`` is raised if
        ``constancy_assumption`` normalizes to anything other than ``'gc'``
        or if ``gnc_schedule`` or ``warping_steps`` is set.

        Returns
        -------
        Callable
            Function computing the optical-flow displacement field.

        Raises
        ------
        ValueError
            If ``flow_backend == 'diso'`` is combined with a non-``'gc'``
            constancy assumption or with ``gnc_schedule``/``warping_steps``,
            or if the named backend is not registered.
        """
        # Priority 1: Direct implementation override
        if self.get_displacement_impl is not None:
            return self.get_displacement_impl

        # Priority 2: Factory override
        if self.get_displacement_factory is not None:
            return self.get_displacement_factory(**self.backend_params)

        # Priority 3: Registry backend
        from pyflowreg.core.backend_registry import get_backend

        constancy_assumption = _normalize_constancy_assumption_value(
            self.constancy_assumption
        )
        if self.flow_backend == "diso" and constancy_assumption != "gc":
            raise ValueError(
                "The 'diso' backend does not support variational constancy "
                f"assumption '{constancy_assumption}'. Use "
                "flow_backend='flowreg' for 'gray' or 'cs'."
            )
        if self.flow_backend == "diso" and (
            self.gnc_schedule is not None or self.warping_steps is not None
        ):
            raise ValueError(
                "The 'diso' backend does not support graduated non-convexity. "
                "Use flow_backend='flowreg' for 'gnc_schedule' or "
                "'warping_steps'."
            )

        factory = get_backend(self.flow_backend)
        return factory(**self.backend_params)

    def to_dict(self) -> dict:
        """Get parameters dict for optical flow functions."""
        return {
            "alpha": self.alpha,
            "weight": self.weight,
            "levels": self.levels,
            "min_level": self.effective_min_level,
            "eta": self.eta,
            "iterations": self.iterations,
            "update_lag": self.update_lag,
            "a_data": self.a_data,
            "a_smooth": self.a_smooth,
            "gnc_schedule": self.gnc_schedule,
            "warping_steps": self.warping_steps,
            "const_assumption": _normalize_constancy_assumption_value(
                self.constancy_assumption
            ),
        }

    def __repr__(self) -> str:
        return (
            f"OFOptions(quality={self.quality_setting.value}, alpha={self.alpha}, "
            f"levels={self.levels}, min_level={self.effective_min_level})"
        )


# Convenience functions
def get_mcp_schema() -> dict:
    """
    Get the JSON schema for the OFOptions model.

    Returns
    -------
    dict
        JSON schema produced by ``OFOptions.model_json_schema()``.
    """
    return OFOptions.model_json_schema()


if __name__ == "__main__":
    # Test basic functionality
    opts = OFOptions(
        input_file="test.h5",
        output_path=Path("./results"),
        quality_setting=QualitySetting.BALANCED,
        alpha=2.0,
        weight=[0.6, 0.4],
    )

    print(opts)
    print("Effective min_level:", opts.effective_min_level)

    # Test save/load
    out_path = Path("test_options.json")
    opts.save_options(out_path)
    loaded = OFOptions.load_options(out_path)
    print("Load/save test passed")
