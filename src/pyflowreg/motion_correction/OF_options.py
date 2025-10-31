"""
Optical Flow Options Configuration Module (Python) - Fixed Version
------------------------------------------------------------------

Python port of MATLAB `OF_options` using Pydantic v2 for validation/IO
with full MATLAB compatibility including proper private attributes,
preregistration, and edge case handling.
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
    QUALITY = "quality"
    BALANCED = "balanced"
    FAST = "fast"
    CUSTOM = "custom"


class ChannelNormalization(str, Enum):
    JOINT = "joint"
    SEPARATE = "separate"


class InterpolationMethod(str, Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"


class ConstancyAssumption(str, Enum):
    GRAY = "gray"
    GRADIENT = "gc"


class NamingConvention(str, Enum):
    DEFAULT = "default"
    BATCH = "batch"


class OFOptions(BaseModel):
    """Python port of MATLAB OF_options class."""

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

    # Preprocessing
    sigma: Any = Field(
        [[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]],
        description="Gaussian [sx, sy, st] per-channel",
    )
    bin_size: StrictInt = Field(1, ge=1, description="Spatial binning factor")
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
        ConstancyAssumption.GRADIENT, description="Constancy assumption"
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
                return float(w)

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
        self, video_reader: Optional[VideoReader] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get reference frame(s), with optional preregistration."""
        if self.n_references > 1:
            warnings.warn(
                "Multi-reference mode not fully implemented; repeating a single computed reference"
            )
            # Create a copy with n_references=1 to avoid recursion
            single_ref_opts = self.model_copy(update={"n_references": 1})
            ref = single_ref_opts.get_reference_frame(video_reader)
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
                print(
                    f"Warning: Reference frames exceed video length ({frame_count} frames). "
                    f"Clipping indices from {self.reference_frames[0]}-{self.reference_frames[-1]} "
                    f"to {valid_indices[0]}-{valid_indices[-1]}"
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
                    sig = self.get_sigma_at(c) + np.array([1, 1, 0.5])
                    frames_smooth[:, :, c, :] = gaussian_filter(
                        frames[:, :, c, :], sigma=tuple(sig), mode="reflect"
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

            # Create a temporary OFOptions for preregistration
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
            )

            # Reshape frames_norm from (H,W,C,T) to (T,H,W,C) for compensate_arr
            frames_for_compensation = np.transpose(frames_norm, (3, 0, 1, 2))

            # Compensate: compute displacement fields using normalized frames
            _, w_fields = compensate_arr(
                frames_for_compensation, ref_mean, options=prereg_options
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
        Resolve the displacement computation function based on configuration.

        Priority order:
        1. get_displacement_impl (direct callable)
        2. get_displacement_factory with backend_params
        3. flow_backend from registry with backend_params

        Returns:
            Callable for computing optical flow
        """
        # Priority 1: Direct implementation override
        if self.get_displacement_impl is not None:
            return self.get_displacement_impl

        # Priority 2: Factory override
        if self.get_displacement_factory is not None:
            return self.get_displacement_factory(**self.backend_params)

        # Priority 3: Registry backend
        from pyflowreg.core.backend_registry import get_backend

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
            "const_assumption": self.constancy_assumption.value,  # Fixed: use const_assumption for API compatibility
        }

    def __repr__(self) -> str:
        return (
            f"OFOptions(quality={self.quality_setting.value}, alpha={self.alpha}, "
            f"levels={self.levels}, min_level={self.effective_min_level})"
        )


# Convenience functions
def compensate_inplace(
    frames: np.ndarray,
    reference: np.ndarray,
    options: Optional[OFOptions] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compensate frames against reference.

    Returns:
        Tuple of (compensated_frames, displacement_fields)
    """
    if options is None:
        options = OFOptions(**kwargs)
    else:
        # Copy and update
        options = options.model_copy(update=kwargs)

    # Ensure 4D frames and 3D reference
    if frames.ndim == 3:
        frames = frames[:, :, np.newaxis, :]
    if reference.ndim == 2:
        reference = reference[:, :, np.newaxis]

    params = options.to_dict()

    try:
        from pyflowreg import get_displacement, compensate_sequence_uv
    except ImportError as e:
        raise RuntimeError("pyflowreg core functions not available") from e

    # Compute displacements
    T = frames.shape[3]
    displacements = np.zeros((frames.shape[0], frames.shape[1], 2, T), dtype=np.float32)

    for t in range(T):
        displacements[:, :, :, t] = get_displacement(
            reference, frames[:, :, :, t], **params
        )

    # Apply compensation
    compensated = compensate_sequence_uv(frames, reference, displacements)

    return compensated, displacements


def get_mcp_schema() -> dict:
    """Get JSON schema for the model."""
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
