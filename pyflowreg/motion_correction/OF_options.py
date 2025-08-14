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
from copy import deepcopy
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import tifffile
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StrictInt, field_validator, model_validator

# Optional heavy deps
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

# Optional IO backends
try:
    from pyflowreg.util.io.hdf5 import HDF5FileReader, HDF5FileWriter
    from pyflowreg.util.io.mdf import MDFFileReader
    from pyflowreg.util.io._base import VideoReader as _VideoReaderBase, VideoWriter as _VideoWriterBase
    from pyflowreg.util.io.tiff import TIFFStackReader, TIFFStackWriter
except ImportError:
    _VideoReaderBase = object
    _VideoWriterBase = object
    HDF5FileReader = None
    HDF5FileWriter = None
    MDFFileReader = None
    TIFFStackReader = None
    TIFFStackWriter = None


# Enums
class OutputFormat(str, Enum):
    TIFF = "TIFF"
    HDF5 = "HDF5"
    MAT = "MAT"
    MULTIFILE_TIFF = "MULTIFILE_TIFF"
    MULTIFILE_MAT = "MULTIFILE_MAT"
    MULTIFILE_HDF5 = "MULTIFILE_HDF5"
    CAIMAN_HDF5 = "CAIMAN_HDF5"
    BEGONIA = "BEGONIA"
    SUITE2P_TIFF = "SUITE2P_TIFF"


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
        validate_assignment=True,
        extra="forbid"
    )

    # I/O
    input_file: Optional[Union[str, Path, np.ndarray, _VideoReaderBase]] = Field(
        None, description="Path/ndarray/VideoReader for input"
    )
    output_path: Path = Field(Path("results"), description="Output directory")
    output_format: OutputFormat = Field(OutputFormat.MAT, description="Output format")
    output_file_name: Optional[str] = Field(None, description="Custom output filename")
    channel_idx: Optional[List[int]] = Field(None, description="Channel indices to process")

    # Flow parameters
    alpha: Union[float, Tuple[float, float]] = Field(1.5, gt=0, description="Regularization strength")
    weight: Union[List[float], np.ndarray] = Field([0.5, 0.5], description="Channel weights")
    levels: StrictInt = Field(100, ge=1, description="Number of pyramid levels")
    min_level: StrictInt = Field(-1, ge=-1, description="Min pyramid level; -1 = from preset")
    quality_setting: QualitySetting = Field(QualitySetting.QUALITY, description="Quality preset")
    eta: float = Field(0.8, gt=0, le=1, description="Downsample factor per level")
    update_lag: StrictInt = Field(5, ge=1, description="Update lag for non-linear diffusion")
    iterations: StrictInt = Field(50, ge=1, description="Iterations per level")
    a_smooth: float = Field(1.0, ge=0, description="Smoothness diffusion parameter")
    a_data: float = Field(0.45, gt=0, le=1, description="Data-term diffusion parameter")

    # Preprocessing
    sigma: Union[List[float], np.ndarray] = Field(
        [[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]],
        description="Gaussian [sx, sy, st] per-channel"
    )
    bin_size: StrictInt = Field(1, ge=1, description="Spatial binning factor")
    buffer_size: StrictInt = Field(400, ge=1, description="Frame buffer size")

    # Reference
    reference_frames: Union[List[int], str, Path, np.ndarray] = Field(
        list(range(50, 500)), description="Indices, path, or ndarray for reference"
    )
    update_reference: bool = Field(False, description="Update reference during processing")
    n_references: StrictInt = Field(1, ge=1, description="Number of references")
    min_frames_per_reference: StrictInt = Field(20, ge=1, description="Min frames per reference cluster")

    # Processing options
    verbose: bool = Field(False, description="Verbose logging")
    save_meta_info: bool = Field(True, description="Save meta info")
    save_w: bool = Field(False, description="Save displacement fields")
    save_valid_mask: bool = Field(False, description="Save valid masks")
    save_valid_idx: bool = Field(False, description="Save valid frame indices")
    output_typename: Optional[str] = Field("double", description="Output dtype tag")
    channel_normalization: ChannelNormalization = Field(ChannelNormalization.JOINT, description="Normalization mode")
    interpolation_method: InterpolationMethod = Field(InterpolationMethod.CUBIC, description="Warp interpolation")
    cc_initialization: bool = Field(False, description="Cross-correlation initialization")
    update_initialization_w: bool = Field(True, description="Propagate flow init across batches")
    naming_convention: NamingConvention = Field(NamingConvention.DEFAULT, description="Output filename style")
    constancy_assumption: ConstancyAssumption = Field(ConstancyAssumption.GRADIENT, description="Constancy assumption")

    # Non-serializable/runtime
    preproc_funct: Optional[Callable] = Field(None, exclude=True)

    # Private attributes (using PrivateAttr for Pydantic v2)
    _video_reader: Optional[_VideoReaderBase] = PrivateAttr(default=None)
    _video_writer: Optional[_VideoWriterBase] = PrivateAttr(default=None)
    _quality_setting_old: QualitySetting = PrivateAttr(default=QualitySetting.QUALITY)
    _datatype: str = PrivateAttr(default="NONE")

    @model_validator(mode="after")
    def validate_and_normalize(self) -> "OFOptions":
        """Normalize fields and maintain MATLAB parity."""
        # Path conversion
        if not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)

        # Alpha normalization
        if isinstance(self.alpha, (int, float)):
            self.alpha = (float(self.alpha), float(self.alpha))
        elif isinstance(self.alpha, (list, tuple)):
            if len(self.alpha) == 1:
                self.alpha = (float(self.alpha[0]), float(self.alpha[0]))
            elif len(self.alpha) == 2:
                self.alpha = (float(self.alpha[0]), float(self.alpha[1]))
            else:
                raise ValueError("Alpha must be scalar or 2-element tuple")

        # Weight normalization
        if isinstance(self.weight, np.ndarray):
            if self.weight.ndim == 1:
                weight_sum = self.weight.sum()
                if weight_sum > 0:
                    self.weight = (self.weight / weight_sum).tolist()
        elif isinstance(self.weight, (list, tuple)):
            arr = np.asarray(self.weight, dtype=float)
            if arr.ndim == 1:
                weight_sum = arr.sum()
                if weight_sum > 0:
                    self.weight = (arr / weight_sum).tolist()

        # Sigma normalization
        sig = np.asarray(self.sigma, dtype=float)
        if sig.ndim == 1:
            if sig.size == 3:
                self.sigma = sig.reshape(1, 3).tolist()
            else:
                raise ValueError("1D sigma must have 3 elements [sx, sy, st]")
        elif sig.ndim == 2:
            if sig.shape[1] != 3:
                raise ValueError("2D sigma must be (n_channels, 3)")
            self.sigma = sig.tolist()
        else:
            raise ValueError("Sigma must be [sx,sy,st] or (n_channels, 3)")

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
            QualitySetting.CUSTOM: max(self.min_level, 0)
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
            if not self.verbose:
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
                if not self.verbose:
                    print(f"Weight for channel {i} not set, using 1/n_channels")
                return 1.0 / n_channels

            return float(w[i])

        # Handle 2D or 3D weights (spatial weights)
        if i >= w.shape[0]:
            if not self.verbose:
                print(f"Weight for channel {i} not set, using 1/n_channels")
            return np.ones(w.shape[1:]) / n_channels

        return w[i]

    def copy(self) -> "OFOptions":
        """Create a deep copy (MATLAB copyable interface)."""
        return self.model_copy(deep=True)

    def get_video_reader(self) -> _VideoReaderBase:
        """Get or create video reader."""
        if self._video_reader is not None:
            return self._video_reader

        if isinstance(self.input_file, _VideoReaderBase):
            self._video_reader = self.input_file
            return self._video_reader

        if isinstance(self.input_file, np.ndarray):
            try:
                from pyflowreg.util.io.matrix import MatrixFileReader
            except ImportError as e:
                raise RuntimeError("MatrixFileReader not available") from e
            self._video_reader = MatrixFileReader(
                self.input_file,
                buffer_size=self.buffer_size,
                bin_size=self.bin_size
            )
            return self._video_reader

        if self.input_file is None:
            raise ValueError("No input_file provided")

        path = Path(self.input_file)
        suffix = path.suffix.lower()

        if suffix == ".mdf":
            if MDFFileReader is None:
                raise RuntimeError("MDFFileReader not available")
            self._video_reader = MDFFileReader(
                str(path),
                buffer_size=self.buffer_size,
                bin_size=self.bin_size
            )
        elif suffix in (".h5", ".hdf5", ".hdf"):
            if HDF5FileReader is None:
                raise RuntimeError("HDF5FileReader not available")
            self._video_reader = HDF5FileReader(
                str(path),
                buffer_size=self.buffer_size,
                bin_size=self.bin_size
            )
        elif suffix in (".tif", ".tiff"):
            if TIFFStackReader is None:
                raise RuntimeError("TIFFStackReader not available")
            self._video_reader = TIFFStackReader(
                str(path),
                buffer_size=self.buffer_size,
                bin_size=self.bin_size
            )
        else:
            raise ValueError(f"Unsupported input format: {suffix}")

        return self._video_reader

    def get_video_writer(self) -> _VideoWriterBase:
        """Get or create video writer."""
        if self._video_writer is not None:
            return self._video_writer

        # Determine filename
        if self.output_file_name:
            base = Path(self.output_file_name)
        else:
            if self.naming_convention == NamingConvention.DEFAULT:
                base = self.output_path / "compensated"
            else:
                reader = self.get_video_reader()
                base_name = Path(getattr(reader, "file_name", "output")).stem
                base = self.output_path / f"{base_name}_compensated"

        # Create writer based on format
        if self.output_format == OutputFormat.HDF5:
            filename = base.with_suffix(".h5")
            if HDF5FileWriter is None:
                raise RuntimeError("HDF5FileWriter not available")
            self._video_writer = HDF5FileWriter(str(filename))
        elif self.output_format in (OutputFormat.TIFF, OutputFormat.SUITE2P_TIFF):
            filename = base.with_suffix(".tif")
            if TIFFStackWriter is None:
                raise RuntimeError("TIFFStackWriter not available")
            self._video_writer = TIFFStackWriter(str(filename))
        else:
            raise NotImplementedError(f"Writer for {self.output_format} not implemented")

        return self._video_writer

    def get_reference_frame(self, video_reader: Optional[_VideoReaderBase] = None) -> Union[
        np.ndarray, List[np.ndarray]]:
        """Get reference frame(s), with optional preregistration."""
        if self.n_references > 1:
            warnings.warn("Multi-reference mode not fully implemented")
            ref = self.get_reference_frame(video_reader)
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
            frames = video_reader.read_frames(self.reference_frames)  # (H,W,C,T)

            if frames.ndim != 4:
                if frames.ndim == 3:
                    return frames  # Single frame
                raise ValueError("read_frames must return (H,W,C) or (H,W,C,T)")

            # Single frame
            if frames.shape[3] == 1:
                return frames[:, :, :, 0]

            n_channels = frames.shape[2]

            # Build weight array
            weight_2d = np.zeros((frames.shape[0], frames.shape[1], n_channels))
            for c in range(n_channels):
                weight_2d[:, :, c] = self.get_weight_at(c, n_channels)

            if not self.verbose:
                print("Preregistering reference frames...")

            # Preprocess with extra smoothing for preregistration
            if gaussian_filter is not None:
                frames_smooth = np.zeros_like(frames)
                for c in range(n_channels):
                    sig = self.get_sigma_at(c) + np.array([1, 1, 0.5])
                    frames_smooth[:, :, c, :] = gaussian_filter(
                        frames[:, :, c, :],
                        sigma=tuple(sig),
                        mode='reflect'
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

            # Compensate if pyflowreg available
            try:
                from pyflowreg import compensate_sequence

                # Use stronger regularization for preregistration
                alpha_prereg = tuple(a + 2.0 for a in self.alpha) if isinstance(self.alpha, tuple) else self.alpha + 2.0

                compensated = compensate_sequence(
                    frames_norm, ref_mean,
                    weight=weight_2d,
                    alpha=alpha_prereg,
                    levels=self.levels,
                    min_level=self.effective_min_level,
                    eta=self.eta,
                    update_lag=self.update_lag,
                    iterations=self.iterations,
                    a_smooth=self.a_smooth,
                    a_data=self.a_data,
                    constancy_assumption=self.constancy_assumption.value
                )
                reference = np.mean(compensated, axis=3)
            except ImportError:
                # Fallback to simple mean
                reference = ref_mean

            if not self.verbose:
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
            exclude={"preproc_funct", "_video_reader", "_video_writer", "_quality_setting_old", "_datatype"}
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
            "constancy_assumption": self.constancy_assumption.value
        }

    def __repr__(self) -> str:
        return (f"OFOptions(quality={self.quality_setting.value}, alpha={self.alpha}, "
                f"levels={self.levels}, min_level={self.effective_min_level})")


# Convenience functions
def compensate_inplace(
        frames: np.ndarray,
        reference: np.ndarray,
        options: Optional[OFOptions] = None,
        **kwargs
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
        weight=[0.6, 0.4]
    )

    print(opts)
    print("Effective min_level:", opts.effective_min_level)

    # Test save/load
    out_path = Path("test_options.json")
    opts.save_options(out_path)
    loaded = OFOptions.load_options(out_path)
    print("Load/save test passed")
