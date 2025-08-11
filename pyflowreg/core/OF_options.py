"""
Optical Flow Options Configuration Module (Python)
-------------------------------------------------

Python port of MATLAB `OF_options` using Pydantic v2 for validation/IO
with parity tweaks for MATLAB interop (headered JSON, min_level/custom
preset semantics, preregistration smoothing/normalization, and writer
extensions).
"""
from __future__ import annotations

import json
import warnings
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import tifffile
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

# Optional heavy deps: only imported when needed
try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None  # type: ignore

# Optional IO backends (provide stubs in dev)
try:  # pyflowreg IO types
    from pyflowreg.util.io.hdf5 import HDF5FileReader, HDF5FileWriter
    from pyflowreg.util.io.mdf import MDFFileReader
    from pyflowreg.util.io._base import VideoReader as _VideoReaderBase, VideoWriter as _VideoWriterBase
    from pyflowreg.util.io.tiff import TIFFStackReader, TIFFStackWriter
except Exception:  # pragma: no cover
    _VideoReaderBase = object  # type: ignore
    _VideoWriterBase = object  # type: ignore
    HDF5FileReader = None  # type: ignore
    HDF5FileWriter = None  # type: ignore
    MDFFileReader = None  # type: ignore
    TIFFStackReader = None  # type: ignore
    TIFFStackWriter = None  # type: ignore


# -----------------------
# Enums (keep values stable)
# -----------------------
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
    GRADIENT = "gc"  # gradient constancy, value must be 'gc' for downstream


class NamingConvention(str, Enum):
    DEFAULT = "default"
    BATCH = "batch"


# -----------------------------------
# OFOptions model (Pydantic v2)
# -----------------------------------
class OFOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, extra="forbid", )

    # I/O
    input_file: Optional[Union[str, Path, np.ndarray, _VideoReaderBase]] = Field(None,
        description="Path/ndarray/VideoReader for input")
    output_path: Path = Field(Path("results"), description="Output directory")
    output_format: OutputFormat = Field(OutputFormat.MAT, description="Output format")
    output_file_name: Optional[str] = Field(None, description="Custom output filename")
    channel_idx: Optional[List[int]] = Field(None, description="Channel indices to process")

    # Flow params
    alpha: Union[float, Tuple[float, float]] = Field(1.5, gt=0, description="Reg. strength (scalar or 2-tuple)")
    weight: Union[List[float], np.ndarray] = Field([0.5, 0.5], description="Channel weights")
    levels: StrictInt = Field(100, ge=1, description="# pyramid levels")
    min_level: StrictInt = Field(-1, ge=-1, description="Min pyramid level; -1 = from preset")
    quality_setting: QualitySetting = Field(QualitySetting.QUALITY, description="Preset for quality/speed")
    eta: float = Field(0.8, gt=0, le=1, description="Downsample factor per level")
    update_lag: StrictInt = Field(5, ge=1, description="Update lag for non-linear diffusion")
    iterations: StrictInt = Field(50, ge=1, description="Iterations per level")
    a_smooth: float = Field(1.0, ge=0, description="Smoothness diffusion param")
    a_data: float = Field(0.45, gt=0, le=1, description="Data-term diffusion param")

    # Preproc
    sigma: Union[List[float], np.ndarray] = Field([[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]],
        description="Gaussian [sx, sy, st] per-channel")
    bin_size: StrictInt = Field(1, ge=1, description="Spatial binning factor")
    buffer_size: StrictInt = Field(400, ge=1, description="Frame buffer size")

    # Reference
    reference_frames: Union[List[int], str, Path, np.ndarray] = Field(list(range(50, 500)),
        description="Indices, path, or ndarray for reference")
    update_reference: bool = Field(False, description="Update reference during processing")
    n_references: StrictInt = Field(1, ge=1, description="# of references (multi-ref mode)")
    min_frames_per_reference: StrictInt = Field(20, ge=1, description="Min frames per reference cluster")

    # Misc processing options
    verbose: bool = Field(False, description="Verbose logging")
    save_meta_info: bool = Field(True, description="Save meta info")
    save_w: bool = Field(False, description="Save displacement fields")
    save_valid_mask: bool = Field(False, description="Save valid masks")
    save_valid_idx: bool = Field(False, description="Save valid frame indices")
    output_typename: Optional[str] = Field("double", description="Output dtype tag")
    channel_normalization: ChannelNormalization = Field(ChannelNormalization.JOINT, description="Normalization mode")
    interpolation_method: InterpolationMethod = Field(InterpolationMethod.CUBIC, description="Warp interpolation")
    cc_initialization: bool = Field(False, description="Cross-correlation init")
    update_initialization_w: bool = Field(True, description="Propagate flow init across batches")
    naming_convention: NamingConvention = Field(NamingConvention.DEFAULT, description="Output filename style")
    constancy_assumption: ConstancyAssumption = Field(ConstancyAssumption.GRADIENT, description="Constancy assumption")

    # Non-serializable/pre-runtime
    preproc_funct: Optional[Callable] = Field(None, exclude=True)

    # Internal state
    _video_reader: Optional[_VideoReaderBase] = Field(default=None, exclude=True)
    _video_writer: Optional[_VideoWriterBase] = Field(default=None, exclude=True)
    _quality_setting_old: QualitySetting = Field(default=QualitySetting.QUALITY, exclude=True)

    # -----------------------
    # Validators / derivations
    # -----------------------
    @model_validator(mode="after")
    def _coerce_and_sync(self) -> "OFOptions":
        """Normalize fields and keep MATLAB parity for presets."""
        # output_path -> Path
        if not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)

        # alpha -> (ax, ay)
        if isinstance(self.alpha, (int, float)):
            self.alpha = (float(self.alpha), float(self.alpha))
        elif isinstance(self.alpha, (list, tuple)):
            if len(self.alpha) == 1:
                self.alpha = (float(self.alpha[0]), float(self.alpha[0]))
            elif len(self.alpha) == 2:
                self.alpha = (float(self.alpha[0]), float(self.alpha[1]))
            else:
                raise ValueError("alpha must be scalar or 2-element tuple")
        else:
            raise ValueError("alpha must be numeric or tuple")

        # weight: normalize 1D vectors
        if isinstance(self.weight, np.ndarray):
            if self.weight.ndim == 1:
                s = float(self.weight.sum())
                if s <= 0:
                    raise ValueError("weight sum must be > 0")
                self.weight = (self.weight / s).tolist()
        elif isinstance(self.weight, (list, tuple)):
            arr = np.asarray(self.weight, dtype=float)
            if arr.ndim == 1:
                s = float(arr.sum())
                if s <= 0:
                    raise ValueError("weight sum must be > 0")
                self.weight = (arr / s).tolist()
            else:  # keep multi-d as-is
                self.weight = self.weight

        # sigma: accept [3] or (n,3)
        sig = np.asarray(self.sigma, dtype=float)
        if sig.ndim == 1 and sig.size == 3:
            self.sigma = sig.reshape(1, 3).tolist()
        elif sig.ndim == 2 and sig.shape[1] == 3:
            self.sigma = sig.tolist()
        else:
            raise ValueError("sigma must be [sx,sy,st] or (n_channels,3)")

        # MATLAB preset logic: remember last non-custom; flip to custom if min_level >= 0
        if self.quality_setting != QualitySetting.CUSTOM:
            self._quality_setting_old = self.quality_setting
        if self.min_level >= 0:
            self.quality_setting = QualitySetting.CUSTOM
        elif self.min_level == -1 and self.quality_setting == QualitySetting.CUSTOM:
            # restore previous preset
            self.quality_setting = self._quality_setting_old or QualitySetting.QUALITY

        return self

    # -----------------------
    # Derived properties
    # -----------------------
    @property
    def effective_min_level(self) -> int:
        if self.min_level >= 0:
            return int(self.min_level)
        mapping = {QualitySetting.QUALITY: 0, QualitySetting.BALANCED: 4, QualitySetting.FAST: 6,
            QualitySetting.CUSTOM: max(int(self.min_level), 0), }
        return mapping.get(self.quality_setting, 0)

    # -----------------------
    # Helpers matching MATLAB behavior
    # -----------------------
    def _normalize_frames(self, arr: np.ndarray) -> np.ndarray:
        """Normalize frames to [0,1] joint or per-channel.

        arr shape: (H, W, C, T)
        """
        eps = 1e-8
        if self.channel_normalization == ChannelNormalization.SEPARATE:
            # min/max over H,W,T for each channel
            mn = arr.min(axis=(0, 1, 3), keepdims=True)
            mx = arr.max(axis=(0, 1, 3), keepdims=True)
            return (arr - mn) / (mx - mn + eps)
        # joint
        a = float(arr.min())
        b = float(arr.max())
        return (arr - a) / (b - a + eps)

    def _smooth_spatiotemporal(self, arr: np.ndarray) -> np.ndarray:
        """Approximate MATLAB imgaussfilt3 over (H,W,T) per channel.

        Applies sigma ~ [1, 1, 0.5]. Requires SciPy.
        expected shape: (H, W, C, T)
        """
        if gaussian_filter is None:
            # Silently skip if SciPy isn't available
            return arr
        H, W, C, T = arr.shape
        out = np.empty_like(arr)
        for c in range(C):
            out[:, :, c, :] = gaussian_filter(arr[:, :, c, :], sigma=(1.0, 1.0, 0.5))
        return out

    def _truncate_and_rebalance_weights(self, n_channels: int) -> None:
        # If weight is a vector with more entries than channels, truncate and renormalize (MATLAB parity)
        w = np.asarray(self.weight, dtype=float)
        if w.ndim == 1 and w.size > n_channels:
            w = w[:n_channels]
            s = float(w.sum())
            if s <= 0:
                w = np.ones(n_channels, dtype=float) / n_channels
            else:
                w = w / s
            self.weight = w.tolist()

    # -----------------------
    # Readers / writers
    # -----------------------
    def get_video_reader(self) -> _VideoReaderBase:
        if self._video_reader is not None:
            return self._video_reader
        if isinstance(self.input_file, _VideoReaderBase):
            self._video_reader = self.input_file
            return self._video_reader
        if isinstance(self.input_file, np.ndarray):
            # Lazy import to avoid hard dep if not needed
            try:
                from pyflowreg.util.io.matrix import MatrixFileReader  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("MatrixFileReader not available") from e
            self._video_reader = MatrixFileReader(self.input_file, buffer_size=int(self.buffer_size),
                bin_size=int(self.bin_size))
            return self._video_reader
        if self.input_file is None:
            raise ValueError("No input_file provided")

        path = Path(self.input_file)
        suf = path.suffix.lower()
        if suf == ".mdf":
            if MDFFileReader is None:
                raise RuntimeError("MDFFileReader backend not available")
            self._video_reader = MDFFileReader(str(path), buffer_size=int(self.buffer_size),
                                               bin_size=int(self.bin_size))
        elif suf in (".h5", ".hdf5", ".hdf"):
            if HDF5FileReader is None:
                raise RuntimeError("HDF5FileReader backend not available")
            self._video_reader = HDF5FileReader(str(path), buffer_size=int(self.buffer_size),
                                                bin_size=int(self.bin_size))
        elif suf in (".tif", ".tiff"):
            if TIFFStackReader is None:
                raise RuntimeError("TIFFStackReader backend not available")
            self._video_reader = TIFFStackReader(str(path), buffer_size=int(self.buffer_size),
                                                 bin_size=int(self.bin_size))
        else:
            raise ValueError(f"Unsupported input format: {suf}")
        return self._video_reader

    def get_video_writer(self) -> _VideoWriterBase:
        if self._video_writer is not None:
            return self._video_writer

        # Base filename (without extension)
        if self.output_file_name:
            base = Path(self.output_file_name)
        else:
            if self.naming_convention == NamingConvention.DEFAULT:
                base = self.output_path / "compensated"
            else:
                rdr = self.get_video_reader()
                base_name = Path(getattr(rdr, "file_name", "output")).stem
                base = self.output_path / f"{base_name}_compensated"

        # Enforce canonical extensions (.h5, .tif)
        of = self.output_format
        if of == OutputFormat.HDF5:
            filename = base.with_suffix(".h5")
            if HDF5FileWriter is None:
                raise RuntimeError("HDF5FileWriter backend not available")
            self._video_writer = HDF5FileWriter(str(filename))
        elif of in (OutputFormat.TIFF, OutputFormat.SUITE2P_TIFF):
            filename = base.with_suffix(".tif")
            if TIFFStackWriter is None:
                raise RuntimeError("TIFFStackWriter backend not available")
            self._video_writer = TIFFStackWriter(str(filename))
        else:
            raise NotImplementedError(f"Writer for {of} not implemented in this module")

        return self._video_writer

    # -----------------------
    # Reference generation (parity with MATLAB preregistration)
    # -----------------------
    def get_multi_reference_frames(self, video_reader: Optional[_VideoReaderBase] = None) -> List[np.ndarray]:
        if video_reader is None:
            raise ValueError("Video reader required for multi-reference mode")
        warnings.warn("Multi-reference mode not fully implemented; returning repeated single reference")
        ref = self.get_reference_frame(video_reader)
        return [ref] * int(self.n_references)

    def get_reference_frame(self, video_reader: Optional[_VideoReaderBase] = None) -> Union[
        np.ndarray, List[np.ndarray]]:
        if self.n_references > 1:
            return self.get_multi_reference_frames(video_reader)

        # Direct ndarray or path
        if isinstance(self.reference_frames, np.ndarray):
            return self.reference_frames
        if isinstance(self.reference_frames, (str, Path)):
            p = Path(self.reference_frames)
            if p.suffix.lower() in (".tif", ".tiff"):
                return tifffile.imread(str(p))
            try:
                import imageio.v3 as iio  # type: ignore
                return iio.imread(str(p))
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Unable to read reference image: {p}") from e

        # List of frame indices -> preregister
        if isinstance(self.reference_frames, list):
            if video_reader is None:
                raise ValueError("Video reader required to fetch reference frames by index")
            frames = video_reader.read_frames(self.reference_frames)  # expected (H,W,C,T)
            if frames.ndim != 4:
                raise ValueError("read_frames must return (H,W,C,T)")

            # weight truncation to channel count
            C = frames.shape[2]
            self._truncate_and_rebalance_weights(C)

            # Smoothing and normalization (MATLAB parity)
            f_smooth = self._smooth_spatiotemporal(frames)
            f_norm = self._normalize_frames(f_smooth)

            if self.verbose:
                print("Preregistering reference frames...")

            ref_mean = np.mean(f_norm, axis=3)

            # Call into pyflowreg batch compensator if available
            try:
                # Prefer batch path to avoid Python-loop overhead
                from pyflowreg import compensate_sequence  # type: ignore
                compensated = compensate_sequence(f_norm, ref_mean, weight=self.weight,
                    alpha=(self.alpha[0] + 2.0, self.alpha[1] + 2.0), levels=int(self.levels),
                    min_level=int(self.effective_min_level), eta=float(self.eta), update_lag=int(self.update_lag),
                    iterations=int(self.iterations), a_smooth=float(self.a_smooth), a_data=float(self.a_data),
                    constancy_assumption=self.constancy_assumption.value, )
                out = np.mean(compensated, axis=3)
            except Exception:
                # Fallback: simple average if compensator not available
                out = ref_mean

            if self.verbose:
                print("Finished pre-registration of the reference frames.")
            return out

        # Fallback
        return np.asarray(self.reference_frames)

    # -----------------------
    # Save/load with MATLAB header parity
    # -----------------------
    def save_options(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """Save options to JSON with MATLAB-compatible header line."""
        path = Path(filepath) if filepath is not None else self.output_path / "options.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(exclude={"preproc_funct", "_video_reader", "_video_writer", "_quality_setting_old"})
        # Cast non-JSON types
        for k, v in list(data.items()):
            if isinstance(v, Path):
                data[k] = str(v)
            elif isinstance(v, np.ndarray):
                data[k] = v.tolist()

        # Store reference array separately to avoid huge JSON
        if isinstance(self.reference_frames, np.ndarray):
            ref_path = path.parent / "reference_frames.tif"
            tifffile.imwrite(str(ref_path), self.reference_frames)
            data["reference_frames"] = str(ref_path)

        with path.open("w", encoding="utf-8") as f:
            f.write(f"Compensation options {date.today().isoformat()}\n\n")
            json.dump(data, f, indent=2)
        if self.verbose:
            print(f"Options saved to {path}")

    @classmethod
    def load_options(cls, filepath: Union[str, Path]) -> "OFOptions":
        """Load options from JSON written by MATLAB or this module (skips header)."""
        p = Path(filepath)
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        # MATLAB writes a header line then a blank line, then JSON
        # Find first line that starts with '{'
        json_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break
        payload = "".join(lines[json_start:])
        data = json.loads(payload)

        # Rehydrate reference if file path provided
        ref = data.get("reference_frames")
        if isinstance(ref, str):
            rpath = Path(ref)
            if rpath.exists() and rpath.suffix.lower() in (".tif", ".tiff"):
                data["reference_frames"] = tifffile.imread(str(rpath))
        return cls(**data)

    def to_dict(self) -> dict:
        """Params for low-level optical flow kernels (Python/C++ layer)."""
        return {"alpha": (float(self.alpha[0]), float(self.alpha[1])), "weight": self.weight,
            "levels": int(self.levels), "min_level": int(self.effective_min_level), "eta": float(self.eta),
            "iterations": int(self.iterations), "update_lag": int(self.update_lag), "a_data": float(self.a_data),
            "a_smooth": float(self.a_smooth), "constancy_assumption": self.constancy_assumption.value, }

    def __repr__(self) -> str:  # pragma: no cover
        return (f"OFOptions(quality={self.quality_setting.value}, alpha={self.alpha}, "
                f"levels={self.levels}, min_level={self.effective_min_level})")


# -----------------------
# Convenience functions
# -----------------------
def compensate_inplace(frames: np.ndarray, reference: np.ndarray, options: Optional[OFOptions] = None, **kwargs, ) -> \
Tuple[np.ndarray, np.ndarray]:
    """Compensate frames against reference and return (compensated, displacements).

    Shapes:
      frames: (H,W,C,T) or (H,W,T)
      reference: (H,W,C) or (H,W)
    """
    if options is None:
        options = OFOptions(**kwargs)
    else:
        # copy-and-update (immutability-style)
        options = options.model_copy(update=kwargs)

    # Ensure 4D frames and 3D reference
    if frames.ndim == 3:
        frames = frames[:, :, np.newaxis, :]
    if reference.ndim == 2:
        reference = reference[:, :, np.newaxis]

    params = options.to_dict()

    # Prefer batch displacement if available
    try:
        from pyflowreg import get_displacement, compensate_sequence_uv  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyflowreg core functions not available") from e

    # Compute displacements per time slice (keeping parity with existing APIs)
    Ts = frames.shape[3]
    disp_list = []
    for t in range(Ts):
        w = get_displacement(reference, frames[:, :, :, t], **params)
        disp_list.append(w)
    displacements = np.stack(disp_list, axis=3)

    compensated = compensate_sequence_uv(frames, reference, displacements)
    return compensated, displacements


# -----------------------
# MCP schema helper
# -----------------------
def get_mcp_schema() -> dict:
    return OFOptions.model_json_schema()


if __name__ == "__main__":  # quick self-check paths
    opts = OFOptions(input_file="test.h5", output_path=Path("./results"), quality_setting=QualitySetting.BALANCED,
        alpha=2.0, weight=[0.6, 0.4], )
    print(opts)
    print("Effective min_level:", opts.effective_min_level)
    # Save/load round-trip (MATLAB header compatible)
    out_path = Path("test_options.json")
    opts.save_options(out_path)
    _ = OFOptions.load_options(out_path)
