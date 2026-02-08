"""
Configuration model for z-alignment workflows.

The z-align pipeline mirrors the MATLAB prototypes with three stages:
1) Build or load a reference volume.
2) Estimate per-pixel z-shifts and optionally write a z-corrected signal.
3) Optionally simulate a baseline recording from volume + z-shifts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator


class ZAlignConfig(BaseModel):
    """Configuration for z-shift alignment and correction."""

    # Core paths
    root: Path
    input_file: Path
    volume_input_file: Optional[Path] = None
    reference_volume: Optional[Path] = None
    reference_source_file: Optional[Path] = None

    # Stage 1 reference-frame estimation from a source recording
    reference_source_frames: int = 2000
    reference_source_buffer_size: int = 10
    reference_source_bin_size: int = 20

    # Outputs
    output_root: Path = Field(default=Path("z_align_outputs"))
    volume_output_dir: Path = Field(default=Path("aligned_stack"))
    z_shift_file: Path = Field(default=Path("z_shift.HDF5"))
    corrected_output_file: Path = Field(default=Path("compensated_shift_corrected.tif"))
    simulated_output_file: Path = Field(default=Path("simulated_from_z.tif"))

    # Control flags
    resume: bool = True
    write_corrected: bool = True
    write_simulated: bool = True

    # Stage 1 (volume build via compensate_recording / OFOptions)
    stage1_alpha: float = 5.0
    stage1_quality_setting: str = "quality"
    stage1_buffer_size: int = 500
    stage1_bin_size: int = 1
    stage1_update_reference: bool = True
    flow_backend: str = "flowreg"
    backend_params: Dict[str, Any] = Field(default_factory=dict)
    stage1_flow_options: Optional[Union[Dict[str, Any], Path]] = None

    # Stage 2 (patch-based z estimation)
    input_buffer_size: int = 50
    input_bin_size: int = 1
    volume_buffer_size: int = 500
    volume_bin_size: int = 1
    win_half: int = 10
    patch_size: int = 128
    overlap: float = 0.75
    spatial_sigma: float = 1.5
    temporal_sigma: float = 1.5
    z_smooth_sigma_spatial: float = 5.0
    z_smooth_sigma_temporal: float = 1.5
    parabolic_tau_scale: float = 1e-3
    output_dtype: str = "uint16"

    @field_validator(
        "root",
        "input_file",
        "volume_input_file",
        "reference_volume",
        "reference_source_file",
        "output_root",
        "volume_output_dir",
        "z_shift_file",
        "corrected_output_file",
        "simulated_output_file",
        mode="before",
    )
    @classmethod
    def _to_path(cls, v):
        if v is None or isinstance(v, Path):
            return v
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("stage1_flow_options", mode="before")
    @classmethod
    def _normalize_stage1_flow_options(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            stripped = v.strip()
            return Path(stripped) if stripped else None
        raise TypeError("stage1_flow_options must be a mapping or path")

    @field_validator("root")
    @classmethod
    def _validate_root(cls, v: Path):
        if not v.exists():
            raise ValueError(f"Root directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Root path is not a directory: {v}")
        return v

    @field_validator(
        "reference_source_frames",
        "reference_source_buffer_size",
        "reference_source_bin_size",
        "stage1_buffer_size",
        "stage1_bin_size",
        "input_buffer_size",
        "input_bin_size",
        "volume_buffer_size",
        "volume_bin_size",
        "win_half",
        "patch_size",
    )
    @classmethod
    def _validate_positive_int(cls, v: int):
        if v < 1:
            raise ValueError("Value must be >= 1")
        return v

    @field_validator(
        "stage1_alpha",
        "spatial_sigma",
        "temporal_sigma",
        "z_smooth_sigma_spatial",
        "z_smooth_sigma_temporal",
        "parabolic_tau_scale",
    )
    @classmethod
    def _validate_positive_float(cls, v: float):
        if v <= 0:
            raise ValueError("Value must be > 0")
        return v

    @field_validator("overlap")
    @classmethod
    def _validate_overlap(cls, v: float):
        if not (0.0 <= v < 1.0):
            raise ValueError("overlap must satisfy 0 <= overlap < 1")
        return v

    @field_validator("output_dtype")
    @classmethod
    def _validate_output_dtype(cls, v: str):
        try:
            np.dtype(v)
        except TypeError as exc:
            raise ValueError(f"Invalid output_dtype: {v}") from exc
        return v

    def _resolve_from_root(self, path: Path) -> Path:
        p = path.expanduser()
        return p if p.is_absolute() else (self.root / p)

    def _resolve_from_output_root(self, path: Path) -> Path:
        p = path.expanduser()
        return p if p.is_absolute() else (self.resolve_output_root() / p)

    def resolve_output_root(self) -> Path:
        return self._resolve_from_root(self.output_root)

    def resolve_input_file(self) -> Path:
        return self._resolve_from_root(self.input_file)

    def resolve_volume_input_file(self) -> Optional[Path]:
        if self.volume_input_file is None:
            return None
        return self._resolve_from_root(self.volume_input_file)

    def resolve_reference_source_file(self) -> Optional[Path]:
        if self.reference_source_file is None:
            return None
        return self._resolve_from_root(self.reference_source_file)

    def resolve_volume_output_dir(self) -> Path:
        return self._resolve_from_output_root(self.volume_output_dir)

    def resolve_z_shift_file(self) -> Path:
        return self._resolve_from_output_root(self.z_shift_file)

    def resolve_corrected_output_file(self) -> Path:
        return self._resolve_from_output_root(self.corrected_output_file)

    def resolve_simulated_output_file(self) -> Path:
        return self._resolve_from_output_root(self.simulated_output_file)

    def resolve_reference_volume_path(self) -> Path:
        """
        Resolve reference volume path.

        If ``reference_volume`` is provided, use it. Otherwise, return the default
        compensated-volume path under ``volume_output_dir``.
        """
        if self.reference_volume is not None:
            return self._resolve_from_root(self.reference_volume)

        volume_dir = self.resolve_volume_output_dir()
        default_candidates = [
            volume_dir / "compensated.HDF5",
            volume_dir / "compensated.hdf5",
        ]
        for candidate in default_candidates:
            if candidate.exists():
                return candidate
        return default_candidates[0]

    def get_stage1_overrides(self) -> Dict[str, Any]:
        """
        Return OFOptions overrides for stage 1.

        The config supports inline dict values or paths to saved OF_options JSON.
        """
        if self.stage1_flow_options is None:
            return {}

        if isinstance(self.stage1_flow_options, dict):
            return dict(self.stage1_flow_options)

        options_path = self.stage1_flow_options.expanduser()
        if not options_path.is_absolute():
            options_path = self.root / options_path

        if not options_path.exists():
            raise ValueError(f"Stage-1 flow options file not found: {options_path}")

        from pyflowreg.motion_correction.OF_options import OFOptions

        options = OFOptions.load_options(options_path)
        allowed_fields = set(OFOptions.model_fields.keys())
        allowed_fields.discard("input_file")
        allowed_fields.discard("output_path")

        return {
            key: value
            for key, value in options.model_dump().items()
            if key in allowed_fields
        }

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> "ZAlignConfig":
        import sys

        p = Path(path)
        if sys.version_info >= (3, 11):
            import tomllib

            with open(p, "rb") as f:
                data = tomllib.load(f)
        else:
            try:
                import tomli
            except ImportError as exc:
                raise ImportError(
                    "TOML support requires 'tomli' for Python < 3.11."
                ) from exc
            with open(p, "rb") as f:
                data = tomli.load(f)

        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ZAlignConfig":
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML support requires 'pyyaml'. Install with: pip install pyyaml"
            ) from exc

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ZAlignConfig":
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".toml":
            return cls.from_toml(p)
        if suffix in {".yaml", ".yml"}:
            return cls.from_yaml(p)
        raise ValueError(
            f"Unsupported config file format: {suffix}. Use .toml, .yaml, or .yml."
        )
