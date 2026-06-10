"""
Configuration model for z-alignment workflows.

The z-align pipeline mirrors the MATLAB prototypes with three stages:
1) Build or load a reference volume.
2) Estimate per-pixel z-shifts and optionally write a z-corrected signal.
3) Optionally simulate a z-shift-only recording from volume + z-shifts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


_STAGE1_PROTECTED_OF_FIELDS = {
    "input_file",
    "output_path",
    "output_format",
    "output_file_name",
    "naming_convention",
    "reference_frames",
}

_RECORDING_PREALIGN_PROTECTED_OF_FIELDS = {
    "input_file",
    "output_path",
    "output_format",
    "output_file_name",
    "naming_convention",
}


class ZAlignConfig(BaseModel):
    """
    Configuration for z-shift alignment and correction.

    Drives the three-stage z-align pipeline in
    ``pyflowreg.z_align.pipeline``: Stage 1 builds or loads a compensated
    reference volume, Stage 2 estimates per-pixel z-shifts patch-wise and
    optionally writes a z-corrected recording, and Stage 3 optionally
    simulates a z-shift-only recording from volume and z-shifts. Fields are
    grouped below as core paths, reference building, output locations,
    control flags, Stage 1 (volume build), and Stage 2 (patch-based z
    estimation).

    Parameters
    ----------
    root : Path
        Base directory; must exist and be a directory. Relative input paths
        are resolved against it.
    input_file : Path
        Recording to estimate z-shifts for (Stage 2 input), relative to
        ``root`` unless absolute.
    volume_input_file : Path, optional, default=None
        Raw reference stack compensated in Stage 1 to build the reference
        volume. Required when ``reference_volume`` is not provided.
    reference_volume : Path, optional, default=None
        Existing compensated reference volume. When set, Stage 1 skips the
        volume build and uses this file directly.
    reference_source_file : Path, optional, default=None
        Recording whose leading frames are averaged into the reference image
        passed to Stage 1 compensation. For recording prealignment, falls
        back to ``input_file`` when None.
    reference_source_frames : int, default=2000
        Maximum number of (binned) frames averaged into the reference image.
    reference_source_buffer_size : int, default=10
        Reader batch size when reading the reference source.
    reference_source_bin_size : int, default=20
        Temporal binning applied when reading the reference source.
    output_root : Path, default="z_align_outputs"
        Directory for all z-align outputs and ``status.json``; resolved
        relative to ``root`` unless absolute.
    volume_output_dir : Path, default="aligned_stack"
        Directory (under ``output_root``) where Stage 1 writes the
        compensated reference volume.
    recording_prealigned_output_dir : Path, default="prealigned_recording"
        Directory (under ``output_root``) for the optional prealigned
        recording (``compensated.HDF5``).
    z_shift_file : Path, default="z_shift.HDF5"
        Stage 2 per-pixel z-shift output (under ``output_root``); stored as
        1-based slice coordinates for MATLAB parity. Must have an HDF5
        extension (validated).
    corrected_output_file : Path, default="compensated_shift_corrected.tif"
        Stage 2 z-corrected recording (under ``output_root``).
    simulated_output_file : Path, default="simulated_from_z.tif"
        Stage 3 simulated recording (under ``output_root``).
    resume : bool, default=True
        Reuse completed stage outputs recorded in ``status.json`` instead of
        recomputing them.
    prealign_stack : bool, default=True
        Motion-compensate the raw reference stack in Stage 1. If False, the
        raw ``volume_input_file`` is used as the volume directly.
    prealign_recording : bool, default=False
        Motion-compensate the input recording before Stage 2 z estimation.
    write_corrected : bool, default=True
        Write the z-corrected recording during Stage 2.
    write_simulated : bool, default=True
        Run Stage 3 and write the simulated z-shift-only recording.
    stage1_alpha : float, default=5.0
        OFOptions ``alpha`` for Stage 1 compensation (also used for
        recording prealignment).
    stage1_quality_setting : str, default="quality"
        OFOptions ``quality_setting`` for Stage 1 (also used for recording
        prealignment).
    stage1_buffer_size : int, default=500
        Reader buffer size for Stage 1 compensation (superseded by
        ``stack_scans_per_slice`` when set).
    stage1_bin_size : int, default=1
        Temporal bin size for Stage 1 compensation.
    stage1_update_reference : bool, default=True
        OFOptions ``update_reference`` for Stage 1; forced True when
        ``stack_scans_per_slice`` is set.
    stack_scans_per_slice : int, optional, default=None
        Number of repeated scans per z slice. When set, used as the Stage 1
        buffer size and as the bin size for reading the reference stack as z
        slices.
    flow_backend : str, default="flowreg"
        Optical-flow backend passed to OFOptions for Stage 1 and
        prealignment.
    backend_params : dict, default={}
        Backend-specific parameters passed to OFOptions.
    stage1_flow_options : dict or Path, optional, default=None
        Extra OFOptions overrides for Stage 1, either inline as a mapping or
        as a path (relative to ``root``) to a saved OF_options JSON file.
        Workflow-owned I/O routing fields and ``reference_frames`` are
        stripped.
    recording_prealign_flow_options : dict or Path, optional, default=None
        Same as ``stage1_flow_options`` but for the optional recording
        prealignment; workflow-owned I/O routing fields are stripped
        (``reference_frames`` may be overridden).
    input_buffer_size : int, default=50
        Reader batch size for the Stage 2 input recording (also used when
        reading z-shifts in Stage 3).
    input_bin_size : int, default=1
        Temporal binning for the input recording. When prealignment is
        enabled, binning happens during prealignment and Stage 2 reads the
        prealigned recording unbinned.
    volume_buffer_size : int, default=500
        Reader batch size when loading the reference volume.
    volume_bin_size : int, default=1
        Temporal binning when reading the reference volume as z slices
        (superseded by ``stack_scans_per_slice``).
    win_half : int, default=10
        Half-width of the z search window around the anchor slice; Stage 2
        scores candidates in ``[anchor_z - win_half, anchor_z + win_half]``
        clipped to the volume.
    patch_size : int, default=128
        Side length of the square spatial patches scored against z
        candidates.
    overlap : float, default=0.75
        Fractional overlap between neighboring patches; the patch stride is
        ``round(patch_size * (1 - overlap))``. Must satisfy
        ``0 <= overlap < 1`` (validated).
    spatial_sigma : float, default=1.5
        Gaussian sigma for spatial smoothing before gradient computation
        (applied to volume slices and input frames).
    temporal_sigma : float, default=1.5
        Temporal Gaussian sigma applied to input frames before gradient
        computation.
    z_smooth_sigma_spatial : float, default=5.0
        Spatial Gaussian sigma for smoothing the estimated z-shift maps.
    z_smooth_sigma_temporal : float, default=1.5
        Temporal Gaussian sigma for smoothing the estimated z-shift maps.
    parabolic_tau_scale : float, default=1e-3
        Scale of the curvature threshold for sub-voxel parabolic refinement;
        near-flat score parabolas keep the integer z estimate.
    output_dtype : str, default="uint16"
        NumPy dtype name for corrected and simulated outputs; integer
        outputs are clipped and rounded before casting.
    n_jobs : int, default=-1
        Worker count for Stage 2 patch scoring; -1 uses all CPU cores.
    parallelization : str, default="sequential"
        Patch-scoring execution mode, "sequential" or "threading".

    Notes
    -----
    Instances can be loaded from configuration files via the ``from_toml``,
    ``from_yaml``, and ``from_file`` (extension auto-detection) classmethods.

    Validation as implemented by the field validators: string paths are
    coerced to ``Path``; ``root`` must exist and be a directory; the listed
    integer fields must be >= 1; ``stage1_alpha``, the sigma fields, and
    ``parabolic_tau_scale`` must be > 0; ``overlap`` must lie in [0, 1);
    ``z_shift_file`` must have a .h5/.hdf5/.hdf extension; ``output_dtype``
    must be a valid NumPy dtype name; ``n_jobs`` must be -1 or >= 1;
    ``parallelization`` is lower-cased and must be "sequential" or
    "threading"; the flow-options fields are normalized to a mapping or
    ``Path`` (blank strings become None). Unknown keys are rejected
    (``extra="forbid"``).

    See Also
    --------
    pyflowreg.session.config.SessionConfig : Analogous configuration for
        multi-recording session processing.
    """

    model_config = ConfigDict(extra="forbid")

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
    recording_prealigned_output_dir: Path = Field(default=Path("prealigned_recording"))
    z_shift_file: Path = Field(default=Path("z_shift.HDF5"))
    corrected_output_file: Path = Field(default=Path("compensated_shift_corrected.tif"))
    simulated_output_file: Path = Field(default=Path("simulated_from_z.tif"))

    # Control flags
    resume: bool = True
    prealign_stack: bool = True
    prealign_recording: bool = False
    write_corrected: bool = True
    write_simulated: bool = True

    # Stage 1 (volume build via compensate_recording / OFOptions)
    stage1_alpha: float = 5.0
    stage1_quality_setting: str = "quality"
    stage1_buffer_size: int = 500
    stage1_bin_size: int = 1
    stage1_update_reference: bool = True
    stack_scans_per_slice: Optional[int] = None
    flow_backend: str = "flowreg"
    backend_params: Dict[str, Any] = Field(default_factory=dict)
    stage1_flow_options: Optional[Union[Dict[str, Any], Path]] = None
    recording_prealign_flow_options: Optional[Union[Dict[str, Any], Path]] = None

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
    n_jobs: int = -1
    parallelization: str = "sequential"

    @field_validator(
        "root",
        "input_file",
        "volume_input_file",
        "reference_volume",
        "reference_source_file",
        "output_root",
        "volume_output_dir",
        "recording_prealigned_output_dir",
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

    @field_validator(
        "stage1_flow_options",
        "recording_prealign_flow_options",
        mode="before",
    )
    @classmethod
    def _normalize_flow_options(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            stripped = v.strip()
            return Path(stripped) if stripped else None
        raise TypeError("Flow options must be a mapping or path")

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
        "stack_scans_per_slice",
        "input_buffer_size",
        "input_bin_size",
        "volume_buffer_size",
        "volume_bin_size",
        "win_half",
        "patch_size",
    )
    @classmethod
    def _validate_positive_int(cls, v: Optional[int]):
        if v is None:
            return v
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

    @field_validator("z_shift_file")
    @classmethod
    def _validate_z_shift_file(cls, v: Path):
        # Stage 2 always writes this file with the HDF5 writer, while stage 3
        # re-opens it through the extension-dispatched reader factory, so a
        # non-HDF5 extension would break read-back after a successful run.
        if v.suffix.lower() not in {".h5", ".hdf5", ".hdf"}:
            raise ValueError(
                "z_shift_file must have an HDF5 extension (.h5/.hdf5/.hdf), "
                f"got: {v.name}"
            )
        return v

    @field_validator("output_dtype")
    @classmethod
    def _validate_output_dtype(cls, v: str):
        try:
            np.dtype(v)
        except TypeError as exc:
            raise ValueError(f"Invalid output_dtype: {v}") from exc
        return v

    @field_validator("n_jobs")
    @classmethod
    def _validate_n_jobs(cls, v: int):
        if v == -1:
            return v
        if v < 1:
            raise ValueError("n_jobs must be -1 or >= 1")
        return v

    @field_validator("parallelization", mode="before")
    @classmethod
    def _validate_parallelization(cls, v):
        if not isinstance(v, str):
            raise TypeError("parallelization must be a string")
        value = v.strip().lower()
        allowed = {"sequential", "threading"}
        if value not in allowed:
            raise ValueError(f"parallelization must be one of {sorted(allowed)}")
        return value

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

    def resolve_recording_prealigned_output_dir(self) -> Path:
        return self._resolve_from_output_root(self.recording_prealigned_output_dir)

    def resolve_recording_prealigned_file(self) -> Path:
        return self.resolve_recording_prealigned_output_dir() / "compensated.HDF5"

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

    def effective_volume_bin_size(self) -> int:
        """Return the bin size used when reading the reference stack as z slices."""
        return self.stack_scans_per_slice or self.volume_bin_size

    def _resolve_flow_options_path(self, path: Path) -> Path:
        options_path = path.expanduser()
        if not options_path.is_absolute():
            options_path = self.root / options_path
        return options_path

    def _get_flow_option_overrides(
        self,
        option_source: Optional[Union[Dict[str, Any], Path]],
        *,
        protected_fields: set[str],
        label: str,
    ) -> Dict[str, Any]:
        """
        Return OFOptions overrides with workflow-owned fields removed.

        The config supports inline dict values or paths to saved OF_options
        JSON. For both sources, only fields the user explicitly provided are
        forwarded; fields a loaded file does not mention keep their
        workflow-computed values instead of leaking OFOptions defaults.
        """
        if option_source is None:
            return {}

        if isinstance(option_source, dict):
            return {
                key: value
                for key, value in option_source.items()
                if key not in protected_fields
            }

        options_path = self._resolve_flow_options_path(option_source)

        if not options_path.exists():
            raise ValueError(f"{label} flow options file not found: {options_path}")

        from pyflowreg.motion_correction.OF_options import OFOptions

        options = OFOptions.load_options(options_path)
        allowed_fields = set(OFOptions.model_fields.keys())
        allowed_fields.difference_update(protected_fields)

        return {
            key: value
            for key, value in options.model_dump(exclude_unset=True).items()
            if key in allowed_fields
        }

    def get_stage1_overrides(self) -> Dict[str, Any]:
        """Return OFOptions overrides for stage 1."""
        return self._get_flow_option_overrides(
            self.stage1_flow_options,
            protected_fields=_STAGE1_PROTECTED_OF_FIELDS,
            label="Stage-1",
        )

    def get_recording_prealign_overrides(self) -> Dict[str, Any]:
        """Return OFOptions overrides for optional recording prealignment."""
        return self._get_flow_option_overrides(
            self.recording_prealign_flow_options,
            protected_fields=_RECORDING_PREALIGN_PROTECTED_OF_FIELDS,
            label="Recording prealignment",
        )

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
