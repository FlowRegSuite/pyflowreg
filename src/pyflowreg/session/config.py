"""
Session configuration using Pydantic for validation and serialization.

Supports both TOML and YAML configuration files for defining multi-recording
session processing parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class SessionConfig(BaseModel):
    """
    Configuration for multi-recording session processing.

    Parameters
    ----------
    root : Path
        Root directory containing input recordings
    pattern : str, default="*.tif"
        Glob pattern for discovering input files
    center : Optional[str], default=None
        Filename (relative to root) of the "center" recording to align to.
        If None, uses lexicographic middle file.
    output_root : Path, default="compensated_outputs"
        Directory for per-recording compensation outputs
    final_results : Path, default="final_results"
        Directory for final aligned results and session mask
    resume : bool, default=True
        Skip stages that have already completed
    scheduler : Literal["local", "array", "dask"], default="local"
        Execution scheduler:
        - "local": Sequential or parallel on single machine
        - "array": Use HPC job array (SLURM/SGE/PBS)
        - "dask": Use Dask distributed scheduler
    flow_backend : str, default="flowreg"
        Optical flow backend (passed to OFOptions)
    backend_params : dict, default={}
        Additional parameters for flow backend
    flow_options : dict | str | Path, optional
        Per-sequence optical flow overrides. Can be provided inline as a
        dictionary or as a path (absolute or relative to ``root``) pointing to
        an ``OF_options`` JSON file. Path form mirrors MATLAB saved options.
    stage1_quality_setting : Optional[str], default=None
        Quality preset for Stage 1 motion correction ("quality", "balanced", or "fast").
        If None, uses OFOptions default (usually "quality").
    cc_upsample : int, default=4
        Cross-correlation upsampling factor for rigid initialization
    sigma_smooth : float, default=6.0
        Gaussian smoothing sigma for inter-sequence alignment
    alpha_between : float, default=25.0
        Regularization for inter-sequence optical flow
    iterations_between : int, default=100
        Iterations for inter-sequence optical flow
    align_chunk_size : int, default=64
        Number of frames to process per batch during Stage 3 video alignment
    align_output_format : str, default="TIFF"
        Output format for aligned videos in Stage 3

    MATLAB Compatibility
    --------------------
    Mirrors configuration from align_full_v3_checkpoint.m and
    get_session_valid_index_v3_progressprint.m
    """

    root: Path
    pattern: str = "*.tif"
    center: Optional[str] = None
    output_root: Path = Field(default=Path("compensated_outputs"))
    final_results: Path = Field(default=Path("final_results"))
    resume: bool = True
    scheduler: Literal["local", "array", "dask"] = "local"
    flow_backend: str = "flowreg"
    backend_params: Dict = Field(default_factory=dict)
    flow_options: Optional[Union[Dict[str, Any], Path]] = None

    # Stage 1 parameters
    stage1_quality_setting: Optional[str] = None  # Pass through to OF_options

    # Stage 2 parameters
    cc_upsample: int = 4
    sigma_smooth: float = 6.0
    alpha_between: float = 25.0
    iterations_between: int = 100

    # Stage 3 parameters
    align_chunk_size: int = 64
    align_output_format: str = "TIFF"

    @field_validator("root", "output_root", "final_results", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("flow_options", mode="before")
    @classmethod
    def validate_flow_options_field(cls, v):
        """Normalize flow_options to mapping or filesystem path."""
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            return Path(v)
        if isinstance(v, Path):
            return v
        raise TypeError("flow_options must be a mapping or path string")

    @field_validator("root")
    @classmethod
    def validate_root_exists(cls, v):
        """Ensure root directory exists."""
        if not v.exists():
            raise ValueError(f"Root directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Root path is not a directory: {v}")
        return v

    def resolve_output_paths(self):
        """
        Resolve output paths relative to root if they are not absolute.

        Returns
        -------
        tuple[Path, Path]
            Absolute paths for (output_root, final_results)
        """
        output_root = self.output_root
        if not output_root.is_absolute():
            output_root = self.root / output_root

        final_results = self.final_results
        if not final_results.is_absolute():
            final_results = self.root / final_results

        return output_root, final_results

    def get_flow_options_override(self) -> Dict[str, Any]:
        """
        Return OFOptions overrides defined in the config.

        Supports inline dictionaries or JSON files saved via OF_options.
        """
        if self.flow_options is None:
            return {}

        if isinstance(self.flow_options, dict):
            # Return a shallow copy so callers can mutate freely
            return dict(self.flow_options)

        flow_path = self.flow_options.expanduser()
        if not flow_path.is_absolute():
            flow_path = self.root / flow_path

        if not flow_path.exists():
            raise ValueError(f"Flow options file not found: {flow_path}")

        from pyflowreg.motion_correction.OF_options import OFOptions

        options = OFOptions.load_options(flow_path)
        allowed_fields = set(OFOptions.model_fields.keys())

        # Never allow users to override per-recording routing
        allowed_fields.discard("input_file")
        allowed_fields.discard("output_path")

        return {
            key: value
            for key, value in options.model_dump().items()
            if key in allowed_fields
        }

    def resolve_center_file(self, input_files):
        """
        Resolve the center reference file from configuration or default.

        Parameters
        ----------
        input_files : list[Path]
            List of discovered input files

        Returns
        -------
        tuple[int, Path]
            Index and path of center file
        """
        if self.center is not None:
            # User specified center file
            center_path = self.root / self.center
            try:
                idx = input_files.index(center_path)
                return idx, center_path
            except ValueError:
                raise ValueError(
                    f"Specified center file '{self.center}' not found in "
                    f"discovered inputs matching pattern '{self.pattern}'"
                )
        else:
            # Default: lexicographic middle (MATLAB: ceil(num_records/2))
            sorted_files = sorted(input_files)
            middle_idx = (len(sorted_files) + 1) // 2 - 1  # 0-indexed
            return middle_idx, sorted_files[middle_idx]

    @classmethod
    def from_toml(cls, path):
        """
        Load configuration from TOML file.

        Parameters
        ----------
        path : str or Path
            Path to TOML configuration file

        Returns
        -------
        SessionConfig
            Parsed and validated configuration
        """
        import sys

        # Python 3.11+ has tomllib in stdlib
        if sys.version_info >= (3, 11):
            import tomllib

            with open(path, "rb") as f:
                data = tomllib.load(f)
        else:
            # Try tomli for older Python versions
            try:
                import tomli

                with open(path, "rb") as f:
                    data = tomli.load(f)
            except ImportError:
                raise ImportError(
                    "TOML support requires 'tomli' package for Python < 3.11. "
                    "Install with: pip install tomli"
                )

        return cls(**data)

    @classmethod
    def from_yaml(cls, path):
        """
        Load configuration from YAML file.

        Parameters
        ----------
        path : str or Path
            Path to YAML configuration file

        Returns
        -------
        SessionConfig
            Parsed and validated configuration
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "YAML support requires 'pyyaml' package. "
                "Install with: pip install pyyaml"
            )

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_file(cls, path):
        """
        Auto-detect format and load configuration.

        Parameters
        ----------
        path : str or Path
            Path to configuration file (.toml or .yaml/.yml)

        Returns
        -------
        SessionConfig
            Parsed and validated configuration
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".toml":
            return cls.from_toml(path)
        elif suffix in [".yaml", ".yml"]:
            return cls.from_yaml(path)
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. "
                "Use .toml, .yaml, or .yml"
            )


def get_array_task_id():
    """
    Get job array task ID from environment variables.

    Checks for SLURM_ARRAY_TASK_ID, SGE_TASK_ID, and PBS_ARRAY_INDEX
    in that order.

    Returns
    -------
    int or None
        Task ID if running in array job, None otherwise
    """
    for env_var in ["SLURM_ARRAY_TASK_ID", "SGE_TASK_ID", "PBS_ARRAY_INDEX"]:
        task_id = os.environ.get(env_var)
        if task_id is not None:
            return int(task_id)
    return None
