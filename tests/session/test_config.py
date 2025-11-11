"""
Tests for session configuration system.

Tests SessionConfig validation, file loading, path resolution, and
job array task ID detection.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.session.config import SessionConfig, get_array_task_id


class TestSessionConfigBasics:
    """Test basic SessionConfig creation and validation."""

    def test_minimal_config(self, tmp_path):
        """Test creating config with minimal required parameters."""
        config = SessionConfig(root=tmp_path)

        assert config.root == tmp_path
        assert config.pattern == "*.tif"
        assert config.resume is True
        assert config.scheduler == "local"

    def test_root_must_exist(self, tmp_path):
        """Test that root directory must exist."""
        nonexistent = tmp_path / "does_not_exist"

        with pytest.raises(ValidationError, match="Root directory does not exist"):
            SessionConfig(root=nonexistent)

    def test_root_must_be_directory(self, tmp_path):
        """Test that root must be a directory, not a file."""
        file_path = tmp_path / "file.txt"
        file_path.touch()

        with pytest.raises(ValidationError, match="not a directory"):
            SessionConfig(root=file_path)

    def test_string_paths_converted(self, tmp_path):
        """Test that string paths are converted to Path objects."""
        config = SessionConfig(
            root=str(tmp_path),
            output_root="outputs",
            final_results="results",
        )

        assert isinstance(config.root, Path)
        assert isinstance(config.output_root, Path)
        assert isinstance(config.final_results, Path)

    @pytest.mark.parametrize("scheduler", ["local", "array", "dask"])
    def test_valid_schedulers(self, tmp_path, scheduler):
        """Test that valid scheduler values are accepted."""
        config = SessionConfig(root=tmp_path, scheduler=scheduler)
        assert config.scheduler == scheduler

    def test_invalid_scheduler(self, tmp_path):
        """Test that invalid scheduler raises validation error."""
        with pytest.raises(ValidationError):
            SessionConfig(root=tmp_path, scheduler="invalid")


class TestSessionConfigPathResolution:
    """Test path resolution methods."""

    def test_resolve_absolute_paths(self, tmp_path):
        """Test resolving absolute output paths."""
        output_dir = tmp_path / "outputs"
        results_dir = tmp_path / "results"

        config = SessionConfig(
            root=tmp_path, output_root=output_dir, final_results=results_dir
        )

        resolved_output, resolved_results = config.resolve_output_paths()

        assert resolved_output == output_dir
        assert resolved_results == results_dir

    def test_resolve_relative_paths(self, tmp_path):
        """Test resolving relative output paths."""
        config = SessionConfig(
            root=tmp_path, output_root="outputs", final_results="results"
        )

        resolved_output, resolved_results = config.resolve_output_paths()

        assert resolved_output == tmp_path / "outputs"
        assert resolved_results == tmp_path / "results"

    def test_resolve_mixed_paths(self, tmp_path):
        """Test resolving mix of absolute and relative paths."""
        abs_output = tmp_path / "abs_outputs"

        config = SessionConfig(
            root=tmp_path, output_root=abs_output, final_results="rel_results"
        )

        resolved_output, resolved_results = config.resolve_output_paths()

        assert resolved_output == abs_output
        assert resolved_results == tmp_path / "rel_results"


class TestSessionConfigCenterFileResolution:
    """Test center reference file resolution."""

    def test_explicit_center_file(self, tmp_path):
        """Test explicitly specified center file."""
        # Create test files
        for i in range(5):
            (tmp_path / f"file_{i:02d}.tif").touch()

        center_name = "file_02.tif"
        config = SessionConfig(root=tmp_path, center=center_name)

        input_files = sorted(tmp_path.glob("*.tif"))
        center_idx, center_file = config.resolve_center_file(input_files)

        assert center_idx == 2
        assert center_file.name == center_name

    def test_lexicographic_middle_odd(self, tmp_path):
        """Test lexicographic middle with odd number of files."""
        # Create 5 files (middle should be index 2, file_02.tif)
        for i in range(5):
            (tmp_path / f"file_{i:02d}.tif").touch()

        config = SessionConfig(root=tmp_path)  # No explicit center

        input_files = sorted(tmp_path.glob("*.tif"))
        center_idx, center_file = config.resolve_center_file(input_files)

        # ceil(5/2) - 1 = 2 (0-indexed)
        assert center_idx == 2
        assert center_file == input_files[2]

    def test_lexicographic_middle_even(self, tmp_path):
        """Test lexicographic middle with even number of files."""
        # Create 4 files (middle should be index 1, file_01.tif)
        for i in range(4):
            (tmp_path / f"file_{i:02d}.tif").touch()

        config = SessionConfig(root=tmp_path)

        input_files = sorted(tmp_path.glob("*.tif"))
        center_idx, center_file = config.resolve_center_file(input_files)

        # ceil(4/2) - 1 = 1 (0-indexed)
        assert center_idx == 1
        assert center_file == input_files[1]

    def test_nonexistent_center_raises(self, tmp_path):
        """Test that non-existent center file raises error."""
        (tmp_path / "file_00.tif").touch()

        config = SessionConfig(root=tmp_path, center="nonexistent.tif")

        input_files = sorted(tmp_path.glob("*.tif"))

        with pytest.raises(ValueError, match="not found"):
            config.resolve_center_file(input_files)


class TestSessionConfigFileLoading:
    """Test loading configuration from files."""

    def test_load_from_toml(self, tmp_path):
        """Test loading configuration from TOML file."""
        pytest.importorskip("tomli", reason="tomli required for Python < 3.11")

        # Convert Windows backslashes to forward slashes for TOML
        root_path = str(tmp_path).replace("\\", "/")
        config_content = f"""
root = "{root_path}"
pattern = "*.tiff"
output_root = "my_outputs"
resume = false
scheduler = "array"
cc_upsample = 8
sigma_smooth = 4.0
alpha_between = 30.0
iterations_between = 150
"""

        config_file = tmp_path / "session.toml"
        config_file.write_text(config_content)

        config = SessionConfig.from_toml(config_file)

        assert config.root == tmp_path
        assert config.pattern == "*.tiff"
        assert config.output_root == Path("my_outputs")
        assert config.resume is False
        assert config.scheduler == "array"
        assert config.cc_upsample == 8
        assert config.sigma_smooth == 4.0
        assert config.alpha_between == 30.0
        assert config.iterations_between == 150

    def test_load_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        pytest.importorskip("yaml", reason="pyyaml required for YAML support")

        config_content = f"""
root: {str(tmp_path)}
pattern: "*.tiff"
output_root: my_outputs
resume: false
scheduler: dask
"""

        config_file = tmp_path / "session.yaml"
        config_file.write_text(config_content)

        config = SessionConfig.from_yaml(config_file)

        assert config.root == tmp_path
        assert config.pattern == "*.tiff"
        assert config.scheduler == "dask"

    def test_from_file_autodetects_toml(self, tmp_path):
        """Test that from_file() auto-detects TOML format."""
        # Convert Windows backslashes to forward slashes for TOML
        root_path = str(tmp_path).replace("\\", "/")
        config_content = f'root = "{root_path}"\n'

        config_file = tmp_path / "session.toml"
        config_file.write_text(config_content)

        config = SessionConfig.from_file(config_file)

        assert config.root == tmp_path

    def test_from_file_autodetects_yaml(self, tmp_path):
        """Test that from_file() auto-detects YAML format."""
        pytest.importorskip("yaml")

        config_content = f"root: {str(tmp_path)}\n"

        config_file = tmp_path / "session.yml"
        config_file.write_text(config_content)

        config = SessionConfig.from_file(config_file)

        assert config.root == tmp_path

    def test_from_file_unsupported_format(self, tmp_path):
        """Test that unsupported format raises error."""
        config_file = tmp_path / "session.json"
        config_file.write_text("{}")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            SessionConfig.from_file(config_file)


class TestArrayTaskIDDetection:
    """Test job array task ID detection from environment."""

    @pytest.mark.parametrize(
        "env_var,task_id",
        [
            ("SLURM_ARRAY_TASK_ID", 5),
            ("SGE_TASK_ID", 10),
            ("PBS_ARRAY_INDEX", 3),
        ],
    )
    def test_detect_array_task_id(self, env_var, task_id, monkeypatch):
        """Test detection of array task ID from various schedulers."""
        # Clear all array-related env vars
        for var in ["SLURM_ARRAY_TASK_ID", "SGE_TASK_ID", "PBS_ARRAY_INDEX"]:
            monkeypatch.delenv(var, raising=False)

        # Set specific env var
        monkeypatch.setenv(env_var, str(task_id))

        detected_id = get_array_task_id()

        assert detected_id == task_id

    def test_no_array_task_id_returns_none(self, monkeypatch):
        """Test that missing array ID returns None."""
        # Clear all array-related env vars
        for var in ["SLURM_ARRAY_TASK_ID", "SGE_TASK_ID", "PBS_ARRAY_INDEX"]:
            monkeypatch.delenv(var, raising=False)

        detected_id = get_array_task_id()

        assert detected_id is None

    def test_slurm_precedence_over_sge(self, monkeypatch):
        """Test that SLURM_ARRAY_TASK_ID takes precedence over SGE_TASK_ID."""
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "5")
        monkeypatch.setenv("SGE_TASK_ID", "10")

        detected_id = get_array_task_id()

        assert detected_id == 5, "SLURM should take precedence"


class TestSessionConfigStage2Parameters:
    """Test Stage 2 (inter-sequence alignment) parameters."""

    def test_default_stage2_parameters(self, tmp_path):
        """Test that Stage 2 parameters have correct defaults."""
        config = SessionConfig(root=tmp_path)

        assert config.cc_upsample == 4
        assert config.sigma_smooth == 6.0
        assert config.alpha_between == 25.0
        assert config.iterations_between == 100

    def test_custom_stage2_parameters(self, tmp_path):
        """Test setting custom Stage 2 parameters."""
        config = SessionConfig(
            root=tmp_path,
            cc_upsample=8,
            sigma_smooth=4.5,
            alpha_between=20.0,
            iterations_between=150,
        )

        assert config.cc_upsample == 8
        assert config.sigma_smooth == 4.5
        assert config.alpha_between == 20.0
        assert config.iterations_between == 150


class TestSessionConfigBackendParameters:
    """Test backend configuration."""

    def test_default_backend(self, tmp_path):
        """Test that default backend is 'flowreg'."""
        config = SessionConfig(root=tmp_path)

        assert config.flow_backend == "flowreg"
        assert config.backend_params == {}

    def test_custom_backend(self, tmp_path):
        """Test setting custom backend."""
        config = SessionConfig(
            root=tmp_path,
            flow_backend="torch",
            backend_params={"device": "cuda:0", "precision": "fp16"},
        )

        assert config.flow_backend == "torch"
        assert config.backend_params["device"] == "cuda:0"
        assert config.backend_params["precision"] == "fp16"


class TestSessionConfigFlowOptions:
    """Test flow_options parsing and loading."""

    def test_inline_dict_override(self, tmp_path):
        """Inline dictionary should be returned unchanged."""
        config = SessionConfig(root=tmp_path, flow_options={"alpha": 8, "save_w": True})

        overrides = config.get_flow_options_override()

        assert overrides["alpha"] == 8
        assert overrides["save_w"] is True

    def test_blank_string_treated_as_none(self, tmp_path):
        """Blank string should act as unset."""
        config = SessionConfig(root=tmp_path, flow_options="   ")

        assert config.flow_options is None
        assert config.get_flow_options_override() == {}

    def test_relative_json_path(self, tmp_path):
        """Relative JSON path should load using OFOptions."""
        options_dir = tmp_path / "options"
        options_dir.mkdir()

        options_path = options_dir / "stage1.json"
        options = OFOptions(
            input_file=str(tmp_path / "recording.tif"),
            output_path=tmp_path / "outputs",
            alpha=6,
            save_meta_info=False,
            save_w=True,
        )
        options.save_options(options_path)

        config = SessionConfig(root=tmp_path, flow_options="options/stage1.json")

        overrides = config.get_flow_options_override()

        alpha_override = overrides["alpha"]
        if isinstance(alpha_override, (tuple, list)):
            assert all(val == pytest.approx(6) for val in alpha_override)
        else:
            assert alpha_override == pytest.approx(6)
        assert overrides["save_meta_info"] is False
        assert overrides["save_w"] is True
        assert "input_file" not in overrides
        assert "output_path" not in overrides


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
