"""
Tests for z-align configuration.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.z_align.config import ZAlignConfig


class TestZAlignConfigBasics:
    """Test basic config creation and validation."""

    def test_minimal_valid_config(self, tmp_path):
        cfg = ZAlignConfig(root=tmp_path, input_file="compensated.tiff")
        assert cfg.root == tmp_path
        assert cfg.input_file == Path("compensated.tiff")
        assert cfg.resume is True
        assert cfg.write_corrected is True
        assert cfg.write_simulated is True

    def test_root_must_exist(self, tmp_path):
        with pytest.raises(ValidationError, match="Root directory does not exist"):
            ZAlignConfig(
                root=tmp_path / "missing_root",
                input_file="compensated.tiff",
            )

    def test_overlap_out_of_range_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="overlap must satisfy"):
            ZAlignConfig(root=tmp_path, input_file="compensated.tiff", overlap=1.0)

    def test_invalid_output_dtype_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="Invalid output_dtype"):
            ZAlignConfig(
                root=tmp_path,
                input_file="compensated.tiff",
                output_dtype="not_a_dtype",
            )


class TestZAlignConfigPathResolution:
    """Test path resolution behavior."""

    def test_resolve_relative_paths(self, tmp_path):
        cfg = ZAlignConfig(
            root=tmp_path,
            input_file="compensated.tiff",
            output_root="z_out",
            volume_output_dir="aligned_stack",
            z_shift_file="z_shift.HDF5",
            corrected_output_file="compensated_shift_corrected.tif",
            simulated_output_file="simulated_from_z.tif",
        )

        assert cfg.resolve_output_root() == tmp_path / "z_out"
        assert cfg.resolve_input_file() == tmp_path / "compensated.tiff"
        assert cfg.resolve_volume_output_dir() == tmp_path / "z_out" / "aligned_stack"
        assert cfg.resolve_z_shift_file() == tmp_path / "z_out" / "z_shift.HDF5"
        assert cfg.resolve_corrected_output_file() == (
            tmp_path / "z_out" / "compensated_shift_corrected.tif"
        )
        assert cfg.resolve_simulated_output_file() == (
            tmp_path / "z_out" / "simulated_from_z.tif"
        )

    def test_resolve_reference_volume_prefers_existing_default(self, tmp_path):
        volume_dir = tmp_path / "z_out" / "aligned_stack"
        volume_dir.mkdir(parents=True)
        existing = volume_dir / "compensated.hdf5"
        existing.touch()

        cfg = ZAlignConfig(
            root=tmp_path,
            input_file="compensated.tiff",
            output_root="z_out",
            volume_output_dir="aligned_stack",
        )
        assert cfg.resolve_reference_volume_path() == existing

    def test_resolve_reference_volume_explicit(self, tmp_path):
        explicit = tmp_path / "my_ref.h5"
        explicit.touch()
        cfg = ZAlignConfig(
            root=tmp_path,
            input_file="compensated.tiff",
            reference_volume="my_ref.h5",
        )
        assert cfg.resolve_reference_volume_path() == explicit


class TestZAlignFlowOptions:
    """Test stage-1 OFOptions override loading."""

    def test_stage1_flow_options_dict_returns_copy(self, tmp_path):
        cfg = ZAlignConfig(
            root=tmp_path,
            input_file="compensated.tiff",
            stage1_flow_options={"alpha": (7.0, 7.0), "buffer_size": 123},
        )
        overrides = cfg.get_stage1_overrides()
        assert overrides["buffer_size"] == 123
        overrides["buffer_size"] = 999
        # Ensure config stored mapping is unaffected
        assert cfg.stage1_flow_options["buffer_size"] == 123

    def test_stage1_flow_options_from_json_file(self, tmp_path):
        options_path = tmp_path / "of_options.json"
        opts = OFOptions(
            input_file="input.tif",
            output_path=tmp_path / "out_dir",
            quality_setting="balanced",
            alpha=3.0,
            buffer_size=777,
        )
        opts.save_options(options_path)

        cfg = ZAlignConfig(
            root=tmp_path,
            input_file="compensated.tiff",
            stage1_flow_options=options_path,
        )
        overrides = cfg.get_stage1_overrides()

        assert overrides["quality_setting"] == "balanced"
        assert overrides["buffer_size"] == 777
        assert "input_file" not in overrides
        assert "output_path" not in overrides

    def test_stage1_flow_options_file_missing_raises(self, tmp_path):
        cfg = ZAlignConfig(
            root=tmp_path,
            input_file="compensated.tiff",
            stage1_flow_options="missing_options.json",
        )
        with pytest.raises(ValueError, match="not found"):
            cfg.get_stage1_overrides()


class TestZAlignConfigFileLoading:
    """Test config file loading helpers."""

    def test_load_from_toml(self, tmp_path):
        root_posix = tmp_path.as_posix()
        cfg_file = tmp_path / "z_align.toml"
        cfg_file.write_text(
            "\n".join(
                [
                    f'root = "{root_posix}"',
                    'input_file = "compensated.tiff"',
                    'output_root = "z_out"',
                    "write_corrected = false",
                    "write_simulated = true",
                    "patch_size = 64",
                ]
            ),
            encoding="utf-8",
        )

        cfg = ZAlignConfig.from_toml(cfg_file)
        assert cfg.root == tmp_path
        assert cfg.input_file == Path("compensated.tiff")
        assert cfg.output_root == Path("z_out")
        assert cfg.write_corrected is False
        assert cfg.write_simulated is True
        assert cfg.patch_size == 64

    def test_from_file_unsupported_suffix_raises(self, tmp_path):
        cfg_file = tmp_path / "z_align.json"
        cfg_file.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported config file format"):
            ZAlignConfig.from_file(cfg_file)
