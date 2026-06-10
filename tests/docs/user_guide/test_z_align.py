"""
Executable-documentation tests for docs/user_guide/z_align.md.

The z-alignment guide is a narrative walkthrough whose only code blocks are a
TOML ``ZAlignConfig`` example and the ``pyflowreg-z-align`` CLI commands. Both
are left inline on the page (TOML/bash are not Python and so are not extracted
into runnable snippets); the full three-stage ``run_all_stages`` pipeline is
never shown as a Python block.

Following the pattern used for the session guide, this test validates the
shipped ``examples/z_align_config.toml`` -- which exercises the same
``ZAlignConfig`` fields the page documents -- by loading it through
``ZAlignConfig.from_toml`` and asserting the parsed field values against the
defaults and semantics defined in ``pyflowreg.z_align.config``. ``extra="forbid"``
on the model means a typo'd or removed key would raise here, so this also
guards the documented field names.
"""

from pathlib import Path

import pytest

from pyflowreg.z_align.config import ZAlignConfig

pytestmark = pytest.mark.docs_example

# tests/docs/user_guide/test_z_align.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


class TestZAlignExampleConfig:
    """Validate examples/z_align_config.toml as documented in z_align.md."""

    def test_z_align_config_toml_loads(self):
        example_path = REPO_ROOT / "examples" / "z_align_config.toml"
        if not example_path.exists():
            pytest.skip("Example z-align config not found")

        # root = "." resolves to an existing directory, so the example loads
        # without path substitution (input files are not existence-checked at
        # construction time -- only `root` is validated).
        config = ZAlignConfig.from_toml(example_path)

        # Core paths
        assert config.root == Path(".")
        assert config.input_file == Path("compensated.tiff")
        assert config.volume_input_file == Path("file_00004_00001.tif")
        assert config.reference_volume is None
        assert config.reference_source_file == Path("compensated.tiff")

        # Reference image building
        assert config.reference_source_frames == 2000
        assert config.reference_source_buffer_size == 10
        assert config.reference_source_bin_size == 20

        # Output locations
        assert config.output_root == Path(".")
        assert config.volume_output_dir == Path("aligned_stack")
        assert config.recording_prealigned_output_dir == Path("prealigned_recording")
        assert config.z_shift_file == Path("z_shift.HDF5")
        assert config.corrected_output_file == Path("compensated_shift_corrected.tif")
        assert config.simulated_output_file == Path("simulated_from_z.tif")

        # Control flags
        assert config.resume is True
        assert config.prealign_stack is True
        assert config.prealign_recording is False
        assert config.write_corrected is True
        assert config.write_simulated is True

        # Stage 1 (volume build)
        assert config.stage1_alpha == 5.0
        assert config.stage1_quality_setting == "quality"
        assert config.stage1_buffer_size == 500
        assert config.stage1_bin_size == 1
        assert config.stage1_update_reference is True
        assert config.stack_scans_per_slice is None  # commented out in the file
        assert config.flow_backend == "flowreg"
        assert config.backend_params == {}
        assert config.stage1_flow_options is None
        assert config.recording_prealign_flow_options is None

        # Stage 2 (patch-based z estimation)
        assert config.input_buffer_size == 50
        assert config.input_bin_size == 1
        assert config.volume_buffer_size == 500
        assert config.volume_bin_size == 1
        assert config.win_half == 10
        assert config.patch_size == 128
        assert config.overlap == 0.75
        assert config.spatial_sigma == 1.5
        assert config.temporal_sigma == 1.5
        assert config.z_smooth_sigma_spatial == 5.0
        assert config.z_smooth_sigma_temporal == 1.5
        assert config.parabolic_tau_scale == pytest.approx(1e-3)
        assert config.output_dtype == "uint16"

        # Defaults not set in the file (verify against config.py)
        assert config.n_jobs == -1
        assert config.parallelization == "sequential"

    def test_effective_volume_bin_size_default(self):
        """effective_volume_bin_size falls back to volume_bin_size when no
        stack_scans_per_slice is set (as in the example config)."""
        example_path = REPO_ROOT / "examples" / "z_align_config.toml"
        if not example_path.exists():
            pytest.skip("Example z-align config not found")

        config = ZAlignConfig.from_toml(example_path)
        assert config.effective_volume_bin_size() == config.volume_bin_size == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
