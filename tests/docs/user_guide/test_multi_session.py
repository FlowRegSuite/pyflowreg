"""
Tests for docs/user_guide/multi_session.md.

Executes the extracted session-pipeline snippet against a tiny synthetic
session (via the ``materialize_session`` / ``snippet_runner`` harness in
``tests/docs/conftest.py``) and validates the example session configuration
files shipped in ``examples/``.

The TOML/YAML config-validation tests were ported from the now-removed
``tests/docs/test_session_examples.py``.
"""

from pathlib import Path

import numpy as np
import pytest

from pyflowreg.session.config import SessionConfig
from pyflowreg.util.io.factory import get_video_file_reader

pytestmark = pytest.mark.docs_example

# tests/docs/user_guide/test_multi_session.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


class TestExampleConfigFiles:
    """Validate the example session config files referenced by the guide."""

    def test_example_toml_loads(self, tmp_path):
        """examples/session_config.toml parses into a valid SessionConfig."""
        example_path = REPO_ROOT / "examples" / "session_config.toml"
        if not example_path.exists():
            pytest.skip("Example config not found")

        content = example_path.read_text()

        # Replace placeholder path with temp directory (forward slashes for TOML)
        content = content.replace(
            "/path/to/your/experiment/", tmp_path.as_posix() + "/"
        )

        config_file = tmp_path / "test_config.toml"
        config_file.write_text(content)

        config = SessionConfig.from_toml(config_file)

        assert config.root == tmp_path
        assert config.pattern == "*.tif"
        assert config.output_root == Path("compensated_outputs")
        assert config.final_results == Path("final_results")
        assert config.resume is True
        assert config.scheduler == "local"
        assert config.flow_backend == "flowreg"
        assert config.cc_upsample == 4
        assert config.sigma_smooth == 6.0
        assert config.alpha_between == 25.0
        assert config.iterations_between == 100
        assert config.stage2_constancy_assumption == "gc"

    def test_example_yaml_loads(self, tmp_path):
        """examples/session_config.yml parses into a valid SessionConfig."""
        pytest.importorskip("yaml", reason="pyyaml required for YAML support")

        example_path = REPO_ROOT / "examples" / "session_config.yml"
        if not example_path.exists():
            pytest.skip("Example config not found")

        content = example_path.read_text()

        content = content.replace(
            "/path/to/your/experiment/", tmp_path.as_posix() + "/"
        )

        config_file = tmp_path / "test_config.yml"
        config_file.write_text(content)

        config = SessionConfig.from_yaml(config_file)

        assert config.root == tmp_path
        assert config.pattern == "*.tif"
        assert config.output_root == Path("compensated_outputs")
        assert config.final_results == Path("final_results")
        assert config.resume is True
        assert config.scheduler == "local"
        assert config.flow_backend == "flowreg"


class TestMultiSessionPipeline:
    """Run the SessionConfig + run_stage1/2/3 snippet from the guide."""

    @pytest.mark.slow
    def test_session_pipeline_executes(self, materialize_session, snippet_runner):
        # Harness defaults: root="session", recording_000/001/002.tif,
        # shape (6, 64, 64, 1) -- matches the snippet's root/pattern.
        materialize_session()

        ns = snippet_runner("user_guide/multi_session/session_pipeline.py")

        config = ns["config"]
        assert isinstance(config, SessionConfig)
        assert config.pattern == "recording_*.tif"

        # Stage 1: one output folder per recording.
        output_folders = ns["output_folders"]
        assert len(output_folders) == 3
        for folder in output_folders:
            assert (folder / "compensated.hdf5").exists()
            assert (folder / "temporal_average.npy").exists()
            assert (folder / "idx.hdf").exists()

        # Stage 2: middle_idx is 0-based center; displacements (H, W, 2) each.
        assert ns["middle_idx"] == 1
        assert Path(ns["center_file"]).name == "recording_001.tif"
        displacements = ns["displacements"]
        assert len(displacements) == 3
        for w in displacements:
            assert w.shape == (64, 64, 2)
        # Center recording has zero displacement.
        assert np.all(displacements[ns["middle_idx"]] == 0)

        # Stage 3: final session mask is a 2-D boolean array.
        final_mask = ns["final_mask"]
        assert final_mask.shape == (64, 64)
        assert final_mask.dtype == bool

        # Final results bundle written to disk and loadable.
        output_root, final_results = config.resolve_output_paths()
        results = np.load(final_results / "final_valid_idx.npz")
        np.testing.assert_array_equal(results["final_valid"], final_mask)
        assert int(results["middle_idx"]) == 1
        assert (final_results / "final_valid_idx.png").exists()

        # Aligned videos are readable through the PyFlowReg I/O system.
        aligned = final_results / "aligned_recording_000.tif"
        assert aligned.exists()
        reader = get_video_file_reader(str(aligned))
        try:
            assert reader[:].shape[1:3] == (64, 64)
        finally:
            reader.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
