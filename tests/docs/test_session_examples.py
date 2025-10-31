"""
Tests for documentation examples and config files.

Ensures that example configuration files and documentation code snippets work.
"""

from pathlib import Path

import pytest

from pyflowreg.session.config import SessionConfig


class TestExampleConfigFiles:
    """Test that example config files in examples/ are valid."""

    def test_example_toml_loads(self, tmp_path):
        """Test that session_config.toml example is valid."""
        # Read example file
        example_path = (
            Path(__file__).parent.parent.parent / "examples" / "session_config.toml"
        )
        if not example_path.exists():
            pytest.skip("Example config not found")

        content = example_path.read_text()

        # Replace placeholder path with temp directory (use forward slashes for TOML)
        content = content.replace(
            "/path/to/your/experiment/", tmp_path.as_posix() + "/"
        )

        # Write to temp file
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(content)

        # Should load without errors
        config = SessionConfig.from_toml(config_file)

        # Verify basic fields
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

    def test_example_yaml_loads(self, tmp_path):
        """Test that session_config.yml example is valid."""
        pytest.importorskip("yaml", reason="pyyaml required for YAML support")

        # Read example file
        example_path = (
            Path(__file__).parent.parent.parent / "examples" / "session_config.yml"
        )
        if not example_path.exists():
            pytest.skip("Example config not found")

        content = example_path.read_text()

        # Replace placeholder path with temp directory (use forward slashes)
        content = content.replace(
            "/path/to/your/experiment/", tmp_path.as_posix() + "/"
        )

        # Write to temp file
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(content)

        # Should load without errors
        config = SessionConfig.from_yaml(config_file)

        # Verify basic fields
        assert config.root == tmp_path
        assert config.pattern == "*.tif"
        assert config.output_root == Path("compensated_outputs")
        assert config.final_results == Path("final_results")
        assert config.resume is True
        assert config.scheduler == "local"
        assert config.flow_backend == "flowreg"


class TestDocumentationCodeSnippets:
    """Test code snippets from documentation work correctly."""

    def test_quickstart_session_example(self, tmp_path):
        """Test the session processing example from quickstart.md."""
        # Create dummy files
        for i in range(3):
            (tmp_path / f"recording_{i:02d}.tif").touch()

        # Code from quickstart.md
        from pyflowreg.session import SessionConfig

        # Configure session
        config = SessionConfig(
            root=tmp_path,
            pattern="recording_*.tif",
            output_root="compensated",
            resume=True,
        )

        # Should resolve paths correctly
        output_root, final_results = config.resolve_output_paths()
        assert output_root == tmp_path / "compensated"
        assert final_results == tmp_path / "final_results"

        # Should find input files
        from pyflowreg.session.stage1_compensate import discover_input_files

        input_files = discover_input_files(config)
        assert len(input_files) == 3

    def test_api_documentation_example(self, tmp_path):
        """Test the complete example from api/session.md."""
        # Create dummy files
        for i in range(5):
            (tmp_path / f"recording_{i:03d}.tif").touch()

        # Code from api/session.md
        from pyflowreg.session import SessionConfig

        config = SessionConfig(
            root=tmp_path,
            pattern="recording_*.tif",
            center="recording_002.tif",
            output_root="compensated",
            final_results="results",
            resume=True,
            flow_backend="flowreg",
            cc_upsample=4,
            sigma_smooth=6.0,
            alpha_between=25.0,
            iterations_between=100,
        )

        # Verify configuration
        assert config.root == tmp_path
        assert config.center == "recording_002.tif"
        assert config.cc_upsample == 4

        # Verify center resolution
        from pyflowreg.session.stage1_compensate import discover_input_files

        input_files = discover_input_files(config)
        middle_idx, center_file = config.resolve_center_file(input_files)
        assert middle_idx == 2
        assert center_file.name == "recording_002.tif"


class TestAtomicSaveFunctions:
    """Test atomic save utilities for crash safety."""

    def test_atomic_save_npy(self, tmp_path):
        """Test atomic_save_npy creates file atomically."""
        import numpy as np
        from pyflowreg.session.stage1_compensate import atomic_save_npy

        target = tmp_path / "test.npy"
        arr = np.array([1, 2, 3, 4, 5])

        # Save atomically
        atomic_save_npy(target, arr)

        # File should exist and contain correct data
        assert target.exists()
        loaded = np.load(str(target))
        np.testing.assert_array_equal(loaded, arr)

        # Temp file should not exist
        assert not (tmp_path / "test.npy.tmp").exists()

    def test_atomic_save_npz(self, tmp_path):
        """Test atomic_save_npz creates file atomically."""
        import numpy as np
        from pyflowreg.session.stage1_compensate import atomic_save_npz

        target = tmp_path / "test.npz"
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([[4, 5], [6, 7]])

        # Save atomically
        atomic_save_npz(target, array1=arr1, array2=arr2)

        # File should exist and contain correct data
        assert target.exists()
        data = np.load(str(target))
        np.testing.assert_array_equal(data["array1"], arr1)
        np.testing.assert_array_equal(data["array2"], arr2)

        # Temp file should not exist
        assert not (tmp_path / "test.npz.tmp").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
