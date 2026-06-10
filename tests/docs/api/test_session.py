"""
Executable-documentation tests for docs/api/session.md.

Each pipeline test materializes a tiny synthetic session via the
``materialize_session`` fixture, executes the extracted snippet from
``docs/snippets/api/session/`` via ``snippet_runner``, and asserts on the
resulting namespace and the files the pipeline writes.

The pipeline tests restrict the runtime to the in-process ``sequential``
executor via ``RuntimeContext.use`` so the published examples (Stage 1
auto-selects the best executor, preferring multiprocessing) stay fast
inside the test suite.
"""

from pathlib import Path

import numpy as np
import pytest

from pyflowreg._runtime import RuntimeContext

pytestmark = pytest.mark.docs_example

# Shape of each synthetic recording written by materialize_session.
RECORDING_SHAPE = (6, 64, 64, 1)
N_RECORDINGS = 3

# Minimal session.toml for the "Stage 1: Per-Recording Compensation"
# example; the root points at the session directory created by the
# materialize_session fixture.
SESSION_TOML = """\
root = "session"
pattern = "recording_*.tif"
output_root = "compensated"
resume = true
"""

SESSION_YAML = """\
root: session
pattern: "recording_*.tif"
output_root: compensated
resume: true
"""


class TestSessionRunStage1Example:
    """docs/api/session.md, section "Stage 1: Per-Recording Compensation"."""

    def test_run_stage1_example_executes(
        self, materialize_session, snippet_runner, tmp_path
    ):
        materialize_session(
            n_recordings=N_RECORDINGS, shape=RECORDING_SHAPE, root="session"
        )
        (tmp_path / "session.toml").write_text(SESSION_TOML)

        with RuntimeContext.use(available_parallelization={"sequential"}):
            ns = snippet_runner("api/session/run_stage1_example.py")

        output_folders = ns["output_folders"]
        assert len(output_folders) == N_RECORDINGS

        for i, folder in enumerate(output_folders):
            assert folder.name == f"recording_{i:03d}"
            # Compensated output (extension case depends on the writer).
            assert (folder / "compensated.HDF5").exists() or (
                folder / "compensated.hdf5"
            ).exists()
            # save_w=True from the flow_options override.
            assert (folder / "w.h5").exists()
            # Stage 1 always persists per-frame valid masks and the
            # streaming temporal average.
            assert (folder / "idx.hdf").exists()
            assert (folder / "temporal_average.npy").exists()
            assert (folder / "status.json").exists()


class TestSessionCompleteExample:
    """docs/api/session.md, section "Complete Example"."""

    @pytest.mark.slow
    def test_complete_session_pipeline_executes(
        self, materialize_session, snippet_runner
    ):
        session_root = materialize_session(
            n_recordings=N_RECORDINGS, shape=RECORDING_SHAPE, root="session"
        )

        with RuntimeContext.use(available_parallelization={"sequential"}):
            ns = snippet_runner("api/session/complete_session_pipeline.py")

        # Stage 1: one output folder per recording.
        output_folders = ns["output_folders"]
        assert len(output_folders) == N_RECORDINGS

        # Stage 2: center resolution and displacement fields.
        assert ns["middle_idx"] == 1
        assert ns["center_file"].name == "recording_001.tif"
        displacement_fields = ns["displacement_fields"]
        assert len(displacement_fields) == N_RECORDINGS
        for w in displacement_fields:
            assert w.shape == (RECORDING_SHAPE[1], RECORDING_SHAPE[2], 2)
        # The center recording gets a zero displacement field.
        assert not np.any(displacement_fields[1])
        for folder in output_folders:
            assert (folder / "w_to_reference.npz").exists()

        # Stage 3: final session mask.
        final_mask = ns["final_mask"]
        assert final_mask.shape == (RECORDING_SHAPE[1], RECORDING_SHAPE[2])
        assert final_mask.dtype == bool
        assert np.any(final_mask)

        # Stage 3 results bundle ("NPZ Bundle Contents" on the page).
        results_dir = session_root / "results"
        assert (results_dir / "final_valid_idx.png").exists()
        npz_path = results_dir / "final_valid_idx.npz"
        assert npz_path.exists()
        bundle = np.load(str(npz_path))
        documented_keys = {
            "final_valid",
            "aligned_valid_masks",
            "per_seq_valid_masks",
            "displacement_fields_u",
            "displacement_fields_v",
            "temporal_averages",
            "compensated_h5_paths",
            "reference_average",
            "middle_idx",
            "aligned_video_paths",
        }
        assert documented_keys <= set(bundle.files)
        np.testing.assert_array_equal(bundle["final_valid"], final_mask)
        assert int(bundle["middle_idx"]) == 1

        # Aligned video export (default TIFF format).
        for i in range(N_RECORDINGS):
            assert (results_dir / f"aligned_recording_{i:03d}.tif").exists()


class TestSessionConfigLoaders:
    """docs/api/session.md, "Configuration File Support" fragments."""

    def test_from_toml_and_from_file_load(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "session").mkdir()
        (tmp_path / "session.toml").write_text(SESSION_TOML)

        from pyflowreg.session.config import SessionConfig

        config = SessionConfig.from_toml("session.toml")
        assert config.root == Path("session")
        assert config.pattern == "recording_*.tif"
        assert config.output_root == Path("compensated")
        assert config.resume is True

        # from_file dispatches on the extension.
        config_auto = SessionConfig.from_file("session.toml")
        assert config_auto == config

    def test_from_yaml_loads(self, tmp_path, monkeypatch):
        pytest.importorskip("yaml", reason="pyyaml required for YAML support")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "session").mkdir()
        (tmp_path / "session.yml").write_text(SESSION_YAML)

        from pyflowreg.session.config import SessionConfig

        config = SessionConfig.from_yaml("session.yml")
        assert config.root == Path("session")
        assert config.pattern == "recording_*.tif"
        assert config.resume is True


class TestSessionAtomicSaveFragments:
    """docs/api/session.md, "atomic_save_npy / atomic_save_npz" fragment."""

    def test_atomic_save_npy_round_trip(self, tmp_path):
        from pyflowreg.session.stage1_compensate import atomic_save_npy

        array = np.arange(6, dtype=np.float64).reshape(2, 3)
        atomic_save_npy(tmp_path / "data.npy", array)

        assert (tmp_path / "data.npy").exists()
        np.testing.assert_array_equal(np.load(str(tmp_path / "data.npy")), array)

    def test_atomic_save_npz_round_trip(self, tmp_path):
        from pyflowreg.session.stage1_compensate import atomic_save_npz

        arr1 = np.array([1, 2, 3])
        arr2 = np.array([[4.0, 5.0], [6.0, 7.0]])
        atomic_save_npz(tmp_path / "data.npz", array1=arr1, array2=arr2)

        assert (tmp_path / "data.npz").exists()
        bundle = np.load(str(tmp_path / "data.npz"))
        np.testing.assert_array_equal(bundle["array1"], arr1)
        np.testing.assert_array_equal(bundle["array2"], arr2)


class TestSessionImportFragments:
    """Import-only fences on docs/api/session.md resolve to real objects."""

    def test_import_fragments_resolve(self):
        from pyflowreg.core.warping import (
            backward_valid_mask,
            compute_batch_valid_masks,
            imregister_binary,
        )
        from pyflowreg.session.config import SessionConfig, get_array_task_id
        from pyflowreg.session.stage1_compensate import (
            atomic_save_npy,
            atomic_save_npz,
            run_stage1,
            run_stage1_array,
        )
        from pyflowreg.session.stage2_between_avgs import run_stage2
        from pyflowreg.session.stage3_valid_mask import run_stage3

        for obj in (
            get_array_task_id,
            run_stage1,
            run_stage1_array,
            atomic_save_npy,
            atomic_save_npz,
            run_stage2,
            run_stage3,
            backward_valid_mask,
            imregister_binary,
            compute_batch_valid_masks,
        ):
            assert callable(obj)
        assert SessionConfig is not None
