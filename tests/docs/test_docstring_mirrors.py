"""
Runnable mirrors for docstring Examples marked ``# doctest: +SKIP``.

Every Example in ``src/pyflowreg`` that carries ``# doctest: +SKIP`` cannot
be executed by the doctest runner (it needs input files, downloads, or
OpenCV). Each class below mirrors the Examples section of one source file
and executes the same calls for real against tiny synthetic inputs from
the docs test harness (``tests/docs/conftest.py``), so the published
examples stay truthful without large data or long runtimes.

Source file -> mirror class:

- ``src/pyflowreg/__init__.py``                            -> ``TestPyflowregInitDocstringMirror``
- ``src/pyflowreg/motion_correction/__init__.py``          -> ``TestMotionCorrectionInitDocstringMirror``
- ``src/pyflowreg/motion_correction/compensate_arr.py``    -> ``TestCompensateArrDocstringMirror``
- ``src/pyflowreg/motion_correction/compensate_recording.py`` -> ``TestCompensateRecordingDocstringMirror``
- ``src/pyflowreg/util/io/factory.py``                     -> ``TestIOFactoryDocstringMirror``
- ``src/pyflowreg/util/download.py``                       -> ``TestDownloadDocstringMirror``
- ``src/pyflowreg/session/stage1_compensate.py``           -> ``TestSessionStage1DocstringMirror``
- ``src/pyflowreg/session/stage2_between_avgs.py``         -> ``TestSessionStage2DocstringMirror``
- ``src/pyflowreg/session/stage3_valid_mask.py``           -> ``TestSessionStage3DocstringMirror``
- ``src/pyflowreg/core/diso_optical_flow.py``              -> ``TestDisoOpticalFlowDocstringMirror``

Speed adjustments versus the published examples (intent is unchanged):
tiny synthetic inputs, ``quality_setting="fast"`` where the example used
defaults, explicit small ``reference_frames`` (the default ``50:500``
exceeds the tiny clips), and the sequential executor to avoid Windows
process-spawn overhead. Examples that explicitly teach
``quality_setting="balanced"`` keep it.
"""

from pathlib import Path

import numpy as np
import pytest

from pyflowreg.motion_correction import (
    OFOptions,
    compensate_arr,
    compensate_recording,
)
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig
from pyflowreg.session.config import SessionConfig
from pyflowreg.session.stage1_compensate import run_stage1
from pyflowreg.session.stage2_between_avgs import run_stage2
from pyflowreg.session.stage3_valid_mask import run_stage3
from pyflowreg.util.download import DEMO_DATA_URLS, download_demo_data
from pyflowreg.util.io.factory import get_video_file_reader

pytestmark = pytest.mark.docs_example

# tests/docs/test_docstring_mirrors.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Source files (relative to src/pyflowreg) currently carrying
# ``# doctest: +SKIP`` examples. Each entry must have a mirror class in
# this module; the inventory test below fails when a new +SKIP example
# appears without a mirror.
DOCTEST_SKIP_FILES = frozenset(
    {
        "__init__.py",
        "core/diso_optical_flow.py",
        "motion_correction/__init__.py",
        "motion_correction/compensate_arr.py",
        "motion_correction/compensate_recording.py",
        "session/stage1_compensate.py",
        "session/stage2_between_avgs.py",
        "session/stage3_valid_mask.py",
        "util/download.py",
        "util/io/factory.py",
    }
)

# Mirrors SessionConfig.from_toml("session.toml") from the session stage
# docstrings. The session root and recordings are created by the
# ``materialize_session`` fixture (3 recordings, (6, 64, 64, 1) each);
# flow_options keeps Stage 1 fast on the 6-frame clips.
SESSION_TOML = """\
root = "session"
pattern = "recording_*.tif"
stage1_quality_setting = "fast"
n_workers = 1

[flow_options]
reference_frames = [0, 1, 2]
"""


def _sequential_config():
    """Sequential executor: keeps docstring mirrors fast and deterministic."""
    return RegistrationConfig(parallelization="sequential")


def _write_session_toml(tmp_path):
    """Write the session.toml the session stage docstrings load."""
    (tmp_path / "session.toml").write_text(SESSION_TOML, encoding="utf-8")


class TestDocstringSkipInventory:
    """Guard: every ``doctest: +SKIP`` example has a mirror in this module."""

    def test_doctest_skip_files_match_mirrors(self):
        src_root = REPO_ROOT / "src" / "pyflowreg"
        found = set()
        for path in src_root.rglob("*.py"):
            if "doctest: +SKIP" in path.read_text(encoding="utf-8"):
                found.add(path.relative_to(src_root).as_posix())

        new = found - DOCTEST_SKIP_FILES
        gone = DOCTEST_SKIP_FILES - found
        assert found == DOCTEST_SKIP_FILES, (
            f"doctest +SKIP inventory changed. New files without a mirror "
            f"class in tests/docs/test_docstring_mirrors.py: {sorted(new)}; "
            f"files no longer carrying +SKIP (remove from "
            f"DOCTEST_SKIP_FILES and drop the mirror): {sorted(gone)}"
        )


class TestPyflowregInitDocstringMirror:
    """Mirrors the Quick Start example in ``src/pyflowreg/__init__.py``."""

    def test_quick_start_executes(self):
        import pyflowreg
        from pyflowreg.motion_correction import compensate_arr, OFOptions

        # Small synthetic video (T, H, W) and a reference frame
        video = np.random.rand(10, 32, 32).astype(np.float32)
        reference = video[:5].mean(axis=0)

        # Set up options (the example explicitly teaches "balanced")
        options = OFOptions(quality_setting="balanced")

        # Register video (array-based)
        registered, flow = compensate_arr(
            video,
            reference,
            options,
            registration_config=_sequential_config(),
        )

        assert registered.shape == video.shape
        assert flow.shape == (10, 32, 32, 2)
        assert np.all(np.isfinite(flow))
        # Module docstring: core exposes get_displacement at package level.
        assert callable(pyflowreg.get_displacement)


class TestMotionCorrectionInitDocstringMirror:
    """Mirrors the Quick Start in ``src/pyflowreg/motion_correction/__init__.py``."""

    def test_quick_start_executes(self):
        from pyflowreg.motion_correction import compensate_arr, OFOptions

        video = np.random.rand(10, 32, 32).astype(np.float32)
        reference = video[:5].mean(axis=0)
        options = OFOptions(quality_setting="balanced")
        registered, flow = compensate_arr(
            video,
            reference,
            options,
            registration_config=_sequential_config(),
        )

        assert registered.shape == video.shape
        assert flow.shape == (10, 32, 32, 2)

    def test_documented_exports_exist(self):
        """The module docstring's Main Functions/Classes are all importable."""
        from pyflowreg.motion_correction import (
            BatchMotionCorrector,
            FlowRegLive,
            OFOptions,
            compensate_arr,
            compensate_recording,
        )

        assert callable(compensate_arr)
        assert callable(compensate_recording)
        assert OFOptions is not None
        assert BatchMotionCorrector is not None
        assert FlowRegLive is not None


class TestCompensateArrDocstringMirror:
    """Mirrors ``compensate_arr`` Examples (progress callback variant)."""

    def test_compensate_arr_progress_callback_executes(self):
        # Docstring uses (100, 256, 256, 2); mirrored with a tiny clip.
        video = np.random.rand(8, 32, 48, 2)
        reference = np.mean(video[:4], axis=0)

        progress_calls = []

        def progress(current, total):
            progress_calls.append((current, total))

        # Docstring used default options; "fast" keeps the mirror quick.
        registered, flow = compensate_arr(
            video,
            reference,
            OFOptions(quality_setting="fast"),
            progress_callback=progress,
            registration_config=_sequential_config(),
        )

        assert registered.shape == video.shape
        # output_typename defaults to "double" -> float64 (documented).
        assert registered.dtype == np.float64
        assert flow.shape == (8, 32, 48, 2)
        assert np.all(np.isfinite(flow))
        # Sequential executor reports frame-wise cumulative progress.
        assert progress_calls, "progress callback was never invoked"
        assert progress_calls[-1] == (video.shape[0], video.shape[0])


class TestCompensateRecordingDocstringMirror:
    """Mirrors ``compensate_recording`` Examples (file-based pipeline)."""

    def test_compensate_recording_executes(self, materialize_input):
        # Harness writes video.h5 with shape (12, 32, 48, 2) into tmp cwd.
        materialize_input("video.h5")

        options = OFOptions(
            input_file="video.h5",
            output_path="results",
            quality_setting="balanced",
            # Default reference_frames (50:500) exceeds the 12-frame clip.
            reference_frames=list(range(4)),
        )
        reference = compensate_recording(options, config=_sequential_config())

        # Returns the raw reference frame cast to float64 (documented).
        assert reference.shape == (32, 48, 2)
        assert reference.dtype == np.float64

        # Compensated video written to <output_path>/compensated.<ext>
        # (default output_format is MAT).
        out_file = Path("results") / "compensated.MAT"
        assert out_file.exists()
        reader = get_video_file_reader(str(out_file))
        try:
            assert reader[:].shape == (12, 32, 48, 2)
        finally:
            reader.close()

        # save_meta_info defaults to True -> statistics + reference saved.
        assert (Path("results") / "statistics.npz").exists()
        assert (Path("results") / "reference_frame.npy").exists()


class TestIOFactoryDocstringMirror:
    """Mirrors ``get_video_file_reader`` Examples in ``util/io/factory.py``."""

    def test_get_video_file_reader_executes(self, materialize_input):
        materialize_input("video.tif")

        reader = get_video_file_reader("video.tif")
        video = reader[:]  # (T, H, W, C)
        reader.close()

        assert video.shape == (12, 32, 48, 2)


@pytest.mark.demo_data
class TestDownloadDocstringMirror:
    """Mirrors ``download_demo_data`` Examples in ``util/download.py``.

    Uses the ``demo_data_file`` fixture (tests/conftest.py): the test is
    skipped when the demo file is not cached in ``data/`` and
    ``PYFLOWREG_TEST_DEMO_DATA=1`` is not set. With a cached file,
    ``download_demo_data`` takes its documented short-circuit path and
    returns the existing file without network access.
    """

    @pytest.mark.parametrize("demo_name", ["jupiter.tiff", "synth_frames.h5"])
    def test_download_demo_data_executes(self, demo_data_file, demo_name):
        assert demo_name in DEMO_DATA_URLS
        cached = demo_data_file(demo_name)

        path = download_demo_data(demo_name, output_folder=cached.parent)

        assert path == cached
        assert path.exists()
        assert path.stat().st_size > 0


@pytest.mark.slow
class TestSessionStage1DocstringMirror:
    """Mirrors ``run_stage1`` Examples in ``session/stage1_compensate.py``."""

    def test_run_stage1_executes(self, materialize_session, tmp_path):
        materialize_session()
        _write_session_toml(tmp_path)

        config = SessionConfig.from_toml("session.toml")
        # Process all recordings
        folders = run_stage1(config)

        assert len(folders) == 3
        for folder in folders:
            assert (folder / "compensated.hdf5").exists() or (
                folder / "compensated.HDF5"
            ).exists()
            assert (folder / "temporal_average.npy").exists()
            assert (folder / "idx.hdf").exists()

    def test_run_stage1_task_index_executes(self, materialize_session, tmp_path):
        materialize_session()
        _write_session_toml(tmp_path)

        config = SessionConfig.from_toml("session.toml")
        # Process only recording at index 2 (for array job)
        folders = run_stage1(config, task_index=2)

        assert len(folders) == 1
        assert folders[0].name == "recording_002"
        assert (folders[0] / "temporal_average.npy").exists()


@pytest.mark.slow
class TestSessionStage2DocstringMirror:
    """Mirrors ``run_stage2`` Examples in ``session/stage2_between_avgs.py``."""

    def test_run_stage2_executes(self, materialize_session, tmp_path):
        materialize_session()
        _write_session_toml(tmp_path)

        config = SessionConfig.from_toml("session.toml")
        run_stage1(config)  # Stage 2 consumes Stage 1 temporal averages

        middle_idx, center_file, displacements = run_stage2(config)

        # Lexicographic middle of 3 recordings (0-based).
        assert middle_idx == 1
        assert Path(center_file).name == "recording_001.tif"
        assert len(displacements) == 3
        for w in displacements:
            assert w.shape == (64, 64, 2)
        # Center recording has zero displacement (documented).
        assert np.all(displacements[middle_idx] == 0)

        # Results persisted as w_to_reference.npz in each output folder.
        output_root, _ = config.resolve_output_paths()
        for i in range(3):
            assert (output_root / f"recording_{i:03d}" / "w_to_reference.npz").exists()


@pytest.mark.slow
class TestSessionStage3DocstringMirror:
    """Mirrors ``run_stage3`` Examples in ``session/stage3_valid_mask.py``."""

    def test_run_stage3_executes(self, materialize_session, tmp_path):
        materialize_session()
        _write_session_toml(tmp_path)

        config = SessionConfig.from_toml("session.toml")
        run_stage1(config)
        run_stage2(config)

        # Docstring call: stage 3 loads middle_idx/displacements itself.
        final_mask = run_stage3(config)

        assert final_mask.shape == (64, 64)
        assert final_mask.dtype == bool
        assert np.any(final_mask), "final session mask should keep valid pixels"

        _, final_results = config.resolve_output_paths()
        assert (final_results / "final_valid_idx.png").exists()
        saved = np.load(final_results / "final_valid_idx.npz")
        np.testing.assert_array_equal(saved["final_valid"], final_mask)


class TestDisoOpticalFlowDocstringMirror:
    """Mirrors the ``DisoOF`` Examples in ``core/diso_optical_flow.py``."""

    def test_diso_call_executes(self):
        pytest.importorskip("cv2")
        from pyflowreg.core.diso_optical_flow import DisoOF

        diso = DisoOF()
        fixed = np.zeros((64, 64), dtype=np.float32)
        fixed[24:40, 24:40] = 1.0
        moving = np.roll(fixed, 2, axis=1)
        w = diso(fixed, moving)

        # Docstring promises an (H, W, 2) float32 displacement field.
        assert w.shape == (64, 64, 2)
        assert w.dtype == np.float32
        assert np.all(np.isfinite(w))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
