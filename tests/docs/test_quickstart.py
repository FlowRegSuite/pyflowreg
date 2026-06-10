"""
Tests for the documentation snippets rendered on docs/quickstart.md.

Each test materializes the exact input files the published example
references, executes the snippet from ``docs/snippets/quickstart/`` via
the ``snippet_runner`` fixture, and asserts on the resulting namespace
and on the files the snippet writes (read back through pyflowreg's I/O
system).
"""

from pathlib import Path

import numpy as np
import pytest

from pyflowreg._runtime import RuntimeContext
from pyflowreg.util.io import get_video_file_reader

pytestmark = pytest.mark.docs_example

# The published file-based examples let compensate_recording auto-select
# the best executor, which prefers multiprocessing. Spawning a process
# pool (one worker per core, each importing pyflowreg) costs far more
# than the tiny synthetic inputs need, so the tests restrict
# auto-selection to in-process executors. ``RuntimeContext.use`` is a
# context-local override and does not leak into other tests.
_IN_PROCESS_EXECUTORS = {"sequential", "threading"}


class TestQuickstartArrayWorkflow:
    """docs/quickstart.md "Basic Array-Based Workflow"."""

    def test_array_workflow_executes(self, materialize_input, snippet_runner):
        materialize_input("my_video.tif", shape=(24, 32, 48, 2))

        ns = snippet_runner("quickstart/array_workflow.py")

        video = ns["video"]
        assert video.shape == (24, 32, 48, 2)

        # Reference averages frames 10-20 of the loaded video
        reference = ns["reference"]
        assert reference.shape == (32, 48, 2)
        np.testing.assert_allclose(reference, np.mean(video[10:21], axis=0), rtol=1e-6)

        # Scalar alpha is expanded to a 2-tuple (documented on the page)
        options = ns["options"]
        assert options.alpha == (4.0, 4.0)
        assert options.quality_setting.value == "balanced"

        registered = ns["registered"]
        flow = ns["flow"]
        assert registered.shape == video.shape
        assert flow.shape == (24, 32, 48, 2)
        assert np.all(np.isfinite(registered))
        assert np.all(np.isfinite(flow))


class TestQuickstartFileWorkflow:
    """docs/quickstart.md "File-Based Workflow"."""

    def test_file_workflow_executes(self, materialize_input, snippet_runner):
        materialize_input("my_video.h5", shape=(24, 32, 48, 2))

        with RuntimeContext.use(available_parallelization=_IN_PROCESS_EXECUTORS):
            ns = snippet_runner("quickstart/file_workflow.py")

        options = ns["options"]
        assert options.output_format.value == "HDF5"
        assert options.reference_frames == list(range(10, 21))
        # save_w stays enabled (it is reset to False if the displacement
        # writer could not be created)
        assert options.save_w is True

        # Default naming convention writes <output_path>/compensated.<ext>
        out_file = Path("results") / "compensated.HDF5"
        assert out_file.exists()

        reader = get_video_file_reader(str(out_file))
        registered = reader[:]
        reader.close()
        assert registered.shape == (24, 32, 48, 2)
        assert np.all(np.isfinite(registered))

        # save_w=True stores the displacement fields next to the video
        assert (Path("results") / "w.h5").exists()


class TestQuickstartParallelProcessing:
    """docs/quickstart.md "Parallel Processing"."""

    def test_parallel_processing_executes(self, materialize_input, snippet_runner):
        materialize_input("my_video.h5", shape=(24, 32, 48, 2))

        ns = snippet_runner("quickstart/parallel_processing.py")

        config = ns["config"]
        assert config.n_jobs == -1
        assert config.parallelization == "threading"

        # Default output format is MAT, written as compensated.MAT
        out_file = Path("results") / "compensated.MAT"
        assert out_file.exists()

        reader = get_video_file_reader(str(out_file))
        registered = reader[:]
        reader.close()
        assert registered.shape == (24, 32, 48, 2)


class TestQuickstartMultiSession:
    """docs/quickstart.md "Multi-Session Processing"."""

    @pytest.mark.slow
    def test_multi_session_executes(self, materialize_session, snippet_runner):
        session_root = materialize_session(
            n_recordings=3,
            filename_pattern="recording_{:03d}.tif",
            shape=(6, 64, 64, 1),
            root="session",
        )

        with RuntimeContext.use(available_parallelization=_IN_PROCESS_EXECUTORS):
            ns = snippet_runner("quickstart/multi_session.py")

        # Stage 1: one output folder per recording with the expected files
        output_folders = ns["output_folders"]
        assert len(output_folders) == 3
        for folder in output_folders:
            # Paths come back relative to the cwd because root="session/"
            # is relative; resolve before comparing against the harness root.
            assert folder.resolve().parent == (session_root / "compensated").resolve()
            compensated = [
                folder / name for name in ("compensated.HDF5", "compensated.hdf5")
            ]
            assert any(p.exists() for p in compensated)
            assert (folder / "temporal_average.npy").exists()
            assert (folder / "status.json").exists()

        # Stage 2: center index plus one displacement field per recording
        middle_idx = ns["middle_idx"]
        displacements = ns["displacements"]
        assert 0 <= middle_idx < 3
        assert ns["center_file"].name == f"recording_{middle_idx:03d}.tif"
        assert len(displacements) == 3
        for w in displacements:
            assert w.shape == (64, 64, 2)
        # The center recording aligns to itself with zero displacement
        assert np.all(displacements[middle_idx] == 0)

        # Stage 3: session-wide boolean valid mask
        final_mask = ns["final_mask"]
        assert final_mask.shape == (64, 64)
        assert final_mask.dtype == bool
        assert np.any(final_mask)


class TestQuickstartKeyParameters:
    """docs/quickstart.md "Configuration Options" / "Key Parameters"."""

    def test_key_parameters_executes(self, snippet_runner):
        ns = snippet_runner("quickstart/key_parameters.py")

        options = ns["options"]
        # Flow parameters (scalar alpha expands to a 2-tuple)
        assert options.alpha == (4.0, 4.0)
        assert options.iterations == 50
        assert options.levels == 100
        assert options.eta == 0.8
        # min_level=-1 keeps the preset; the default preset is "quality"
        assert options.min_level == -1
        assert options.quality_setting.value == "quality"
        assert options.effective_min_level == 0

        # Preprocessing (a single sigma triple is stored per-channel)
        assert options.bin_size == 1
        assert options.sigma == [[1.0, 1.0, 0.1]]

        # Reference
        assert options.reference_frames == [0, 1, 2, 3, 4]

        # I/O
        assert options.buffer_size == 400
        assert options.output_format.value == "HDF5"
        assert options.save_w is True
