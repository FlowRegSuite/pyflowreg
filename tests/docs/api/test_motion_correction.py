"""
Executable-documentation tests for docs/api/motion_correction.md.

Each test materializes the input files a snippet references, executes the
extracted snippet from ``docs/snippets/api/motion_correction/`` via the
``snippet_runner`` fixture, and asserts on the resulting namespace and the
files the snippet writes.

The tests restrict the runtime to the in-process ``sequential`` executor via
``RuntimeContext.use`` so the published examples (which auto-select the best
executor, preferring multiprocessing) stay fast inside the test suite.
"""

from pathlib import Path

import numpy as np
import pytest

from pyflowreg._runtime import RuntimeContext
from pyflowreg.motion_correction import OFOptions
from pyflowreg.util.io import get_video_file_reader

pytestmark = pytest.mark.docs_example

# Shape of the synthetic input video written by materialize_input.
SHAPE = (12, 32, 48, 2)


class TestMotionCorrectionCallbackExample:
    """docs/api/motion_correction.md, section "Example with Callbacks"."""

    def test_callback_example_executes(self, materialize_input, snippet_runner, capsys):
        materialize_input("recording.tif", shape=SHAPE)

        with RuntimeContext.use(available_parallelization={"sequential"}):
            ns = snippet_runner("api/motion_correction/callback_example.py")

        # Input read through pyflowreg I/O as (T, H, W, C).
        assert ns["video"].shape == SHAPE
        assert ns["reference"].shape == SHAPE[1:]

        # compensate_arr returns (registered, w).
        assert ns["registered"].shape == SHAPE
        assert ns["w"].shape == (SHAPE[0], SHAPE[1], SHAPE[2], 2)
        # Default output_typename is "double".
        assert ns["registered"].dtype == np.float64

        options = ns["options"]
        assert isinstance(options, OFOptions)
        assert options.buffer_size == 20

        # The w_callback ran for every frame of the (single) batch.
        captured = capsys.readouterr()
        assert "Frame 0: mean motion" in captured.out
        assert f"Frame {SHAPE[0] - 1}: mean motion" in captured.out


class TestMotionCorrectionFileWorkflow:
    """docs/api/motion_correction.md, section "File-Based Workflow"."""

    def test_file_workflow_callbacks_executes(self, materialize_input, snippet_runner):
        materialize_input("recording.h5", shape=SHAPE)

        with RuntimeContext.use(available_parallelization={"sequential"}):
            ns = snippet_runner("api/motion_correction/file_workflow_callbacks.py")

        # 12 frames < default buffer_size (400) -> exactly one batch, and
        # the registered callback fired once for it.
        assert ns["monitor"].batch_count == 1

        # Default naming convention writes <output_path>/compensated.HDF5.
        compensated_path = Path("results") / "compensated.HDF5"
        assert compensated_path.exists()
        reader = get_video_file_reader(str(compensated_path))
        compensated = reader[:]
        reader.close()
        assert compensated.shape == SHAPE

        # save_w=True writes displacement fields to <output_path>/w.h5
        # with datasets "u" and "v" (read back as two channels).
        w_path = Path("results") / "w.h5"
        assert w_path.exists()
        reader = get_video_file_reader(str(w_path))
        w = reader[:]
        reader.close()
        assert w.shape == (SHAPE[0], SHAPE[1], SHAPE[2], 2)
