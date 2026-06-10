"""
Tests for docs/user_guide/workflows.md.

Each test materializes the input file(s) a snippet references (via the
harness fixtures in ``tests/docs/conftest.py``), executes the extracted
snippet from ``docs/snippets/user_guide/workflows/`` with
``snippet_runner``, and asserts on the namespace the published example
creates. Callback-invocation counts, output shapes, and the
configuration-option assertions were ported from the removed
``tests/docs/test_workflows.py``.
"""

import numpy as np
import pytest

from pyflowreg.motion_correction import compensate_arr
from pyflowreg.motion_correction.OF_options import (
    OFOptions,
    OutputFormat,
    QualitySetting,
)
from pyflowreg.util.io import get_video_file_reader

pytestmark = pytest.mark.docs_example

# Snippets build their reference from video[10:21] (array workflow) or
# reference_frames=list(range(10, 21)) (file workflow), so inputs need
# at least 21 frames.
VIDEO_SHAPE = (24, 32, 48, 2)


class TestWorkflowsArrayBasic:
    """docs/user_guide/workflows.md, "Array-Based Workflow" / "Basic Usage"."""

    def test_array_basic_executes(self, materialize_input, snippet_runner):
        materialize_input("my_video.h5", shape=VIDEO_SHAPE)

        ns = snippet_runner("user_guide/workflows/array_basic.py")

        T, H, W, C = VIDEO_SHAPE
        assert ns["video"].shape == VIDEO_SHAPE
        assert ns["reference"].shape == (H, W, C)
        # Documented returns: registered keeps the input shape, flow is
        # (T, H, W, 2) with (u, v) components.
        assert ns["registered"].shape == VIDEO_SHAPE
        assert ns["flow"].shape == (T, H, W, 2)
        assert np.all(np.isfinite(ns["flow"]))


class TestWorkflowsCallbacksBasic:
    """docs/user_guide/workflows.md, "Basic Callback Usage"."""

    def test_callbacks_basic_executes(self, materialize_input, snippet_runner):
        materialize_input("my_video.h5", shape=VIDEO_SHAPE)

        ns = snippet_runner("user_guide/workflows/callbacks_basic.py")

        T, H, W, C = VIDEO_SHAPE
        # w_callback fired and saw every frame exactly once.
        assert len(ns["motion_per_frame"]) == T
        # registered_callback fired and the batches tile the recording.
        corrected_batches = ns["corrected_batches"]
        assert len(corrected_batches) > 0
        assert all(batch.ndim == 4 for batch in corrected_batches)
        assert sum(batch.shape[0] for batch in corrected_batches) == T
        # Return values are still produced alongside the callbacks.
        assert ns["registered"].shape == VIDEO_SHAPE
        assert ns["w"].shape == (T, H, W, 2)


class TestWorkflowsCallbacksBatchCorrector:
    """docs/user_guide/workflows.md, "Callbacks in the File-Based Workflow"."""

    def test_callbacks_batch_corrector_executes(
        self, materialize_input, snippet_runner, tmp_path
    ):
        materialize_input("raw_video.h5", shape=VIDEO_SHAPE)

        ns = snippet_runner("user_guide/workflows/callbacks_batch_corrector.py")

        T, H, W, C = VIDEO_SHAPE
        # Both registered callbacks were invoked during run().
        assert len(ns["motion_per_frame"]) == T
        corrected_batches = ns["corrected_batches"]
        assert len(corrected_batches) > 0
        assert sum(batch.shape[0] for batch in corrected_batches) == T

        # The file-based pipeline wrote the compensated recording, readable
        # through the PyFlowReg I/O system.
        compensated = tmp_path / "results" / "compensated.HDF5"
        assert compensated.exists()
        reader = get_video_file_reader(str(compensated))
        try:
            assert reader[:].shape == VIDEO_SHAPE
        finally:
            reader.close()


class TestWorkflowsCallbacksLiveProcessor:
    """docs/user_guide/workflows.md, "Using Callbacks" (LiveProcessor)."""

    def test_callbacks_live_processor_executes(self, materialize_input, snippet_runner):
        materialize_input("my_video.h5", shape=VIDEO_SHAPE)

        ns = snippet_runner("user_guide/workflows/callbacks_live_processor.py")

        statistics = ns["processor"].statistics
        assert len(statistics) > 0
        assert all("mean_motion" in stats for stats in statistics)
        # Batches tile the whole recording.
        assert statistics[0]["start"] == 0
        assert statistics[-1]["end"] == VIDEO_SHAPE[0]
        assert ns["registered"].shape == VIDEO_SHAPE


class TestWorkflowsFileBasic:
    """docs/user_guide/workflows.md, "File-Based Workflow" / "Basic Usage"."""

    def test_file_basic_executes(self, materialize_input, snippet_runner, tmp_path):
        materialize_input("raw_video.h5", shape=VIDEO_SHAPE)

        ns = snippet_runner("user_guide/workflows/file_basic.py")

        assert ns["options"].save_w is True

        # Output files documented in the "Output Files" section.
        results = tmp_path / "results"
        compensated = results / "compensated.HDF5"
        assert compensated.exists()
        assert (results / "w.h5").exists()  # save_w=True
        assert (results / "statistics.npz").exists()  # save_meta_info default
        assert (results / "reference_frame.npy").exists()

        reader = get_video_file_reader(str(compensated))
        try:
            assert reader[:].shape == VIDEO_SHAPE
        finally:
            reader.close()


class TestWorkflowsBufferSize:
    """docs/user_guide/workflows.md, "Buffer Size Selection"."""

    def test_buffer_size_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/workflows/buffer_size.py")

        options_display = ns["options_display"]
        assert options_display.buffer_size == 10
        assert options_display.output_format == OutputFormat.NULL

        options_batch = ns["options_batch"]
        assert options_batch.buffer_size == 100
        assert options_batch.output_format == OutputFormat.HDF5

        # Ported from the removed test_workflows.py: compensate_arr is
        # always in-memory and returns arrays regardless of output_format
        # (the page states this in "Output Formats").
        video = np.random.default_rng(0).random((4, 16, 16)).astype(np.float32)
        reference = np.mean(video[:2], axis=0)
        options = options_display.model_copy(
            update={"quality_setting": QualitySetting.FAST}
        )
        registered, w = compensate_arr(video, reference, options)
        assert registered.shape == video.shape
        assert w.shape == (video.shape[0], video.shape[1], video.shape[2], 2)


class TestWorkflowsKeyOptions:
    """docs/user_guide/workflows.md, "Key Configuration Options"."""

    def test_key_options_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/workflows/key_options.py")

        options = ns["options"]
        assert isinstance(options, OFOptions)
        assert options.output_format == OutputFormat.NULL
        assert options.buffer_size == 20
        assert options.save_w is True
        assert options.levels == 5
        assert options.iterations == 50
        assert options.quality_setting == QualitySetting.BALANCED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
