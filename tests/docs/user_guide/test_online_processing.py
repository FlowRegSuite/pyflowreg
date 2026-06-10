"""
Execute the documentation snippets from docs/user_guide/online_processing.md.

Each test materializes the input files the published example references,
runs the snippet from docs/snippets/user_guide/online_processing/ via
``snippet_runner``, and asserts on the resulting namespace.
"""

import numpy as np
import pytest

from pyflowreg._runtime import RuntimeContext

pytestmark = pytest.mark.docs_example


class TestOnlineProcessingCreateCorrector:
    """docs/snippets/user_guide/online_processing/create_corrector.py"""

    def test_create_corrector_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/online_processing/create_corrector.py")

        flow_reg = ns["flow_reg"]
        # quality_setting is always forced to "fast"
        assert flow_reg.options.quality_setting.value == "fast"
        assert flow_reg.reference_buffer.maxlen == 50
        assert flow_reg.reference_update_interval == 20
        assert flow_reg.reference_update_weight == 0.2
        assert flow_reg.truncate == 4.0
        # No reference set yet, no frames processed
        assert flow_reg.reference_raw is None
        assert flow_reg.frame_count == 0


class TestOnlineProcessingStreamingLoop:
    """docs/snippets/user_guide/online_processing/streaming_loop.py"""

    def test_streaming_loop_executes(self, materialize_input, snippet_runner):
        materialize_input("recording.tif")  # default shape (12, 32, 48, 2)

        # Keep the docs test fast and deterministic: restrict executor
        # auto-selection (used by set_reference's preregistration via
        # compensate_arr) to the sequential executor instead of letting it
        # spawn a multiprocessing pool. RuntimeContext.use sets a
        # context-local override that is dropped when the block exits.
        with RuntimeContext.use(available_parallelization={"sequential"}):
            ns = snippet_runner("user_guide/online_processing/streaming_loop.py")

        video = ns["video"]
        T, H, W, C = video.shape
        assert (T, H, W, C) == (12, 32, 48, 2)

        # Last loop iteration: corrected frame is (H, W, C), flow is (H, W, 2)
        registered = ns["registered"]
        flow = ns["flow"]
        assert registered.shape == (H, W, C)
        assert flow.shape == (H, W, 2)
        assert np.all(np.isfinite(registered))
        assert np.all(np.isfinite(flow))

        # Flow magnitude computed in the loop body is per-pixel
        assert ns["magnitude"].shape == (H, W)

        # The reference was set before streaming, so every frame was corrected
        flow_reg = ns["flow_reg"]
        assert flow_reg.reference_raw is not None
        assert flow_reg.frame_count == T
        assert flow_reg.get_current_flow().shape == (H, W, 2)
