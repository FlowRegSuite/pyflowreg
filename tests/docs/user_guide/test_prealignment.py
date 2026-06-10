"""
Execute the documentation snippets from docs/user_guide/prealignment.md.

Each test materializes the input files the published example references,
runs the snippet from docs/snippets/user_guide/prealignment/ via
``snippet_runner``, and asserts on the resulting namespace and the
files the pipeline writes.
"""

import numpy as np
import pytest

from pyflowreg._runtime import RuntimeContext
from pyflowreg.util.io import get_video_file_reader

pytestmark = pytest.mark.docs_example


class TestPrealignmentEnablePrealignment:
    """docs/snippets/user_guide/prealignment/enable_prealignment.py"""

    def test_enable_prealignment_executes(
        self, materialize_input, snippet_runner, tmp_path
    ):
        materialize_input("recording.h5")  # default shape (12, 32, 48, 2)

        # Keep the docs test fast and deterministic: restrict executor
        # auto-selection to the sequential executor (which implements
        # cc_initialization like all three executors) instead of letting
        # it spawn a multiprocessing pool. RuntimeContext.use sets a
        # context-local override that is dropped when the block exits.
        with RuntimeContext.use(available_parallelization={"sequential"}):
            ns = snippet_runner("user_guide/prealignment/enable_prealignment.py")

        options = ns["options"]
        assert options.cc_initialization is True
        assert options.cc_hw == 256
        assert options.cc_up == 4

        # Default naming convention writes <output_path>/compensated.<ext>
        output_file = tmp_path / "results" / "compensated.HDF5"
        assert output_file.exists()

        reader = get_video_file_reader(str(output_file))
        registered = reader[:]
        reader.close()
        assert registered.shape == (12, 32, 48, 2)
        assert np.all(np.isfinite(registered))
