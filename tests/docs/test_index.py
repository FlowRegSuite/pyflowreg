"""
Tests for the documentation snippets rendered on docs/index.md.

Each test materializes the exact input file the published example
references, executes the snippet from ``docs/snippets/index/`` via the
``snippet_runner`` fixture, and asserts on the resulting namespace.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.docs_example


class TestIndexGettingStarted:
    """docs/index.md "Getting Started" (snippets/index/getting_started.py)."""

    def test_getting_started_executes(self, materialize_input, snippet_runner):
        materialize_input("my_video.tif", shape=(24, 32, 48, 1))

        ns = snippet_runner("index/getting_started.py")

        video = ns["video"]
        assert video.shape == (24, 32, 48, 1)

        # Reference averages frames 10-20 of the loaded video
        reference = ns["reference"]
        assert reference.shape == (32, 48, 1)
        np.testing.assert_allclose(reference, np.mean(video[10:21], axis=0), rtol=1e-6)

        assert ns["options"].quality_setting.value == "balanced"

        registered = ns["registered"]
        flow = ns["flow"]
        assert registered.shape == video.shape
        assert flow.shape == (24, 32, 48, 2)
        assert np.all(np.isfinite(registered))
        assert np.all(np.isfinite(flow))
