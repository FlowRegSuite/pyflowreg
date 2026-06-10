"""
Tests for the code examples in docs/user_guide/parallelization.md.

Each test materializes the input files a snippet references (if any),
executes the snippet from ``docs/snippets/user_guide/parallelization/``
via the ``snippet_runner`` fixture, and asserts on the resulting module
namespace and produced output files.
"""

import pytest

from pyflowreg.util.io import get_video_file_reader

pytestmark = pytest.mark.docs_example


class TestParallelizationSequentialRun:
    """docs/user_guide/parallelization.md -- "Sequential" full-run example."""

    def test_sequential_run_executes(self, materialize_input, snippet_runner):
        video_path = materialize_input("video.h5", shape=(12, 32, 48, 2))

        snippet_runner("user_guide/parallelization/sequential_run.py")

        output_file = video_path.parent / "results" / "compensated.HDF5"
        assert output_file.exists(), "compensate_recording did not write output"

        reader = get_video_file_reader(str(output_file))
        registered = reader[:]
        reader.close()
        assert registered.shape == (12, 32, 48, 2)


class TestParallelizationThreadingConfig:
    """docs/user_guide/parallelization.md -- "Threading" executor selection."""

    def test_threading_config_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/parallelization/threading_config.py")

        config = ns["config"]
        assert config.parallelization == "threading"
        assert config.n_jobs == 8


class TestParallelizationMultiprocessingConfig:
    """docs/user_guide/parallelization.md -- "Multiprocessing" executor selection."""

    def test_multiprocessing_config_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/parallelization/multiprocessing_config.py")

        config = ns["config"]
        assert config.parallelization == "multiprocessing"
        assert config.n_jobs == -1


class TestParallelizationBufferSize:
    """docs/user_guide/parallelization.md -- "Buffer Size" configuration."""

    def test_buffer_size_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/parallelization/buffer_size.py")

        assert ns["options"].buffer_size == 400


class TestParallelizationExecutorRegistry:
    """docs/user_guide/parallelization.md -- "Executor Registration" introspection."""

    def test_executor_registry_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/parallelization/executor_registry.py")

        available = ns["available"]
        assert isinstance(available, set)
        # Sequential and threading are always detected by RuntimeContext.
        assert "sequential" in available
        assert "threading" in available
