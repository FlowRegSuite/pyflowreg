"""
Pytest configuration and fixtures for PyFlowReg tests.
"""

import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np

from tests.fixtures import (
    create_test_video_hdf5,
    create_simple_test_data,
    get_minimal_of_options,
    cleanup_temp_files,
)
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig
from pyflowreg._runtime import RuntimeContext


@pytest.fixture(scope="session", autouse=True)
def initialize_runtime_context():
    """Initialize RuntimeContext for all tests."""
    RuntimeContext.init(force=True)

    # Import parallelization module to trigger executor registration
    import pyflowreg.motion_correction.parallelization  # noqa: F401


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def small_test_video(temp_dir):
    """Create a small test video file for quick testing."""
    shape = (10, 16, 32, 2)  # Small size for fast tests
    video_path = create_test_video_hdf5(
        shape=shape,
        output_path=str(Path(temp_dir) / "small_test.h5"),
        pattern="motion",
        noise_level=0.05,
    )
    yield video_path, shape
    cleanup_temp_files(video_path)


@pytest.fixture(scope="function")
def medium_test_video(temp_dir):
    """Create a medium-sized test video for more comprehensive testing."""
    shape = (50, 32, 64, 2)  # Original requested size
    video_path = create_test_video_hdf5(
        shape=shape,
        output_path=str(Path(temp_dir) / "medium_test.h5"),
        pattern="motion",
        noise_level=0.1,
    )
    yield video_path, shape
    cleanup_temp_files(video_path)


@pytest.fixture(scope="function")
def static_test_video(temp_dir):
    """Create a static test video for baseline testing."""
    shape = (20, 16, 32, 2)
    video_path = create_test_video_hdf5(
        shape=shape,
        output_path=str(Path(temp_dir) / "static_test.h5"),
        pattern="static",
        noise_level=0.02,
    )
    yield video_path, shape
    cleanup_temp_files(video_path)


@pytest.fixture(scope="function")
def test_data_array():
    """Create test data as numpy array without file I/O."""
    shape = (20, 16, 32, 2)
    data = create_simple_test_data(shape)
    return data, shape


@pytest.fixture(scope="function")
def basic_of_options(temp_dir):
    """Create basic OF_options for testing."""
    options = get_minimal_of_options()
    options.output_path = Path(temp_dir)  # Convert to Path
    return options


@pytest.fixture(scope="function")
def fast_of_options(temp_dir):
    """Create very fast OF_options for quick testing."""
    options = get_minimal_of_options()
    options.output_path = Path(temp_dir)  # Convert to Path
    options.levels = 1  # Single level for speed
    options.iterations = 2  # Minimal iterations
    options.alpha = (50.0, 50.0)
    return options


@pytest.fixture(scope="function")
def sequential_config():
    """Create configuration for sequential executor."""
    return RegistrationConfig(n_jobs=1, verbose=True, parallelization="sequential")


@pytest.fixture(scope="function")
def threading_config():
    """Create configuration for threading executor."""
    return RegistrationConfig(n_jobs=2, verbose=True, parallelization="threading")


@pytest.fixture(scope="function")
def multiprocessing_config():
    """Create configuration for multiprocessing executor."""
    return RegistrationConfig(n_jobs=2, verbose=True, parallelization="multiprocessing")


@pytest.fixture(scope="function")
def auto_config():
    """Create configuration with auto-selection of executor."""
    return RegistrationConfig(
        n_jobs=2,
        batch_size=10,
        verbose=True,
        parallelization=None,  # Auto-select
    )


@pytest.fixture(params=["sequential", "threading", "multiprocessing"])
def executor_config(request):
    """Parametrized fixture to test all executor types."""
    return RegistrationConfig(
        n_jobs=2, batch_size=5, verbose=True, parallelization=request.param
    )


@pytest.fixture(scope="function")
def reference_frame():
    """Create a simple reference frame for testing."""
    H, W, C = 16, 32, 2
    ref = np.zeros((H, W, C), dtype=np.float32)

    # Add some structure
    center_y, center_x = H // 2, W // 2
    y, x = np.ogrid[:H, :W]

    for c in range(C):
        # Create circular pattern
        radius = min(H, W) // 4
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
        ref[:, :, c] = mask.astype(np.float32) * 0.8 + 0.2

    return ref


@pytest.fixture(scope="session")
def available_executors():
    """Get list of available executors for testing."""
    return list(RuntimeContext.get_available_parallelization())


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "executor: marks tests that test specific executors"
    )
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        # Mark slow tests
        if "large" in item.name or "comprehensive" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark executor tests
        if "executor" in item.name or item.name.startswith("test_compensate"):
            item.add_marker(pytest.mark.executor)

        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)

        # Mark unit tests (default for most tests)
        if not any(
            marker.name in ["integration", "slow"] for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)
