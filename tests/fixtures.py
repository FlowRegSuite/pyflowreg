"""
Test fixtures and utilities for PyFlowReg tests.
"""

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np

from pyflowreg.util.io.factory import get_video_file_writer


def create_test_video_hdf5(
    shape: Tuple[int, int, int, int] = (50, 32, 64, 2),
    output_path: str = None,
    pattern: str = "motion",
    noise_level: float = 0.1,
) -> str:
    """
    Create a test HDF5 video file with simulated motion.

    Args:
        shape: Video shape in TxHxWxC format (T=time, H=height, W=width, C=channels)
        output_path: Output file path. If None, creates temporary file.
        pattern: Motion pattern ('motion', 'static', 'random')
        noise_level: Amount of noise to add (0.0 to 1.0)

    Returns:
        Path to created HDF5 file
    """
    T, H, W, C = shape

    if output_path is None:
        temp_dir = tempfile.mkdtemp()
        output_path = str(Path(temp_dir) / "test_video.h5")

    # Generate test frames based on pattern
    if pattern == "motion":
        # Create frames with simulated translational motion
        frames = np.zeros(shape, dtype=np.float32)

        # Create a simple object to move
        center_y, center_x = H // 2, W // 2
        radius = min(H, W) // 8

        for t in range(T):
            for c in range(C):
                # Simulate motion - circular pattern
                offset_x = int(radius * 0.5 * np.cos(2 * np.pi * t / T))
                offset_y = int(radius * 0.5 * np.sin(2 * np.pi * t / T))

                # Create simple circular object
                y, x = np.ogrid[:H, :W]
                obj_y = center_y + offset_y
                obj_x = center_x + offset_x

                # Create circular object
                mask = (x - obj_x) ** 2 + (y - obj_y) ** 2 <= radius**2
                frames[t, :, :, c] = mask.astype(np.float32) * 0.8

                # Add some background texture
                frames[t, :, :, c] += 0.2 * np.random.random((H, W))

                # Add noise
                if noise_level > 0:
                    frames[t, :, :, c] += noise_level * np.random.random((H, W))

    elif pattern == "static":
        # Create static frames with just noise
        frames = np.ones(shape, dtype=np.float32) * 0.5
        if noise_level > 0:
            frames += noise_level * np.random.random(shape)

    elif pattern == "random":
        # Create random frames
        frames = np.random.random(shape).astype(np.float32)

    else:
        # Default zeros
        frames = np.zeros(shape, dtype=np.float32)

    # Ensure values are in [0, 1] range
    frames = np.clip(frames, 0.0, 1.0)

    # Convert to uint16 for more realistic microscopy data
    frames = (frames * 65535).astype(np.uint16)

    # Write using PyFlowReg's video writer
    writer = get_video_file_writer(output_path, "HDF5")

    try:
        # Write frames in batches to simulate real usage
        batch_size = 10
        for start_idx in range(0, T, batch_size):
            end_idx = min(start_idx + batch_size, T)
            batch = frames[start_idx:end_idx]
            writer.write_frames(batch)
    finally:
        writer.close()

    return output_path


def create_simple_test_data(
    shape: Tuple[int, int, int, int] = (50, 32, 64, 2),
) -> np.ndarray:
    """
    Create simple test data as numpy array without file I/O.

    Args:
        shape: Data shape in TxHxWxC format

    Returns:
        Test data array
    """
    T, H, W, C = shape

    # Create frames with a simple moving pattern
    frames = np.zeros(shape, dtype=np.float32)

    for t in range(T):
        for c in range(C):
            # Simple gradient that shifts over time
            y, x = np.mgrid[:H, :W]
            shift = t * 2  # Pixels shift per frame
            pattern = np.sin((x + shift) * 2 * np.pi / W) * 0.5 + 0.5
            frames[t, :, :, c] = pattern

    return frames


def get_minimal_of_options():
    """
    Create minimal OF_options for testing.

    Returns:
        Basic OFOptions configuration suitable for testing
    """
    from pyflowreg.motion_correction.OF_options import OFOptions

    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()

    return OFOptions(
        input_file="dummy.h5",  # Will be overridden in tests
        output_path=temp_dir,
        quality_setting="fast",  # Use fast for testing
        alpha=100.0,
        levels=2,  # Fewer levels for faster testing
        iterations=5,  # Fewer iterations for faster testing
        eta=0.95,
        update_lag=5,
        a_smooth=1.0,
        a_data=1.0,
        sigma=[[1.0, 1.0, 0.5], [1.0, 1.0, 0.5]],  # [sx, sy, st] per channel
        weight=[1.0, 1.0],  # Equal weight for 2 channels
        min_level=0,
        interpolation_method="cubic",
        update_reference=False,
        update_initialization_w=True,
        save_w=False,  # Don't save flow fields by default for testing
        save_meta_info=False,  # Don't save metadata by default for testing
        channel_normalization="joint",
    )


def cleanup_temp_files(*file_paths):
    """
    Clean up temporary files and directories.

    Args:
        *file_paths: Paths to files or directories to clean up
    """
    import shutil

    for path in file_paths:
        if path and Path(path).exists():
            try:
                path_obj = Path(path)
                if path_obj.is_file():
                    path_obj.unlink()
                elif path_obj.is_dir():
                    shutil.rmtree(path_obj)
            except Exception as e:
                print(f"Warning: Could not clean up {path}: {e}")
