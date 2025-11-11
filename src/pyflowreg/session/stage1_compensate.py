"""
Stage 1: Per-recording motion correction.

Discovers input recordings, compensates each sequence independently,
and computes temporal averages for inter-sequence alignment.

Mirrors MATLAB align_full_v3_checkpoint.m Stage 1 logic.
"""

import json
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional

import numpy as np

from pyflowreg.motion_correction.compensate_recording import compensate_recording
from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.session.config import SessionConfig, get_array_task_id


def _build_overrides(
    config: SessionConfig, runtime_override: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge config-level and runtime OFOptions overrides."""
    overrides: Dict[str, Any] = {}

    config_override = config.get_flow_options_override()
    if config_override:
        overrides.update(config_override)

    if runtime_override:
        overrides.update(runtime_override)

    # Session module manages these fields per recording
    overrides.pop("input_file", None)
    overrides.pop("output_path", None)

    return overrides


def discover_input_files(config: SessionConfig) -> List[Path]:
    """
    Discover input files matching pattern in root directory.

    Parameters
    ----------
    config : SessionConfig
        Session configuration

    Returns
    -------
    list[Path]
        Sorted list of matching input files

    Raises
    ------
    ValueError
        If no files found matching pattern
    """
    input_files = sorted(config.root.glob(config.pattern))

    if not input_files:
        raise ValueError(
            f"No files found matching pattern '{config.pattern}' "
            f"in directory {config.root}"
        )

    return input_files


def load_or_create_status(output_folder: Path) -> Dict:
    """
    Load existing status.json or create empty status dict.

    Parameters
    ----------
    output_folder : Path
        Folder to check for status.json

    Returns
    -------
    dict
        Status dictionary with completion flags
    """
    status_path = output_folder / "status.json"

    if status_path.exists():
        with open(status_path, "r") as f:
            return json.load(f)
    else:
        return {}


def save_status(output_folder: Path, status: Dict):
    """
    Atomically save status.json.

    Parameters
    ----------
    output_folder : Path
        Folder to save status.json
    status : dict
        Status dictionary to save
    """
    status_path = output_folder / "status.json"
    temp_path = status_path.with_suffix(".json.tmp")

    # Write to temp file first
    with open(temp_path, "w") as f:
        json.dump(status, f, indent=2)

    # Atomic rename
    temp_path.replace(status_path)


def atomic_save_npy(path: Path, arr: np.ndarray):
    """
    Atomically save numpy array to .npy file.

    Parameters
    ----------
    path : Path
        Target path for .npy file
    arr : ndarray
        Array to save

    Notes
    -----
    Uses write-to-temp then replace() to avoid half-written files on crash.
    Note: np.save() automatically adds .npy extension, so temp file is created
    without extension and np.save adds it.
    """
    # Remove extension for temp file (np.save will add .npy automatically)
    temp_path = path.with_suffix(".tmp")
    np.save(str(temp_path), arr)
    # np.save created temp_path.npy, so move that to final location
    temp_path_with_npy = temp_path.with_suffix(".tmp.npy")
    temp_path_with_npy.replace(path)


def atomic_save_npz(path: Path, **arrays):
    """
    Atomically save numpy arrays to .npz file.

    Parameters
    ----------
    path : Path
        Target path for .npz file
    **arrays : dict
        Arrays to save (keyword arguments)

    Notes
    -----
    Uses write-to-temp then replace() to avoid half-written files on crash.
    Note: np.savez() automatically adds .npz extension, so temp file is created
    without extension and np.savez adds it.
    """
    # Remove extension for temp file (np.savez will add .npz automatically)
    temp_path = path.with_suffix(".tmp")
    np.savez(str(temp_path), **arrays)
    # np.savez created temp_path.npz, so move that to final location
    temp_path_with_npz = temp_path.with_suffix(".tmp.npz")
    temp_path_with_npz.replace(path)


def verify_hdf5_completeness(h5_path: Path, expected_frame_count: int) -> bool:
    """
    Verify HDF5 file has expected dataset and frame count.

    Parameters
    ----------
    h5_path : Path
        Path to HDF5 file
    expected_frame_count : int
        Expected number of frames

    Returns
    -------
    bool
        True if file is complete and valid
    """
    if not h5_path.exists():
        return False

    try:
        from pyflowreg.util.io.factory import get_video_file_reader

        reader = get_video_file_reader(str(h5_path))
        actual_count = len(reader)

        if actual_count != expected_frame_count:
            print(
                f"Warning: {h5_path.name} has {actual_count} frames, "
                f"expected {expected_frame_count}. Will rerun compensation."
            )
            return False

        return True

    except Exception as e:
        print(
            f"Warning: Failed to verify {h5_path.name}: {e}. Will rerun compensation."
        )
        return False


def is_stage1_complete(output_folder: Path, input_file: Optional[Path] = None) -> bool:
    """
    Check if Stage 1 is complete for a sequence.

    Parameters
    ----------
    output_folder : Path
        Output folder for the sequence
    input_file : Path, optional
        Input file to verify frame count (for robustness check)

    Returns
    -------
    bool
        True if compensated.hdf5, temporal_average.npy, and status indicate completion
    """
    # Handle both .hdf5 and .HDF5
    h5_candidates = [
        output_folder / "compensated.hdf5",
        output_folder / "compensated.HDF5",
    ]
    compensated_h5 = next((p for p in h5_candidates if p.exists()), None)

    temporal_avg_npy = output_folder / "temporal_average.npy"
    status = load_or_create_status(output_folder)

    if not (
        compensated_h5 and temporal_avg_npy.exists() and status.get("stage1") == "done"
    ):
        return False

    # Robust check: verify HDF5 has expected frame count
    if input_file is not None:
        try:
            from pyflowreg.util.io.factory import get_video_file_reader

            input_reader = get_video_file_reader(str(input_file))
            expected_count = len(input_reader)

            if not verify_hdf5_completeness(compensated_h5, expected_count):
                return False

        except Exception as e:
            print(
                f"Warning: Could not verify HDF5 completeness for {output_folder.name}: {e}"
            )
            # If we can't verify, err on the side of caution and rerun
            return False

    return True


def compute_and_save_temporal_average(
    compensated_path: Path, output_folder: Path
) -> np.ndarray:
    """
    Compute temporal average from compensated video and save.

    Parameters
    ----------
    compensated_path : Path
        Path to compensated.hdf5 file
    output_folder : Path
        Folder to save temporal_average.npy

    Returns
    -------
    ndarray
        Temporal average array

    Notes
    -----
    Mirrors MATLAB logic (align_full_v3_checkpoint.m lines 66-76):
        vid = vid_reader.read_frames(1:vid_reader.frame_count);
        temporal_avg = mean(vid,4);
    """
    from pyflowreg.util.io.factory import get_video_file_reader

    avg_path = output_folder / "temporal_average.npy"

    if avg_path.exists():
        print(f"Loading existing temporal average from {avg_path.name}")
        return np.load(str(avg_path))

    print(f"Computing temporal average from {compensated_path.name}...")
    vid_reader = get_video_file_reader(str(compensated_path))

    # Stream frames to avoid loading entire video into RAM
    frame_count = len(vid_reader)

    # Safety check
    if frame_count == 0:
        raise ValueError(
            f"{compensated_path.name} has 0 frames - cannot compute temporal average"
        )

    # Read first frame to get shape (VideoReader supports array indexing)
    first_frame = vid_reader[[0]]  # Returns (1, H, W, C) or (1, H, W)
    if first_frame.ndim == 4:  # (T, H, W, C)
        first_frame = first_frame[0]  # Remove batch dimension

    # Ensure float64 for accumulation
    first_frame = first_frame.astype(np.float64)

    # Initialize accumulator
    accumulator = first_frame.copy()

    # Accumulate remaining frames in batches to avoid RAM spike
    batch_size = 1000  # Process 1000 frames at a time
    for start_idx in range(1, frame_count, batch_size):
        end_idx = min(start_idx + batch_size, frame_count)
        batch_indices = list(range(start_idx, end_idx))
        batch = vid_reader[batch_indices]  # Use array indexing

        # Sum over time axis (axis=0)
        accumulator += np.sum(batch, axis=0, dtype=np.float64)

    # Compute average
    temporal_avg = accumulator / frame_count

    # Verify no inf/nan values
    if np.any(~np.isfinite(temporal_avg)):
        raise ValueError(
            f"Temporal average contains inf/nan values. "
            f"This suggests corrupted data in {compensated_path.name}"
        )

    # Save atomically (write-to-temp then replace)
    atomic_save_npy(avg_path, temporal_avg)
    print(f"Saved temporal average to {avg_path.name}")

    return temporal_avg


def compensate_single_recording(
    input_file: Path,
    config: SessionConfig,
    of_options_override: Optional[Dict] = None,
) -> Path:
    """
    Compensate a single recording with resume support.

    Parameters
    ----------
    input_file : Path
        Path to input recording
    config : SessionConfig
        Session configuration
    of_options_override : dict, optional
        Override specific OFOptions parameters

    Returns
    -------
    Path
        Output folder containing results

    Notes
    -----
    Mirrors MATLAB logic (align_full_v3_checkpoint.m lines 30-79):
    - Skip if compensated.hdf5 exists (line 56-60)
    - Run compensate_recording() with configured parameters (line 58)
    - Compute and cache temporal average (lines 66-76)
    """
    output_root, _ = config.resolve_output_paths()
    output_folder = output_root / input_file.stem

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Check resume (with robustness check for HDF5 completeness)
    if config.resume and is_stage1_complete(output_folder, input_file):
        print(f"Skipping {input_file.name} - already complete")
        return output_folder

    # Setup OFOptions
    of_params = {
        "input_file": str(input_file),
        "output_path": str(output_folder),
        "output_format": "HDF5",
        "save_valid_idx": True,
        "save_w": False,  # Can be overridden
        "save_meta_info": True,
        "verbose": False,
        "flow_backend": config.flow_backend,
        "backend_params": config.backend_params,
    }

    # Apply quality setting from config if specified
    if config.stage1_quality_setting is not None:
        of_params["quality_setting"] = config.stage1_quality_setting

    # Apply overrides if provided
    if of_options_override:
        of_params.update(of_options_override)

    options = OFOptions(**of_params)

    # Run compensation
    print(f"\nStarting motion correction of {input_file.name}...")
    start_time = time()

    # Handle both .hdf5 and .HDF5 (case-insensitive filesystems)
    candidates = [
        output_folder / "compensated.hdf5",
        output_folder / "compensated.HDF5",
    ]
    compensated_h5 = next((p for p in candidates if p.exists()), candidates[0])

    if not compensated_h5.exists():
        compensate_recording(options)
        # Re-check both candidates after compensation
        compensated_h5 = next((p for p in candidates if p.exists()), candidates[0])
    else:
        print(f"Found existing {compensated_h5.name}, skipping compensation")

    # Verify output exists
    if not compensated_h5.exists():
        raise RuntimeError(
            "Compensation failed - missing compensated.hdf5 or compensated.HDF5 after compensate_recording()"
        )

    # Compute temporal average
    compute_and_save_temporal_average(compensated_h5, output_folder)

    # Mark as complete
    status = load_or_create_status(output_folder)
    status["stage1"] = "done"
    save_status(output_folder, status)

    elapsed = time() - start_time
    print(f"Done after {elapsed:.2f} seconds")

    return output_folder


def run_stage1(
    config: SessionConfig,
    of_options_override: Optional[Dict] = None,
    task_index: Optional[int] = None,
) -> List[Path]:
    """
    Run Stage 1 for all or subset of recordings.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    of_options_override : dict, optional
        Override specific OFOptions parameters for all recordings
    task_index : int, optional
        If provided, process only this recording (for array jobs).
        Uses 0-based indexing.

    Returns
    -------
    list[Path]
        Output folders for processed recordings

    Notes
    -----
    Mirrors MATLAB align_full_v3_checkpoint.m lines 22-80.

    For array jobs, set task_index to select which recording to process.
    The index is 0-based (unlike SLURM/SGE which are 1-based).

    Examples
    --------
    >>> config = SessionConfig.from_toml("session.toml")
    >>> # Process all recordings
    >>> folders = run_stage1(config)
    >>> # Process only recording at index 2 (for array job)
    >>> folder = run_stage1(config, task_index=2)
    """
    complete_script_timer = time()

    # Create output directory
    output_root, final_results = config.resolve_output_paths()
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover inputs
    input_files = discover_input_files(config)
    print(f"Found {len(input_files)} files matching pattern '{config.pattern}'\n")

    # Filter by task index if provided
    if task_index is not None:
        if task_index < 0 or task_index >= len(input_files):
            raise ValueError(
                f"Task index {task_index} out of range [0, {len(input_files)-1}]"
            )
        input_files = [input_files[task_index]]
        print(f"Processing task {task_index}: {input_files[0].name}\n")

    print("Starting Step 1: Motion correction of each sequence...\n")

    overrides = _build_overrides(config, of_options_override)
    effective_overrides = overrides or None

    # Process each recording
    output_folders = []
    for idx, input_file in enumerate(input_files):
        output_folder = compensate_single_recording(
            input_file, config, effective_overrides
        )
        output_folders.append(output_folder)

        # Progress
        remaining = len(input_files) - (idx + 1)
        if remaining > 0:
            print(f"{remaining} file(s) remaining.\n")

    total_time = time() - complete_script_timer
    print(f"\nStage 1 complete. Total time: {total_time:.2f} seconds")

    return output_folders


def run_stage1_array(
    config: SessionConfig, of_options_override: Optional[Dict] = None
) -> Path:
    """
    Run Stage 1 for array job (auto-detect task ID).

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    of_options_override : dict, optional
        Override specific OFOptions parameters

    Returns
    -------
    Path
        Output folder for processed recording

    Raises
    ------
    RuntimeError
        If no array task ID found in environment

    Notes
    -----
    Automatically detects task ID from SLURM_ARRAY_TASK_ID, SGE_TASK_ID,
    or PBS_ARRAY_INDEX environment variables.

    Converts 1-based scheduler indices to 0-based Python indexing.
    """
    task_id = get_array_task_id()

    if task_id is None:
        raise RuntimeError(
            "No array task ID found in environment. "
            "Set SLURM_ARRAY_TASK_ID, SGE_TASK_ID, or PBS_ARRAY_INDEX."
        )

    # Convert to 0-based index (schedulers are typically 1-based)
    task_index = task_id - 1

    print(f"Array task ID: {task_id} (processing index {task_index})")

    folders = run_stage1(config, of_options_override, task_index=task_index)

    return folders[0] if folders else None
